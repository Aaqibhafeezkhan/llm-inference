# Tensor and pipeline parallelism

Two orthogonal ways to split a model too large (or too latency-sensitive) for
one GPU. Tensor parallelism (TP) splits individual matmuls within a layer
across devices — low latency, high communication frequency. Pipeline
parallelism (PP) splits layers across devices — infrequent communication, but
introduces idle "bubble" time. Production multi-node serving typically
combines both: TP within a node (NVLink), PP across nodes (slower fabric).

## Tensor parallelism — Megatron-LM style sharding

**MLP block** (`X -> W1 -> GeLU -> W2 -> out`, hidden `h`, intermediate `4h`):
shard `W1` column-wise across `N` GPUs — each GPU holds `4h/N` columns and
computes `GeLU(X @ W1_i)` independently, no communication needed (GeLU is
elementwise, safe to apply per-shard). Shard `W2` row-wise, aligned with
`W1`'s column split — each GPU computes a partial output
`GeLU(X @ W1_i) @ W2_i`, shape `[batch, seq, h]`, which is only a *partial
sum* toward the true output. One **all-reduce** (sum) across the `N` GPUs
produces the correct final output. Total communication: one all-reduce per
MLP block.

**Attention block**: shard attention heads across GPUs — each GPU owns
`n_heads/N` heads and computes Q/K/V projections, attention, per-head output
entirely locally (heads are independent, no cross-head interaction in
standard MHA/GQA attention math). The output projection `W_O` is sharded
row-wise like `W2` above, so again each GPU produces a partial sum and one
all-reduce combines them. Total communication: one all-reduce per attention
block.

Net: **2 all-reduces per transformer layer**, each over an activation tensor
of shape `[batch, seq, hidden]`.

## All-reduce cost

Ring all-reduce across `N` GPUs moving `S` bytes per GPU costs, in the
standard bandwidth-bound model:

```
data_moved_per_gpu = 2 * (N-1)/N * S
time = data_moved_per_gpu / B
```

`B` = achievable interconnect bandwidth (NVLink within a node, far higher than
cross-node Ethernet/InfiniBand).

**Worked**: `N=8` (single-node NVLink, `B ~ 300 GB/s` per-direction achievable),
batch=8, seq=2048, hidden=8192, fp16:

```
S = 8 * 2048 * 8192 * 2 bytes = 268,435,456 bytes = 256 MB
time = 2 * (7/8) * 256 MB / 300 GB/s = 448 MB / 300 GB/s ≈ 1.49 ms
```

At 2 all-reduces/layer x 80 layers (70B-scale depth): `160 * 1.49 ms ≈ 238 ms`
of pure communication per forward pass — comparable to or larger than a
decode-step latency budget (tens of ms/token TPOT target). This is why TP
degree is normally capped at the GPU count within one NVLink domain (8 on a
DGX-class node): the same all-reduce over cross-node fabric (~25-100x lower
bandwidth than NVLink) would cost proportionally more and blow the latency
budget entirely.

## Pipeline parallelism — layer sharding and bubbles

Split the `L` transformer layers into `P` sequential stages, one (or more) per
device — stage 1 holds layers `1..L/P`, stage 2 holds `L/P+1..2L/P`, etc. A
batch is split into `M` microbatches that flow through the stages in a
pipeline: while stage 2 processes microbatch 1, stage 1 can already start
microbatch 2. Communication is just the activation handoff at each stage
boundary — one tensor transfer per microbatch per boundary, far less frequent
than TP's per-layer all-reduce, and point-to-point rather than collective.

**The bubble**: the pipeline needs `P-1` steps to fill (stage `P` can't start
until stage 1's output has propagated through all prior stages) and `P-1`
steps to drain at the end. During fill and drain, some stages are idle.

Exact bubble fraction (GPipe schedule, forward-then-backward per microbatch
treated as one unit of work per stage):

```
bubble_fraction = (P - 1) / (M + P - 1)
```

Derivation sketch: total wall-clock time is `M + P - 1` stage-time-units;
useful work performed is `M * P` stage-time-units out of `P * (M + P - 1)`
available device-time-units; `1 - M/(M+P-1) = (P-1)/(M+P-1)`.

**Worked**: `P=4` stages.
`M=8`: `bubble = 3/11 ≈ 27.3%` idle.
`M=32`: `bubble = 3/35 ≈ 8.6%` idle.
`M=64`: `bubble = 3/67 ≈ 4.5%` idle.

Bubble amortizes with more microbatches per pipeline flush — this is the
whole reason to prefer many small microbatches over few large ones under PP,
independent of any memory concern.

**Interleaved (virtual pipeline) schedules**: Megatron's interleaved 1F1B
assigns each device several non-contiguous chunks of layers (`v` virtual
stages per device) instead of one contiguous block, round-robining
microbatches through them. This cuts the effective bubble fraction by
approximately a factor of `v` (`bubble_fraction ~ (P-1)/(v*(M+P-1))`
informally) at the cost of `v` times more activation-handoff communication
rounds — a bandwidth-for-bubble trade, useful when `M` can't be made large
(e.g. small-batch inference).

## Why this matters more for inference than the numbers alone suggest

Training amortizes bubble over large `M` (many microbatches per optimizer
step, chosen freely). Inference decode has no such freedom — the "microbatch
count" available per step is the concurrent request batch size, which is
often small, especially for latency-sensitive single-request serving. A PP
degree of 4 serving batch size 2 has `M=2 < P`, so bubble fraction is huge
(`(4-1)/(2+4-1)=3/5=60%`) — PP is a poor fit for low-batch, latency-sensitive
decode. TP doesn't have this failure mode (its per-layer all-reduce cost is
independent of batch/microbatch count), which is why TP, not PP, is the
default choice for splitting a single request across GPUs within a node; PP
is reserved for cases where the model doesn't fit within one node's TP domain
at all, or for high-throughput batch-heavy serving where `M` is naturally
large.
