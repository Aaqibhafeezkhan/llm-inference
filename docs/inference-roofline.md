# Inference roofline

Roofline model: achievable performance is `min(peak_compute, AI * peak_bandwidth)`,
where arithmetic intensity `AI = FLOPs / bytes_moved`. The **ridge point**
`peak_compute / peak_bandwidth` (units: FLOP/byte) is where a workload
transitions from memory-bound (left of ridge, bandwidth caps you) to
compute-bound (right of ridge, FLOPs cap you).

## Ridge points, real hardware

A100 80GB SXM: 312 TFLOPS fp16/bf16 tensor (dense), 2039 GB/s HBM2e.
`ridge = 312e12 / 2.039e12 ≈ 153 FLOP/byte`.

H100 80GB SXM: 989 TFLOPS bf16 tensor (dense), 3.35 TB/s HBM3.
`ridge = 989e12 / 3.35e12 ≈ 295 FLOP/byte`.

Anything under these AI values is memory-bound on the respective GPU no
matter how much raw compute is available — you're waiting on HBM, tensor
cores sit idle.

## Decode at batch=1: deeply memory-bound

Forward pass FLOPs for one token through an `N`-parameter model, matmul-
dominated: `~2N` FLOPs (standard forward-pass approximation, ignoring
nonlinearities/norm/attention as sub-dominant terms). At batch=1, every
weight is read from HBM exactly once and used once — no reuse — so bytes
moved `= N * bytes_per_param`.

```
AI_decode(batch=1) = 2N / (N * bytes_per_param) = 2 / bytes_per_param
```

fp16 (2B): `AI = 1 FLOP/byte`. int8 (1B): `AI = 2`. int4 (0.5B): `AI = 4`.
All three sit 2-3 orders of magnitude below the 153-295 ridge — decode at
batch=1 is memory-bound *regardless of quantization level*; quantization
helps by shrinking the bytes term directly (fewer bytes to move in the same
memory-bound regime), not by moving you toward compute-bound.

## Why batching helps decode

With batch `B`, a batched GEMM reads each weight matrix once and reuses it
across all `B` sequences (weights loaded to compute units once, multiplied
against `B` activation vectors). Compute scales with `B`, weight-read bytes
don't:

```
AI_decode(B) = 2NB / (N * bytes_per_param) = 2B / bytes_per_param
```

fp16: `AI(B) ≈ B`. To reach the A100 ridge (153) needs `B ≈ 153`; H100 ridge
(295) needs `B ≈ 295`. This is the quantitative reason serving stacks push
batch size aggressively — it's not just "more requests per second" bookkeeping,
it's the only lever that moves decode out of the memory-bound regime at all.

## The KV-cache term breaks the naive scaling

The `AI(B) ≈ B` result above only counts *weight* reads. Every decode step
also reads the *entire* KV cache for every sequence in the batch (attention
needs full past context each step) — and unlike weights, KV cache is not
shared across sequences, so this term scales as `B * seq_len`, not reused at
all.

Per-step memory traffic, full picture:

```
bytes/step = N * bytes_per_param            (weights, read once, shared)
           + kv_bytes_per_token * seq_len * B   (KV cache, read every step, not shared)
```

**Worked**: Llama-2 7B, fp16, `seq_len=2048`, `B=64`.
Weight bytes: `14 GB` (constant, from kv-cache.md math).
KV bytes: `512 KiB/token * 2048 * 64 = 64 GiB`.
Total: `~78 GB` moved in one decode step. At A100 bandwidth (2039 GB/s):
`78 GB / 2.039 TB/s ≈ 38.3 ms` for that single step — and every subsequent
decode step pays a similar or growing cost as `seq_len` climbs.

At this batch/context point, KV-cache traffic (64 GiB) dwarfs weight traffic
(14 GiB) by ~4.5x — arithmetic intensity does *not* keep climbing linearly
with `B` once KV-cache reads dominate; it saturates and can even fall as
`seq_len` grows. This is the concrete reason GQA/MQA (fewer KV heads, see
kv-cache.md), KV-cache quantization, and admission control on
`batch_size * seq_len` matter as much as raw weight quantization for
throughput at scale — the memory-bandwidth wall shows up in the cache term
first at long context, not the weight term.

## Prefill: compute-bound, for free

Prefill processes all `S` prompt tokens in one parallel forward pass; weights
are read once and reused across all `S` tokens, structurally identical to the
batching argument above but with `S` (prompt length) playing the role of `B`:

```
AI_prefill = 2S / bytes_per_param
```

fp16, `S=2048`: `AI = 2048` — 7x past the H100 ridge (295), solidly
compute-bound. A single reasonably long prompt gets you into the compute-bound
regime without needing concurrent requests at all; short prompts (`S` in the
tens) don't, and behave more like memory-bound decode. This compute/memory
split across the two phases (prefill parallel-and-compute-bound, decode
sequential-and-memory-bound) is the entire reason they're scheduled and
batched differently — see continuous-batching.md.

## Practical reading of the roofline

- Low-batch decode: bandwidth wall, quantization (fewer bytes/param) and GQA
  (fewer bytes/token in cache) are the direct levers.
- High-batch decode / long context: KV-cache bandwidth wall, paged attention
  and cache eviction/admission policy are the direct levers.
- Prefill / long prompts: compute wall, tensor-core throughput and
  precision (fp8/int8 compute, not just storage) are the direct levers.
