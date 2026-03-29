# Quantization

Maps fp16/bf16 weights (and optionally activations) to low-bit integers via an
affine transform: `x_int = round(x / scale) + zero_point`, dequantize with
`x ~= (x_int - zero_point) * scale`. Symmetric quantization fixes
`zero_point=0` (simpler, slightly worse for skewed distributions); asymmetric
keeps it free (better range utilization, one more value to store/compute).

## Bit width and granularity

**int8**: 256 levels, range `[-128, 127]` symmetric. Quantization error is
usually small enough that per-tensor or per-channel scales suffice for weights;
activations are harder (see SmoothQuant below).

**int4**: 16 levels, range `[-8, 7]`. An order of magnitude coarser — a weight
matrix with per-tensor int4 quantization typically wrecks model quality
(double-digit perplexity increase) because a single scale can't represent both
a tensor's typical magnitude and its outlier elements at 16 levels. int4 is
usable in practice only with finer-grained scales (per-channel, and usually
group-wise) plus a calibration-aware method (GPTQ, AWQ).

**Granularity, coarse to fine**:
- *Per-tensor*: one `(scale, zero_point)` for the entire weight matrix.
  Cheapest (2-8 bytes of overhead total), worst accuracy — one outlier element
  blows up the scale for every other element in the tensor.
- *Per-channel*: one scale per output channel (row of the weight matrix, i.e.
  per output neuron). Overhead = `out_features` scales per matrix — for a
  4096x4096 layer, 4096 scales (8KB in fp16) against a 32MB fp16 weight, ~0.02%
  overhead. Standard for weight-only int8.
- *Per-group*: one scale per contiguous block of `group_size` elements along
  the reduction (input) dimension — typically `group_size=128`. Overhead scales
  as `num_weights / group_size` scales. For 7B params at `group_size=128`:
  `7e9/128 ≈ 54.7M` groups; at 2 bytes scale + 2 bytes zero-point per group,
  `54.7M * 4B ≈ 219 MB` — small next to the 3.5GB int4 payload itself but not
  negligible. This is the granularity that makes int4 viable.

## GPTQ

Post-training, layer-by-layer weight quantization using a calibration set
(~128 sequences is typical, no gradient training). For each layer, quantizes
weight columns one at a time; after quantizing a column, adjusts the
*remaining unquantized* columns to compensate for the error just introduced,
using the inverse Hessian of the layer's reconstruction loss (this is the
"optimal brain quantization" lineage — greedy per-column quantization with
second-order error compensation, rather than naive round-to-nearest per
element). Runs on a single GPU, hours for a 7B-70B model, no backprop through
the full network. Typically int4, group-size 128, near-fp16 perplexity on
standard benchmarks.

## AWQ (Activation-aware Weight Quantization)

Observation: quantization error concentrates in a small fraction of weight
channels (~1%) — specifically the channels multiplied by activations with the
largest magnitudes, because those channels dominate the layer's output and any
rounding error there dominates output error. AWQ doesn't touch those channels'
precision directly (no mixed-precision storage); instead it rescales: multiply
the salient weight channels up by a per-channel factor `s` before
quantization (shrinking their *relative* quantization step), and divide the
corresponding activation channels by the same `s` at inference time — the
product `s * (1/s)` cancels, so the computed output is unchanged, but the
salient weights now quantize with less relative error. `s` per channel is
found by a small grid search minimizing output error on the calibration set,
not gradient descent. Cheaper to run than GPTQ (no per-layer Hessian), broadly
competitive accuracy for int4 group-wise.

## SmoothQuant (enables W8A8)

Weight-only quantization avoids activation quantization error entirely, but
compute stays fp16. To get int8 *compute* (int8 x int8 GEMM), activations must
also be quantized — and activations have systematic outlier channels (specific
feature dimensions with 10-100x the typical magnitude) that make naive int8
activation quantization lossy, more so than weights.

SmoothQuant migrates quantization difficulty from activations to weights via a
mathematically exact per-channel rescaling: for input `X` and weight `W`
(with `Y = XW`), pick a per-channel smoothing vector `s` and compute
`Y = (X * diag(s)^-1) * (diag(s) * W)` — algebraically identical to `XW`, but
now the activation channels are divided by `s` (shrinking outliers) and the
weight channels are multiplied by `s` (absorbing them). Choose
`s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)`, `alpha ~ 0.5`, balancing how
much outlier magnitude moves to which side — `alpha` near 1 favors smoothing
activations, near 0 favors leaving weights alone; 0.5 is the typical default
and works because weights are comparatively easy to quantize regardless. After
smoothing, both `X'` and `W'` quantize well under per-tensor or per-channel
int8.

## W8A8 vs W4A16

**W8A8** (weights and activations both int8): enables true int8 GEMM on tensor
cores that support it — roughly 2x throughput vs fp16 matmul on hardware with
native int8 tensor-core paths, on top of the 2x memory-bandwidth reduction
from int8 weights. Requires activation quantization at runtime (dynamic
per-token or per-tensor scales, since activation distributions shift with
input) — SmoothQuant is what makes this tractable. Wins when compute-bound:
prefill, or decode at large batch (see inference-roofline.md).

**W4A16** (int4 weights, activations stay fp16/bf16): dequantize weights to
fp16 on the fly (or use fused int4 x fp16 kernels) and matmul in fp16 — FLOPs
are unchanged from a full fp16 model, only weight-read bytes shrink 4x. Doesn't
help a compute-bound phase at all, but decode at low-to-moderate batch is
memory-bound on weight reads (arithmetic intensity ~1 FLOP/byte at batch=1,
see inference-roofline.md), so a 4x cut in weight bytes moving through HBM
translates almost directly into decode throughput. This is why W4A16 (GPTQ/AWQ
int4) is the default choice for single-user / low-batch local and edge
serving, while W8A8 shows up in high-throughput batched serving.

## Worked memory numbers, Llama-2 7B

fp16: `7e9 * 2 = 14 GB`. int8 (W8): `7e9 * 1 = 7 GB`. int4 (W4, group=128):
`7e9 * 0.5 = 3.5 GB` payload + `~219 MB` scale/zero overhead `≈ 3.72 GB` total
— a 3.76x reduction vs fp16, close to the naive 4x since group overhead is
small.
