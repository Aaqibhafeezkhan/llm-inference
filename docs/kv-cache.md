# KV cache

Autoregressive decode recomputes nothing from prior tokens if K and V projections
are cached per layer per head. Without the cache, generating token `t` requires
re-running attention over all `t-1` prior tokens' Q/K/V projections — O(t^2) work
across a sequence. With the cache, each new token does one forward pass and appends
its K/V to the cache, O(t) total.

## Size formula

```
bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes * batch_size
```

`2` = one tensor for K, one for V. `n_kv_heads` is the number of key/value heads —
equal to `n_heads` under plain multi-head attention (MHA), smaller under
grouped-query attention (GQA) or multi-query attention (MQA, `n_kv_heads = 1`).

Per-token, per-sequence bytes (drop `seq_len`, `batch_size`):

```
per_token_bytes = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
```

## Worked: Llama-2 7B (MHA)

`n_layers=32`, `n_heads=n_kv_heads=32`, `head_dim=128`, fp16 (`dtype_bytes=2`).

```
per_token = 2 * 32 * 32 * 128 * 2 = 524,288 bytes = 512 KiB/token
```

At `seq_len=2048`: `512 KiB * 2048 = 1 GiB` per sequence.
At `seq_len=4096`: `2 GiB` per sequence.
Batch of 32 at 4096 context: `2 GiB * 32 = 64 GiB` — larger than the 14 GiB the
fp16 weights themselves occupy. Cache, not weights, is the serving memory
bottleneck at any real batch size or context length.

## Worked: Llama-2 70B (GQA, 8 KV heads)

`n_layers=80`, `n_heads=64`, `n_kv_heads=8`, `head_dim=128`, fp16.

```
per_token = 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KiB/token
```

At `seq_len=4096`, batch=1: `320 KiB * 4096 = 1.25 GiB`.

Counterfactual — if 70B used plain MHA (`n_kv_heads=64` instead of 8):

```
per_token = 2 * 80 * 64 * 128 * 2 = 2,621,440 bytes = 2.5 MiB/token
```

`2.5 MiB * 4096 = 10 GiB` per sequence — 8x the GQA number, exactly the
`n_heads / n_kv_heads` ratio. GQA trades a small quality delta (fewer distinct KV
projections) for linear cache reduction; MQA (`n_kv_heads=1`) pushes this further
at higher quality risk.

## dtype and batching pressure

Dropping fp16 (2 bytes) to int8 (1 byte) halves cache size for identical
`seq_len` and `batch_size` — commonly done via KV-cache-only quantization even
when weights/activations stay fp16, since cache size, not weight size, gates
concurrent request count.

Cache size grows linearly in `batch_size * seq_len`, whereas weight footprint is
fixed. This is why serving frameworks size KV cache as "remaining HBM after
weights" and admit requests until that budget is exhausted — the working
variable is `max_batch_size * max_seq_len`, not model size.

## HBM budget example, A100 80GB, Llama-2 7B fp16

80 GiB HBM - 14 GiB weights - ~2 GiB activations/overhead ≈ 64 GiB for cache.
At 512 KiB/token: `64 GiB / 512 KiB ≈ 131,000` cacheable tokens total, shared
across all concurrent sequences — e.g. 32 sequences at ~4096 tokens each, or 512
sequences at ~256 tokens each. This budget, not compute, is what paged
attention and eviction/preemption policies are managing.
