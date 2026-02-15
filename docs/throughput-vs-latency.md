# Throughput vs latency

Two latency metrics, one throughput metric, and a batch-size knob that trades
between them. Everything here composes with inference-roofline.md's
memory-bandwidth math — these are the same numbers viewed as user-facing SLAs
instead of hardware utilization.

## TTFT — time to first token

Wall-clock from request arrival to the first generated token. Dominated by:

```
TTFT = queue_wait + prefill_time(S)
```

`prefill_time(S)` is compute-bound (roofline doc): for `S` prompt tokens
through an `N`-parameter model, `~2NS` FLOPs.

**Worked**: 7B model, `S=500` prompt tokens, A100 (312 TFLOPS fp16 peak):
`compute = 2 * 7e9 * 500 = 7e12 FLOPs`, `time = 7e12 / 312e12 ≈ 22.4 ms` at
100% utilization. Real achieved matmul utilization (MFU) on unfused,
real-world kernels typically lands 30-50% of peak, so realistic TTFT
contribution from compute alone is `45-75 ms` before any queueing — queueing
delay under load can dwarf this, which is why TTFT is a scheduling problem
(admission, chunked prefill) as much as a raw-FLOPs problem.

## TPOT / ITL — time per output token

Also called inter-token latency (ITL). Average wall-clock between consecutive
generated tokens during decode — the memory-bound decode-step time from
inference-roofline.md:

```
TPOT ≈ bytes_moved_per_step / HBM_bandwidth
```

Total response latency for `n` output tokens: `TTFT + (n-1) * TPOT`. For long
generations TPOT dominates total latency; for short generations (a handful of
tokens, e.g. classification-style prompts) TTFT dominates.

## Batch size: throughput up, TPOT up — not free

Batching raises aggregate throughput (roofline doc: `AI(B) ≈ B` for weight
reads) but each decode step also gets slower as batch and context grow,
because per-step memory traffic grows with `B` (KV-cache term). Individual
requests get *higher* TPOT (worse) even as the system serves *more*
tokens/sec in aggregate (better) — throughput and per-request latency move in
opposite directions on this knob.

**Worked**, continuing the kv-cache.md / inference-roofline.md example —
Llama-2 7B, fp16, A100 (2039 GB/s), `seq_len=2048` (KV = 512 KiB/token, so
exactly 1 GiB per sequence at this length):

| batch | bytes/step | TPOT | per-request tok/s | aggregate tok/s |
|---|---|---|---|---|
| 1 | 14 GB (weights only) | 14GB/2.039TB/s ≈ 6.9 ms | 144.9 | 144.9 |
| 64 | 14 GB + 64 GiB KV ≈ 78 GB | 78GB/2.039TB/s ≈ 38.3 ms | 26.1 | 1670 |

Going `batch=1 -> 64`: per-request throughput drops ~5.5x (7ms/token feels
instant, 38ms/token is still fine for reading speed but noticeably slower),
aggregate system throughput rises ~11.5x. Whether that trade is worth it is
entirely a product decision — interactive chat wants low TPOT even at low
aggregate throughput; batch summarization/offline scoring wants the opposite.

## SLA-based batching

Instead of picking a fixed batch size, pick a target TPOT (e.g. `50 ms/token`,
roughly the threshold where token-by-token streaming still reads as "live" to
a human) and solve for the maximum batch size that keeps decode-step time
under that bound, given the deployed context length.

```
budget_bytes = TPOT_target * bandwidth
max_batch = (budget_bytes - weight_bytes) / kv_bytes_per_sequence
```

**Worked**: same model/hardware, `seq_len=2048` (1 GiB KV/sequence),
`TPOT_target = 50 ms`:

```
budget = 0.050 s * 1899 GiB/s ≈ 94.95 GiB   (2039 GB/s ≈ 1899 GiB/s)
weight_bytes = 13.04 GiB (14 GB in GiB)
kv_budget = 94.95 - 13.04 ≈ 81.9 GiB
max_batch = 81.9 GiB / 1 GiB per sequence ≈ 82 sequences
```

At 82 concurrent sequences and 2048-token context on one A100, this model
just holds a 50ms/token TPOT SLA; sequence 83 pushes the average over budget.
Real schedulers (vLLM's continuous batching, TGI's dynamic batching) compute
a version of this bound continuously as sequences enter/exit and context
grows, and throttle admission — not batch size alone, since `batch * seq_len`
is the actual quantity that determines step time (a batch of 164 at half the
context length hits the same bound).

## Interaction with prefill/decode mixing

Continuous batching (continuous-batching.md) interleaves prefill chunks from
newly-admitted requests with ongoing decode steps. An unchunked large prefill
inserted into a decode-heavy iteration spikes that iteration's wall-clock
time — which shows up as a TPOT spike (a stutter) for every other in-flight
request, not just added TTFT for the new one. This is the direct link between
the scheduling policy and both latency metrics simultaneously: chunked
prefill exists specifically to decouple TTFT-for-new-request from
TPOT-stability-for-existing-requests, trading a slightly higher TTFT (prefill
spread across more iterations) for bounded per-iteration latency.

## Reading the two metrics together

- Low TTFT, low TPOT: small batch, short queue — low hardware utilization,
  expensive per token served.
- Low TTFT, higher TPOT: admit fast, batch aggressively once running —
  typical interactive chat serving under moderate-to-high load.
- Higher TTFT, low TPOT: queue before admitting, but run small batches once
  admitted — unusual, only makes sense under strict per-token latency SLAs
  with slack on initial response time.
- High TTFT, high TPOT: overloaded — throughput-maximizing but past the
  point either metric meets a reasonable interactive SLA; this is where
  autoscaling or load shedding needs to trigger.
