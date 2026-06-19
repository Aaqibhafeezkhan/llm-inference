# Continuous batching

Also called iteration-level scheduling (Orca, OSDI '22). Replaces
request-level batching with per-decode-step scheduling, because request
lifetimes in autoregressive generation are wildly uneven and static batching
wastes GPU cycles on that mismatch.

## Static batching's failure mode

Static batching groups N requests, runs the batch to completion, only forms
the next batch once all N finish. Problem: sequence lengths vary — one request
emits an EOS at 20 tokens, another runs to 2000. Once the short request
finishes, its batch slot is either padded (compute wasted on pad tokens that
produce nothing useful) or the whole batch stalls waiting, and no new request
can enter that batch until every member finishes.

Concretely: batch of 8, seven requests finish at ~50 tokens, one runs to 1000.
GPU spends 950 decode steps computing forward passes for 7 dead slots — at
best masked out, at worst literally padded and wasted, and either way batch
slots 0-6 sit idle instead of picking up new queued requests. Effective
throughput on those slots is 0 for 95% of the batch's wall-clock life.

## Prefill vs decode phases

Two distinct phases per request, different compute profiles:

- **Prefill**: process the full input prompt (all N prompt tokens) in one
  forward pass, computing K/V for all of them at once. Matmul-heavy,
  parallel over sequence length — compute-bound, high arithmetic intensity
  (see inference-roofline.md).
- **Decode**: process one new token per step, attending over the full cached
  context. One token's worth of matmul work per step, but has to read the
  entire KV cache and full weight set from HBM. Memory-bound, low arithmetic
  intensity, and this is the phase that runs for hundreds to thousands of
  iterations per request.

Prefill for a 500-token prompt costs roughly as much raw FLOPs as 500 decode
steps' matmuls combined, but takes far less wall-clock time because it's one
parallel pass over 500 tokens instead of 500 sequential HBM-bound round trips.

## Iteration-level scheduling

Continuous batching schedules at the granularity of a single forward pass
(one iteration), not a whole request:

1. At each iteration, the scheduler looks at all in-flight sequences plus the
   queue of waiting requests.
2. Any sequence that hit EOS or its length limit is evicted from the batch
   immediately — its slot is free the very next iteration, not after the
   slowest batchmate finishes.
3. A freed slot (or spare KV-cache budget) is immediately backfilled from the
   waiting queue, mixing a new request's prefill into the same iteration as
   other requests' decode steps.
4. This repeats every iteration — the "batch" is really a fluid set that
   changes membership every single step.

Consequence: no request waits for an unrelated request's tail latency, and no
GPU cycle is spent on a dead slot. Reported throughput gains in the Orca paper
and in vLLM: 2-4x over static/dynamic (request-level) batching baselines at
equivalent latency SLOs, mostly by eliminating the padding/straggler waste
above.

## Mixing prefill and decode in one iteration

Because prefill (compute-bound, wants big token count per pass) and decode
(memory-bound, wants big batch count per pass) have opposite bottlenecks,
naively interleaving a large prefill with many small decode steps in the same
iteration can stall decode latency for every other in-flight sequence — the
prefill's matmul dominates the iteration's wall-clock time. Production
schedulers (e.g. vLLM's chunked prefill) split a long prompt's prefill across
multiple iterations in bounded-size chunks, interleaved with ongoing decode
steps, to bound the per-iteration latency spike instead of running prefill as
one atomic block.

## Dependency on paged KV cache

Iteration-level admission of new requests mid-flight requires being able to
grow a sequence's KV-cache allocation incrementally, one block at a time,
without a contiguous-buffer reservation up front — otherwise the scheduler
can't know ahead of time how much cache a newly-admitted request will
eventually need. This is why continuous batching and paged attention
(paged-attention.md) are shipped together in practice: continuous batching
supplies the scheduling policy, paged attention supplies the memory layout
that makes arbitrary admit/evict timing cheap.
