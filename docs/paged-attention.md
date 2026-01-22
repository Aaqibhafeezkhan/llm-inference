# PagedAttention

vLLM's answer to KV-cache memory waste. Treats the KV cache like OS virtual
memory: fixed-size blocks, a per-sequence block table mapping logical block
index to physical block, non-contiguous physical placement.

## The problem it fixes

Naive serving allocates one contiguous buffer per sequence sized for
`max_seq_len` (e.g. 4096 tokens) at request start, because attention kernels
expect contiguous K/V. Two failure modes:

- **Internal fragmentation**: a request that finishes at 300 tokens still holds
  the full 4096-token allocation for its lifetime. Observed vLLM paper numbers:
  60-80% of allocated KV cache memory wasted this way under realistic length
  distributions.
- **External fragmentation**: even unused memory is unusable because it's
  scattered in variously-sized freed contiguous chunks that don't fit the next
  request's contiguous requirement — classic malloc-style fragmentation.

Both stem from the contiguity requirement. Remove it, the waste goes away.

## Block table mechanics

KV cache is divided into fixed-size **blocks**, each holding `block_size`
tokens' worth of K and V for one sequence (`block_size=16` is the common vLLM
default). Physical blocks are allocated from a single global pool as needed —
one block at a time, on demand, not `max_seq_len` blocks up front.

Each sequence has a **block table**: `logical_block_idx -> physical_block_idx`,
analogous to a page table. Logical blocks are numbered 0, 1, 2, ... in
generation order; physical blocks can be anywhere in the pool, including
interleaved with blocks belonging to other sequences.

```
seq A block table: [0 -> phys 7, 1 -> phys 3, 2 -> phys 19]
seq B block table: [0 -> phys 4, 1 -> phys 7]   # phys 7 shared, see COW below
```

Attention kernels gather K/V for a sequence by walking its block table instead
of assuming a flat contiguous range — a level of indirection paid once per
block, not per token.

**Fragmentation bound**: internal fragmentation is now capped at
`block_size - 1` leftover tokens in the last (partially-filled) block per
sequence — at `block_size=16`, worst case 15 wasted token-slots per sequence
instead of thousands. External fragmentation disappears because any free
physical block satisfies any request; nothing needs to be contiguous.

**Cost of the indirection**: smaller `block_size` -> less internal
fragmentation but more block-table entries and more per-block kernel
overhead/scheduling metadata; larger `block_size` -> opposite tradeoff. 16-32
tokens/block is the typical sweet spot for a 100-2000 token average request.

## Copy-on-write for shared prefixes

Requests sharing a prefix — a common system prompt, or parallel-sampling /
beam-search branches from one prompt — can share physical blocks for that
prefix instead of duplicating them.

Mechanism: each physical block carries a **reference count**. A shared prefix's
blocks have `ref_count = N` for N sequences pointing at them via their own
block tables. Both sequences read the shared block freely (attention is
read-only over past KV). The moment one sequence needs to *write* a new token
into what was the last (possibly partially full) shared block, that write
would corrupt the other sequence's view — so the allocator copies that one
block to a fresh physical location first, decrements the ref count on the
original, and the writer proceeds on its private copy. Fully-shared interior
blocks (already full, never written again) are never copied — only the
boundary block where divergence starts.

Effect: sharing a 1000-token system prompt across 64 parallel samples costs
`1000 / block_size` blocks once, not 64 times over — at `block_size=16`, ~63
blocks saved x sample count, versus naive per-sequence allocation. Savings
scale with prefix length and fan-out (parallel sampling `n`, beam width, or
shared-prefix batched requests).

## Interaction with scheduling

Because allocation is incremental (one block per `block_size` new tokens,
not `max_seq_len` up front), the scheduler can admit more concurrent sequences
than a static-allocation scheme for the same HBM budget — average-case memory
use tracks average-case sequence length, not the worst case. This is what lets
continuous batching (see continuous-batching.md) push batch size up without
pre-reserving worst-case cache space per slot.
