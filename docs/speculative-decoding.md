# Speculative decoding

Target model (the model whose output distribution you actually want) is
memory-bandwidth-bound at batch=1 decode — one token costs one full pass over
every weight (see inference-roofline.md). Speculative decoding trades some of
that idle compute for wall-clock: a cheap draft model proposes several tokens
ahead, the expensive target model verifies all of them in a single parallel
forward pass, at cost close to one token's worth of target latency.

## Draft/verify loop

1. Draft model (small, fast — e.g. a distilled or smaller-parameter sibling of
   the target) autoregressively generates `gamma` candidate tokens
   `x1...x_gamma`, cheaply, one at a time.
2. Target model runs a single forward pass over the prompt plus all `gamma`
   draft tokens simultaneously (parallel, like prefill) — this is the
   expensive step, but it happens once per round instead of once per token.
   This pass yields the target's true distribution `p(.)` at every one of the
   `gamma` positions, plus one bonus position past the last draft token.
3. **Modified rejection sampling**, applied left to right: for draft token
   `x_i` with draft probability `q(x_i)` and target probability `p(x_i)`,
   accept it with probability `min(1, p(x_i)/q(x_i))`. If accepted, move to
   `x_{i+1}`. If rejected — stop; sample a replacement token from the residual
   distribution `norm(max(0, p(x) - q(x)))` and discard the rest of the draft.
   If all `gamma` are accepted, sample one bonus token from `p(.)` at the
   final position for free (it came from the same forward pass).

This procedure is provably distribution-preserving: the output token
distribution is exactly the target model's distribution, not an approximation
— rejection sampling's whole point is matching `p` using only samples from `q`
plus one extra uniform draw per position.

## Acceptance rate and expected tokens per round

Let `alpha` = per-token average acceptance probability (empirically measured
per draft/target pair, typically 0.6-0.9 for a well-matched draft). Expected
number of tokens emitted in one draft/verify round (draft tokens accepted plus
the bonus/replacement token):

```
E[tokens_per_round] = (1 - alpha^(gamma+1)) / (1 - alpha)
```

As `gamma -> inf` this saturates at `1/(1-alpha)` — diminishing returns from
proposing more draft tokens than the acceptance rate can sustain.

## Expected speedup

Cost of one round in units of "one target forward pass": the target
verification pass costs 1 (parallel over `gamma+1` positions, same order of
cost as a single-token forward pass on modern accelerators since it's
compute-bound like prefill). The draft steps cost `gamma * c`, where
`c = draft_step_time / target_step_time` (typically 0.03-0.15 for a draft
model 5-20x smaller than the target).

```
speedup = E[tokens_per_round] / (gamma * c + 1)
        = (1 - alpha^(gamma+1)) / ((1 - alpha) * (gamma * c + 1))
```

## Worked numbers

`alpha=0.7, gamma=4, c=0.1`:
`E[tokens] = (1 - 0.7^5)/(1 - 0.7) = 0.8319/0.3 = 2.773`
`speedup = 2.773 / (4*0.1 + 1) = 2.773/1.4 = 1.98x`

`alpha=0.9, gamma=5, c=0.05` (strong draft, cheap relative to target):
`E[tokens] = (1 - 0.9^6)/0.1 = 0.4686/0.1 = 4.686`
`speedup = 4.686 / (5*0.05 + 1) = 4.686/1.25 = 3.75x`

`alpha=0.4, gamma=4, c=0.1` (weak draft):
`E[tokens] = (1 - 0.4^5)/0.6 = 0.9898/0.6 = 1.650`
`speedup = 1.650/1.4 = 1.18x` — marginal; a slightly worse draft or higher `c`
pushes this under 1.0, i.e. speculative decoding becomes a net slowdown.

## Break-even and tuning

Speedup < 1 whenever `gamma * c` grows faster than the accepted-token yield —
concretely, a bad draft (`alpha` low) wastes rounds on rejections while still
paying `gamma * c` up front. There's an optimal `gamma` for given `(alpha, c)`
(diminishing marginal yield from more draft tokens vs linear marginal draft
cost); production systems tune `gamma` per draft/target pair, typically
landing at `gamma=3-8`. Draft-model quality (`alpha`) matters more than draft
speed (`c`) in most regimes because `alpha` appears exponentiated in the
numerator while `c` only scales linearly in the denominator.
