# llm-inference

The LLM inference stack, written out in NumPy so the mechanics are visible and every piece self-checks —
no GPU, no framework, just the algorithms.

```
attention/     MHA / MQA / GQA, KV cache, RoPE, ALiBi, flash-attention (numpy)
quantization/  int8/int4, weight-only, GPTQ / AWQ / SmoothQuant (toy), KV-cache quant
serving/       paged attention, continuous batching, speculative decoding, scheduler, prefix cache
sampling/      greedy, temperature, top-k, top-p, min-p, repetition penalty, beam search
docs/          kv cache math, paged attention, batching, spec decoding, quant, parallelism, roofline
engine.py      a tiny end-to-end transformer: prefill + incremental decode with a KV cache and a sampler
```

Everything runs on CPU with NumPy and checks itself against a reference:

```sh
python3 attention/gqa.py            # GQA with groups==heads matches MHA; 1 group matches MQA
python3 serving/paged_attention.py  # attention over paged KV == contiguous attention
python3 engine.py                   # generates a short sequence, deterministic with a fixed seed
```

The point is the invariants: a KV-cached decode gives the same logits as recomputing the whole prefix,
paged attention matches contiguous attention, int8 quant stays inside its error bound, GPTQ/AWQ/SmoothQuant
beat naive round-to-nearest, and speculative decoding's accept/reject preserves the target distribution.
`docs/` works the numbers — KV-cache size, the memory-bandwidth wall on decode, why batching helps.
