# llm-inference

[![checks](https://github.com/bhargavchintam/llm-inference/actions/workflows/checks.yml/badge.svg)](https://github.com/bhargavchintam/llm-inference/actions/workflows/checks.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
![dependencies](https://img.shields.io/badge/dependencies-numpy%20only-013243?logo=numpy&logoColor=white)
[![license](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

The LLM inference stack, written out in NumPy so the mechanics are visible and every piece self-checks —
no GPU, no framework, just the algorithms.

<p align="center">
  <img src="docs/media/demo.gif" alt="python3 check_all.py runs all 30 self-checks; every one passes" width="860">
</p>

## Quick start

```sh
pip install -r requirements.txt   # numpy is the only dependency
python3 check_all.py              # runs all 30 self-checks, about 4 s on a laptop CPU
python3 engine.py                 # a tiny transformer: prefill, KV-cached decode, sampling
```

Each script also runs on its own, for example `python3 serving/paged_attention.py`.

## What's inside

```
attention/     MHA / MQA / GQA, KV cache, RoPE, ALiBi, flash-attention (numpy)
quantization/  int8/int4, weight-only, GPTQ / AWQ / SmoothQuant (toy), KV-cache quant
serving/       paged attention, chunked prefill, continuous batching, speculative decoding, scheduler, prefix cache
sampling/      greedy, temperature, top-k, top-p, min-p, repetition penalty, beam search
docs/          kv cache math, paged attention, batching, spec decoding, quant, parallelism, roofline
engine.py      a tiny end-to-end transformer: prefill + incremental decode with a KV cache and a sampler
check_all.py   runs every script and fails if any check fails (this is what CI runs)
```

## What each check proves

The point is the invariants. Every script compares an optimized path against a plain reference and asserts
the two agree. All runs are seeded, and the numbers below are what CI prints on Linux. Rounding-level errors can differ
on other machines: on macOS, `sliding_window.py` prints `max_err=0.00e+00`.

| Area | Script | Invariant | Result |
|---|---|---|---|
| Engine | [`engine.py`](engine.py) | KV-cached incremental decode gives the same logits as recomputing the full prefix at every position | max error 1.7e-16 |
| Attention | [`kv_cache.py`](attention/kv_cache.py) | incremental decode equals full recompute over the prefix | max error 4.4e-16 |
| | [`flash_attention_numpy.py`](attention/flash_attention_numpy.py) | tiled online-softmax attention equals full `softmax(QKᵀ/√d)V`, under two different tilings | max error 3.3e-16 |
| | [`gqa.py`](attention/gqa.py) | GQA with one group per head is MHA; GQA with a single group is MQA | exact |
| | [`sliding_window.py`](attention/sliding_window.py) | window-W attention with a W-entry KV cache equals full attention with a band mask | max error 2.2e-16 |
| Serving | [`paged_attention.py`](serving/paged_attention.py) | attention over a paged KV block table equals attention over contiguous KV | exact (12 blocks of 16 tokens) |
| | [`chunked_prefill.py`](serving/chunked_prefill.py) | chunked prefill preserves KV and next-token logits while allowing decode between chunks | max error 5.6e-17; wait falls from 16 to at most 8 token slots |
| | [`continuous_batching.py`](serving/continuous_batching.py) | iteration-level batching beats static batching on the same 8 requests | 1.90× throughput (30 vs 57 steps) |
| | [`prefix_cache.py`](serving/prefix_cache.py) | reusing a cached prefix KV equals recomputing the whole sequence | max error 8.3e-17 |
| | [`speculative_decoding.py`](serving/speculative_decoding.py) | accept/reject keeps the target distribution, and the acceptance rate equals Σ min(p, q) | distribution error 2.5e-3 over 150k trials; 3.0 tokens per round |
| | [`scheduler.py`](serving/scheduler.py) | the token budget holds at every step; an oversized request is rejected at admission | holds |
| Quantization | [`gptq_toy.py`](quantization/gptq_toy.py) | 4-bit GPTQ beats round-to-nearest on layer output error | 37.1% lower error |
| | [`awq_toy.py`](quantization/awq_toy.py) | 4-bit AWQ beats round-to-nearest | 33.9% lower error |
| | [`smoothquant_toy.py`](quantization/smoothquant_toy.py) | SmoothQuant int8 beats naive int8 when activations have outliers | 0.011 vs 0.037 relative error |
| | [`kv_cache_quant.py`](quantization/kv_cache_quant.py) | an int8 KV cache stays accurate at a fraction of fp32 memory | 3.76× smaller |
| Sampling | [`sampling/`](sampling) | greedy, temperature, top-k, top-p, min-p, repetition penalty and beam search each satisfy their definition | all hold |

## Notes

`docs/` works the numbers behind the code:

- [KV cache](docs/kv-cache.md): why caching K and V removes the O(t²) recompute, and what the cache costs in memory
- [PagedAttention](docs/paged-attention.md): fixed-size KV blocks and per-sequence block tables, as in vLLM
- [Continuous batching](docs/continuous-batching.md): iteration-level scheduling (Orca) and why static batching wastes cycles
- [Speculative decoding](docs/speculative-decoding.md): trading idle compute at batch 1 for wall-clock time, without changing the output distribution
- [Quantization](docs/quantization.md): affine int8/int4, symmetric vs asymmetric, GPTQ, AWQ and SmoothQuant
- [Inference roofline](docs/inference-roofline.md): arithmetic intensity, the ridge point, and why decode is memory-bound
- [Throughput vs latency](docs/throughput-vs-latency.md): TTFT, inter-token latency, and the batch-size knob between them
- [Tensor and pipeline parallelism](docs/tensor-and-pipeline-parallelism.md): two ways to split a model across GPUs

## Related

[gpu-kernels](https://github.com/bhargavchintam/gpu-kernels) has the same primitives as CUDA and Triton kernels,
each checked against a NumPy reference.

## Contributing

New checks are welcome. Each one is a single seeded script that asserts an invariant against a reference.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the conventions and a list of open ideas.

## License

[MIT](LICENSE)
