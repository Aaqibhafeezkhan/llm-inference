# Contributing

Thanks for taking a look. This repo stays small on purpose: one idea per file, NumPy only, and every file
proves something about itself.

## Run the checks

```sh
pip install -r requirements.txt
python3 check_all.py
```

CI runs the same command on Python 3.10, 3.12 and 3.13. A pull request is ready when it prints
`N/N checks passed`.

## Adding a check

1. Put the script in the folder it belongs to (`attention/`, `quantization/`, `serving/`, `sampling/`).
2. Seed it (`np.random.seed(0)`) so the numbers are the same on every machine.
3. Compare the optimized path against a plain reference: full recompute, contiguous memory, brute force,
   or the textbook formula.
4. `assert` the invariant with a tight tolerance, then print one line that contains `PASS`, the measured
   error or gain, and the invariant in words, for example:

   ```
   paged_attention PASS max_err=0.00e+00 (paged-KV attention == contiguous attention)
   ```

5. Keep it readable top to bottom: no classes unless they help, no dependencies beyond NumPy.
6. If the idea needs explaining, add or extend a note in `docs/`, and add a row to the table in the README.

`check_all.py` picks up new scripts in those folders automatically.

## Ideas that would fit

- Multi-head latent attention (MLA) with a compressed KV cache, checked against full MHA
- Medusa or EAGLE-style draft heads for speculative decoding
- Chunked prefill interleaved with decode, checked against unchunked prefill
- Sliding-window attention and attention sinks, checked against a masked full attention
- FP8 (E4M3/E5M2) simulation for weights and the KV cache, with error bounds
- A tensor-parallel matmul split across simulated devices, checked against the single-device result

Open an issue first if you want to discuss an idea before writing it.

## Pull requests

- One topic per pull request, with a short description of the invariant it checks.
- By contributing you agree that your work is released under the [MIT License](LICENSE).
