import numpy as np

np.random.seed(0)


def quant_per_channel_int8(w, axis=0):
    qmax = 2 ** 7 - 1
    reduce_axes = tuple(a for a in range(w.ndim) if a != axis)
    amax = np.maximum(np.abs(w).max(axis=reduce_axes, keepdims=True), 1e-8)
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


# W8A16: weights stored int8 (per-out-channel scale), activations stay fp16, dequant weight
# on the fly and matmul in fp16 (accumulate fp32, cast back), as done in weight-only kernels.
d_out, d_in, batch = 512, 1024, 32
W = np.random.randn(d_out, d_in).astype(np.float32) * 0.05
x = np.random.randn(d_in, batch).astype(np.float32)
x16 = x.astype(np.float16)

Wq, scale = quant_per_channel_int8(W, axis=0)
W_deq16 = (Wq.astype(np.float32) * scale).astype(np.float16)

acc = np.zeros((d_out, batch), dtype=np.float32)
acc += W_deq16.astype(np.float32) @ x16.astype(np.float32)
y_w8a16 = acc

y_fp = W @ x
rel_err = np.linalg.norm(y_w8a16 - y_fp) / np.linalg.norm(y_fp)
assert rel_err < 0.02, f"W8A16 matmul rel_err too high: {rel_err}"
print(f"PASS weight_only_quant (W8A16) matmul rel_err={rel_err:.6f} d_out={d_out} d_in={d_in} batch={batch}")

# memory savings check: int8 weight + fp16 scale vs fp32 weight
bytes_fp32 = W.size * 4
bytes_w8a16 = Wq.size * 1 + scale.size * 4
ratio = bytes_fp32 / bytes_w8a16
print(f"PASS weight_only_quant memory ratio fp32/w8a16={ratio:.2f}x")
