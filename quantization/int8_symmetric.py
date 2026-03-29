import numpy as np

np.random.seed(0)


def quant_per_tensor(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = max(np.abs(w).max(), 1e-8) / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def quant_per_channel(w, axis=0, bits=8):
    qmax = 2 ** (bits - 1) - 1
    reduce_axes = tuple(a for a in range(w.ndim) if a != axis)
    amax = np.maximum(np.abs(w).max(axis=reduce_axes, keepdims=True), 1e-8)
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def dequant(q, scale):
    return q.astype(np.float32) * scale


# round-trip: per-tensor
w = np.random.randn(256, 256).astype(np.float32) * 0.3
q, s = quant_per_tensor(w)
wr = dequant(q, s)
err = np.abs(w - wr).max()
assert err <= s / 2 + 1e-6, f"per-tensor round-trip bound violated: {err} > {s/2}"
print(f"PASS int8 symmetric per-tensor round-trip max_err={err:.6f} scale={s:.6f}")

# round-trip: per-channel (per output row)
qc, sc = quant_per_channel(w, axis=0)
wrc = dequant(qc, sc)
errc = np.abs(w - wrc).max()
bound = (sc / 2).max()
assert errc <= bound + 1e-6, f"per-channel round-trip bound violated: {errc} > {bound}"
print(f"PASS int8 symmetric per-channel round-trip max_err={errc:.6f} max_scale={sc.max():.6f}")

# int8 matmul vs fp: W (out,in) @ x (in,batch), W quantized per-out-channel, x quantized per-tensor
d_out, d_in, batch = 128, 256, 64
W = np.random.randn(d_out, d_in).astype(np.float32) * 0.1
x = np.random.randn(d_in, batch).astype(np.float32)

Wq, sw = quant_per_channel(W, axis=0)
xq, sx = quant_per_tensor(x)

acc = Wq.astype(np.int32) @ xq.astype(np.int32)
y_int8 = acc.astype(np.float32) * sw * sx

y_fp = W @ x
rel_err = np.linalg.norm(y_int8 - y_fp) / np.linalg.norm(y_fp)
assert rel_err < 0.05, f"int8 matmul rel_err too high: {rel_err}"
print(f"PASS int8 symmetric matmul rel_err={rel_err:.6f}")
