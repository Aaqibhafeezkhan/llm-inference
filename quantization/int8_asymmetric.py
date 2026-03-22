import numpy as np

np.random.seed(0)


def quant_per_tensor(w, bits=8):
    qmax = 2 ** bits - 1
    wmin, wmax = min(w.min(), 0.0), max(w.max(), 0.0)
    scale = max((wmax - wmin) / qmax, 1e-8)
    zp = np.clip(np.round(-wmin / scale), 0, qmax)
    q = np.clip(np.round(w / scale + zp), 0, qmax).astype(np.uint8)
    return q, scale, zp


def quant_per_channel(w, axis=0, bits=8):
    qmax = 2 ** bits - 1
    reduce_axes = tuple(a for a in range(w.ndim) if a != axis)
    wmin = np.minimum(w.min(axis=reduce_axes, keepdims=True), 0.0)
    wmax = np.maximum(w.max(axis=reduce_axes, keepdims=True), 0.0)
    scale = np.maximum((wmax - wmin) / qmax, 1e-8)
    zp = np.clip(np.round(-wmin / scale), 0, qmax)
    q = np.clip(np.round(w / scale + zp), 0, qmax).astype(np.uint8)
    return q, scale, zp


def dequant(q, scale, zp):
    return (q.astype(np.float32) - zp) * scale


def matmul_asym(wq, sw, zw, xq, sx, zx):
    # wq: (out,in) uint8, xq: (in,batch) uint8, zero-point cross-term correction
    acc = wq.astype(np.int32) @ xq.astype(np.int32)
    sum_x = xq.astype(np.int32).sum(axis=0, keepdims=True)
    sum_w = wq.astype(np.int32).sum(axis=1, keepdims=True)
    n = wq.shape[1]
    out = acc - zw * sum_x - zx * sum_w + zw * zx * n
    return out.astype(np.float32) * (sw * sx)


# round-trip: per-tensor, biased distribution (asymmetric quant shines here)
w = (np.random.randn(256, 256).astype(np.float32) * 0.3 + 0.5)
q, s, zp = quant_per_tensor(w)
wr = dequant(q, s, zp)
err = np.abs(w - wr).max()
assert err <= s / 2 + 1e-6, f"per-tensor round-trip bound violated: {err} > {s/2}"
print(f"PASS int8 asymmetric per-tensor round-trip max_err={err:.6f} scale={s:.6f} zp={zp:.1f}")

# round-trip: per-channel
qc, sc, zpc = quant_per_channel(w, axis=0)
wrc = dequant(qc, sc, zpc)
errc = np.abs(w - wrc).max()
bound = (sc / 2).max()
assert errc <= bound + 1e-6, f"per-channel round-trip bound violated: {errc} > {bound}"
print(f"PASS int8 asymmetric per-channel round-trip max_err={errc:.6f} max_scale={sc.max():.6f}")

# int8 matmul vs fp, with zero-point correction
d_out, d_in, batch = 128, 256, 64
W = np.random.randn(d_out, d_in).astype(np.float32) * 0.1 + 0.05
x = np.random.rand(d_in, batch).astype(np.float32)  # non-negative activations, typical post-relu

Wq, sw, zw = quant_per_channel(W, axis=0)
xq, sx, zx = quant_per_tensor(x)

y_int8 = matmul_asym(Wq, sw, zw, xq, sx, zx)
y_fp = W @ x
rel_err = np.linalg.norm(y_int8 - y_fp) / np.linalg.norm(y_fp)
assert rel_err < 0.05, f"int8 matmul rel_err too high: {rel_err}"
print(f"PASS int8 asymmetric matmul rel_err={rel_err:.6f}")
