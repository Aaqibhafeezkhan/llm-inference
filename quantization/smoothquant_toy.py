import numpy as np

np.random.seed(0)


def quant_per_tensor_int8(w):
    qmax = 127
    scale = max(np.abs(w).max(), 1e-8) / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def quant_per_channel_int8(w, axis=0):
    qmax = 127
    reduce_axes = tuple(a for a in range(w.ndim) if a != axis)
    amax = np.maximum(np.abs(w).max(axis=reduce_axes, keepdims=True), 1e-8)
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


n, d_in, d_out = 256, 128, 64

# large per-channel activation outliers (the pattern SmoothQuant targets: a few fixed input
# channels with activation magnitude 10-100x the rest, which wreck per-tensor int8 activation quant)
X = np.random.randn(n, d_in).astype(np.float32)
outlier_ch = np.random.choice(d_in, 6, replace=False)
X[:, outlier_ch] *= 50.0

W = (np.random.randn(d_out, d_in) * 0.1).astype(np.float32)
Y_true = X @ W.T

# naive: X quantized per-tensor int8, W quantized per-out-channel int8
Xq, sx = quant_per_tensor_int8(X)
Wq, sw = quant_per_channel_int8(W, axis=0)
Y_naive = (Xq.astype(np.int32) @ Wq.T.astype(np.int32)).astype(np.float32) * sx * sw.reshape(1, -1)
err_naive = rel_err(Y_naive, Y_true)

# smoothquant: migrate the outlier difficulty from activations into weights via a per-input-channel
# smoothing factor s_j = max|X_j|^alpha / max|W_j|^(1-alpha), X' = X / s, W' = W * s
alpha = 0.5
aX = np.abs(X).max(axis=0)
aW = np.abs(W).max(axis=0)
s = (aX ** alpha) / (aW ** (1 - alpha) + 1e-8)

Xs = X / s[None, :]
Ws = W * s[None, :]

Xsq, sxs = quant_per_tensor_int8(Xs)
Wsq, sws = quant_per_channel_int8(Ws, axis=0)
Y_sq = (Xsq.astype(np.int32) @ Wsq.T.astype(np.int32)).astype(np.float32) * sxs * sws.reshape(1, -1)
err_sq = rel_err(Y_sq, Y_true)

assert err_sq < err_naive, f"SmoothQuant should beat naive W8A8: {err_sq} >= {err_naive}"
print(f"PASS smoothquant_toy int8 output rel_err: smoothquant={err_sq:.6f} (alpha={alpha}) < naive={err_naive:.6f} (reduction={100*(1-err_sq/err_naive):.1f}%)")
