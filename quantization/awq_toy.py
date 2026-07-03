import numpy as np

np.random.seed(0)


def rtn_quant_per_channel(w, axis=0, bits=4):
    qmax = 2 ** (bits - 1) - 1
    reduce_axes = tuple(a for a in range(w.ndim) if a != axis)
    amax = np.maximum(np.abs(w).max(axis=reduce_axes, keepdims=True), 1e-8)
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax)
    return q * scale


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


d_out, d_in, n_cal = 64, 128, 512

# a handful of salient (high-activation) input channels, as seen in real transformer
# activations -- these dominate the output and are the ones AWQ protects
X = np.random.randn(n_cal, d_in).astype(np.float32)
salient = np.random.choice(d_in, 8, replace=False)
X[:, salient] *= 20.0

W = (np.random.randn(d_out, d_in) * 0.1).astype(np.float32)
Y_true = X @ W.T

# naive RTN, per-output-channel int4
W_rtn = rtn_quant_per_channel(W, axis=0, bits=4)
Y_rtn = X @ W_rtn.T
err_rtn = rel_err(Y_rtn, Y_true)

# AWQ: scale up salient (high-activation) weight columns before quant, undo the scale on the
# activation side -- math is unchanged pre-quant, but quant error moves off the salient channels
act_scale = np.abs(X).mean(axis=0)

best_err, best_alpha = None, None
for alpha in np.linspace(0, 1, 11):
    s = act_scale ** alpha
    s = s / np.sqrt(s.max() * s.min())  # keep dynamic range centered
    W_scaled = W * s[None, :]
    Wq_scaled = rtn_quant_per_channel(W_scaled, axis=0, bits=4)
    W_eff = Wq_scaled / s[None, :]
    Y_awq = X @ W_eff.T
    e = rel_err(Y_awq, Y_true)
    if best_err is None or e < best_err:
        best_err, best_alpha = e, alpha

assert best_err < err_rtn, f"AWQ should beat naive RTN: {best_err} >= {err_rtn}"
print(f"PASS awq_toy 4-bit output rel_err: awq={best_err:.6f} (alpha={best_alpha:.1f}) < rtn={err_rtn:.6f} (reduction={100*(1-best_err/err_rtn):.1f}%)")
