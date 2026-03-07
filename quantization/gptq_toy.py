import numpy as np

np.random.seed(0)


def rtn_quant_row(w_row, bits=4):
    qmax = 2 ** (bits - 1) - 1
    scale = max(np.abs(w_row).max(), 1e-8) / qmax
    q = np.clip(np.round(w_row / scale), -qmax - 1, qmax)
    return q * scale


def gptq_quantize(W, H, bits=4, damp=0.01):
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    scale = np.maximum(np.abs(W).max(axis=1), 1e-8) / qmax  # per-row scale, fixed upfront

    H = H.copy()
    H[np.diag_indices_from(H)] += damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)

    Wq = W.copy()
    for i in range(d_in):
        w_col = Wq[:, i]
        q_col = np.clip(np.round(w_col / scale), -qmax - 1, qmax) * scale
        err = (w_col - q_col) / Hinv[i, i]
        Wq[:, i] = q_col
        if i < d_in - 1:
            Wq[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])
    return Wq


d_out, d_in, n_cal = 64, 128, 1024

# correlated calibration activations (real activations are far from iid; this is what makes
# the Hessian off-diagonal and gives GPTQ's error compensation something to work with)
k = 16
basis = np.random.randn(k, d_in)
Z = np.random.randn(n_cal, k)
X = Z @ basis + 0.1 * np.random.randn(n_cal, d_in)
X = X.astype(np.float32)

W = (np.random.randn(d_out, d_in) * 0.1).astype(np.float32)

H = (X.T @ X / n_cal).astype(np.float64)

W_gptq = gptq_quantize(W.astype(np.float64), H, bits=4)
W_rtn = np.stack([rtn_quant_row(W[i], bits=4) for i in range(d_out)])

Y_true = X @ W.T
Y_gptq = X @ W_gptq.T
Y_rtn = X @ W_rtn.T

err_gptq = np.linalg.norm(Y_gptq - Y_true) / np.linalg.norm(Y_true)
err_rtn = np.linalg.norm(Y_rtn - Y_true) / np.linalg.norm(Y_true)

assert err_gptq < err_rtn, f"GPTQ should beat naive RTN: {err_gptq} >= {err_rtn}"
print(f"PASS gptq_toy 4-bit output rel_err: gptq={err_gptq:.6f} < rtn={err_rtn:.6f} (reduction={100*(1-err_gptq/err_rtn):.1f}%)")
