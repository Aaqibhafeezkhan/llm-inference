import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def mha_vectorized(x, Wq, Wk, Wv, Wo, n_heads):
    n, d = x.shape
    dh = d // n_heads
    q = (x @ Wq).reshape(n, n_heads, dh).transpose(1, 0, 2)
    k = (x @ Wk).reshape(n, n_heads, dh).transpose(1, 0, 2)
    v = (x @ Wv).reshape(n, n_heads, dh).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # causal
    scores = np.where(mask, -np.inf, q @ k.transpose(0, 2, 1) * scale)
    attn = softmax(scores, axis=-1)
    out = (attn @ v).transpose(1, 0, 2).reshape(n, d)
    return out @ Wo


def mha_reference(x, Wq, Wk, Wv, Wo, n_heads):
    # naive per-head, per-position loop, no fused matmuls
    n, d = x.shape
    dh = d // n_heads
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scale = 1.0 / np.sqrt(dh)
    out = np.zeros((n, d))
    for h in range(n_heads):
        qh, kh, vh = q[:, h*dh:(h+1)*dh], k[:, h*dh:(h+1)*dh], v[:, h*dh:(h+1)*dh]
        for i in range(n):
            scores = np.full(n, -np.inf)
            for j in range(i + 1):
                scores[j] = np.dot(qh[i], kh[j]) * scale
            a = softmax(scores)
            out[i, h*dh:(h+1)*dh] = a @ vh
    return out @ Wo


if __name__ == "__main__":
    n, d, n_heads = 6, 16, 4
    x = np.random.randn(n, d)
    Wq = np.random.randn(d, d) * 0.1
    Wk = np.random.randn(d, d) * 0.1
    Wv = np.random.randn(d, d) * 0.1
    Wo = np.random.randn(d, d) * 0.1

    out1 = mha_vectorized(x, Wq, Wk, Wv, Wo, n_heads)
    out2 = mha_reference(x, Wq, Wk, Wv, Wo, n_heads)
    err = np.max(np.abs(out1 - out2))
    assert err < 1e-10
    print(f"mha PASS max_err={err:.2e}")
