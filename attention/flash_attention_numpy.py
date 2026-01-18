import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def full_attention(q, k, v, causal=True):
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    scores = q @ k.T * scale
    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    return softmax(scores, axis=-1) @ v


def flash_attention(q, k, v, block_q=2, block_k=2, causal=True):
    # tiled attention with online softmax, never materializes the full n x n score matrix
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((n, d))
    for qi in range(0, n, block_q):
        qb = q[qi:qi+block_q]
        bq = qb.shape[0]
        m = np.full(bq, -np.inf)
        l = np.zeros(bq)
        acc = np.zeros((bq, d))
        for ki in range(0, n, block_k):
            kb, vb = k[ki:ki+block_k], v[ki:ki+block_k]
            scores = qb @ kb.T * scale
            if causal:
                row = np.arange(qi, qi + bq)[:, None]
                col = np.arange(ki, ki + kb.shape[0])[None, :]
                if np.all(col > row):
                    continue
                scores = np.where(col > row, -np.inf, scores)
            m_new = np.maximum(m, np.max(scores, axis=-1))
            p = np.exp(scores - m_new[:, None])
            alpha = np.exp(m - m_new)
            l = alpha * l + np.sum(p, axis=-1)
            acc = acc * alpha[:, None] + p @ vb
            m = m_new
        out[qi:qi+bq] = acc / l[:, None]
    return out


if __name__ == "__main__":
    n, d = 17, 8
    q = np.random.randn(n, d)
    k = np.random.randn(n, d)
    v = np.random.randn(n, d)

    ref = full_attention(q, k, v)
    got = flash_attention(q, k, v, block_q=3, block_k=4)
    err = np.max(np.abs(ref - got))
    assert err < 1e-10
    print(f"flash_attention PASS max_err={err:.2e} (n={n}, block_q=3, block_k=4)")

    got2 = flash_attention(q, k, v, block_q=5, block_k=6)
    err2 = np.max(np.abs(ref - got2))
    assert err2 < 1e-10
    print(f"flash_attention PASS max_err={err2:.2e} (n={n}, block_q=5, block_k=6, different tiling)")
