import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def band_mask(n, W):
    # True where the query i must not look at key j: future keys, and keys older than W
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (j > i) | (j <= i - W)


def masked_full_attention(x, Wq, Wk, Wv, n_heads, W):
    n, d = x.shape
    dh = d // n_heads
    q = (x @ Wq).reshape(n, n_heads, dh).transpose(1, 0, 2)
    k = (x @ Wk).reshape(n, n_heads, dh).transpose(1, 0, 2)
    v = (x @ Wv).reshape(n, n_heads, dh).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(dh)
    scores = np.where(band_mask(n, W), -np.inf, np.einsum('hnd,hmd->hnm', q, k) * scale)
    attn = softmax(scores, axis=-1)
    return np.einsum('hnm,hmd->hnd', attn, v).transpose(1, 0, 2).reshape(n, d)


class WindowKVCache:
    def __init__(self, W):
        self.W = W
        self.k = []
        self.v = []

    def append(self, k_t, v_t):
        self.k.append(k_t)
        self.v.append(v_t)
        # drop everything older than the window, so the cache never holds more than W entries
        del self.k[:-self.W]
        del self.v[:-self.W]


def sliding_window_attention(x, Wq, Wk, Wv, n_heads, W):
    n, d = x.shape
    dh = d // n_heads
    scale = 1.0 / np.sqrt(dh)
    cache = WindowKVCache(W)
    out = np.zeros((n, d))
    max_cache = 0
    for t in range(n):
        q_t = (x[t] @ Wq).reshape(n_heads, dh)
        cache.append((x[t] @ Wk).reshape(n_heads, dh), (x[t] @ Wv).reshape(n_heads, dh))
        K = np.stack(cache.k)  # (len, n_heads, dh), len <= W
        V = np.stack(cache.v)
        attn = softmax(np.einsum('hd,lhd->hl', q_t, K) * scale, axis=-1)
        out[t] = np.einsum('hl,lhd->hd', attn, V).reshape(d)
        max_cache = max(max_cache, len(cache.k))
    return out, max_cache


if __name__ == "__main__":
    n, d, n_heads = 12, 16, 4
    x = np.random.randn(n, d)
    Wq = np.random.randn(d, d) * 0.1
    Wk = np.random.randn(d, d) * 0.1
    Wv = np.random.randn(d, d) * 0.1

    worst = 0.0
    for W in (2, 5):
        ref = masked_full_attention(x, Wq, Wk, Wv, n_heads, W)
        out, max_cache = sliding_window_attention(x, Wq, Wk, Wv, n_heads, W)
        worst = max(worst, np.max(np.abs(out - ref)))
        assert max_cache == W  # KV memory per layer is W entries, flat in sequence length

    assert worst < 1e-10
    print(f"sliding_window PASS max_err={worst:.2e} (window-W attention == full attention with band mask; cache holds W entries)")
