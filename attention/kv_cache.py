import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def full_attention(x, Wq, Wk, Wv, n_heads):
    n, d = x.shape
    dh = d // n_heads
    q = (x @ Wq).reshape(n, n_heads, dh).transpose(1, 0, 2)
    k = (x @ Wk).reshape(n, n_heads, dh).transpose(1, 0, 2)
    v = (x @ Wv).reshape(n, n_heads, dh).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, np.einsum('hnd,hmd->hnm', q, k) * scale)
    attn = softmax(scores, axis=-1)
    out = np.einsum('hnm,hmd->hnd', attn, v)
    return out.transpose(1, 0, 2).reshape(n, d)


class KVCache:
    def __init__(self, n_heads):
        self.k = [[] for _ in range(n_heads)]
        self.v = [[] for _ in range(n_heads)]
        self.n_heads = n_heads

    def append(self, k_new, v_new):
        for h in range(self.n_heads):
            self.k[h].append(k_new[h])
            self.v[h].append(v_new[h])

    def get(self, h):
        return np.stack(self.k[h]), np.stack(self.v[h])


def decode_step(x_t, Wq, Wk, Wv, cache, n_heads, dh):
    q_t = (x_t @ Wq).reshape(n_heads, dh)
    k_t = (x_t @ Wk).reshape(n_heads, dh)
    v_t = (x_t @ Wv).reshape(n_heads, dh)
    cache.append(k_t, v_t)
    scale = 1.0 / np.sqrt(dh)
    out = np.zeros(n_heads * dh)
    for h in range(n_heads):
        kh, vh = cache.get(h)
        scores = (kh @ q_t[h]) * scale
        a = softmax(scores)
        out[h*dh:(h+1)*dh] = a @ vh
    return out


if __name__ == "__main__":
    n, d, n_heads = 8, 16, 4
    dh = d // n_heads
    x = np.random.randn(n, d)
    Wq = np.random.randn(d, d) * 0.1
    Wk = np.random.randn(d, d) * 0.1
    Wv = np.random.randn(d, d) * 0.1

    full_out = full_attention(x, Wq, Wk, Wv, n_heads)

    cache = KVCache(n_heads)
    cached_out = np.zeros((n, d))
    for t in range(n):
        cached_out[t] = decode_step(x[t], Wq, Wk, Wv, cache, n_heads, dh)

    err = np.max(np.abs(full_out - cached_out))
    assert err < 1e-10
    print(f"kv_cache PASS max_err={err:.2e} (incremental decode == full recompute over prefix)")
