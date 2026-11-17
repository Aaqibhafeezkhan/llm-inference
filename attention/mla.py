import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _validate(x, Wq, Wc, Wuk, Wuv, Wo, n_heads):
    if x.ndim != 2:
        raise ValueError("x must have shape (sequence, model_dim)")
    n, d = x.shape
    if n_heads <= 0 or d % n_heads:
        raise ValueError("model_dim must be divisible by a positive n_heads")
    dh = d // n_heads
    latent_dim = Wc.shape[1]
    if Wq.shape != (d, d) or Wo.shape != (d, d):
        raise ValueError("Wq and Wo must both have shape (model_dim, model_dim)")
    if Wc.shape[0] != d or Wuk.shape != (latent_dim, d) or Wuv.shape != (latent_dim, d):
        raise ValueError("Wc, Wuk and Wuv have incompatible latent/model dimensions")
    return n, d, dh, latent_dim


def mla(x, Wq, Wc, Wuk, Wuv, Wo, n_heads):
    """Causal multi-head latent attention; cache representation is c = x @ Wc."""
    n, d, dh, _ = _validate(x, Wq, Wc, Wuk, Wuv, Wo, n_heads)
    q = (x @ Wq).reshape(n, n_heads, dh).transpose(1, 0, 2)
    latent = x @ Wc
    k = (latent @ Wuk).reshape(n, n_heads, dh).transpose(1, 0, 2)
    v = (latent @ Wuv).reshape(n, n_heads, dh).transpose(1, 0, 2)
    scores = np.einsum("hnd,hmd->hnm", q, k) / np.sqrt(dh)
    scores = np.where(np.triu(np.ones((n, n), dtype=bool), k=1), -np.inf, scores)
    attn = softmax(scores, axis=-1)
    out = np.einsum("hnm,hmd->hnd", attn, v).transpose(1, 0, 2).reshape(n, d)
    return out @ Wo


def mla_reference(x, Wq, Wc, Wuk, Wuv, Wo, n_heads):
    """Readable per-position reference that reconstructs K and V from each latent."""
    n, d, dh, _ = _validate(x, Wq, Wc, Wuk, Wuv, Wo, n_heads)
    q = (x @ Wq).reshape(n, n_heads, dh)
    latent = x @ Wc
    k = (latent @ Wuk).reshape(n, n_heads, dh)
    v = (latent @ Wuv).reshape(n, n_heads, dh)
    context = np.zeros((n, d))
    for t in range(n):
        for h in range(n_heads):
            weights = softmax(k[:t + 1, h] @ q[t, h] / np.sqrt(dh))
            context[t, h * dh:(h + 1) * dh] = weights @ v[:t + 1, h]
    return context @ Wo


class LatentKVCache:
    """Per-layer cache that stores one compressed latent vector per token."""

    def __init__(self, latent_dim, dtype=np.float64):
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.dtype = np.dtype(dtype)
        self.latents = []

    def append(self, latent):
        latent = np.asarray(latent, dtype=self.dtype)
        if latent.shape != (self.latent_dim,):
            raise ValueError(f"latent must have shape ({self.latent_dim},)")
        self.latents.append(latent.copy())

    def get(self):
        if not self.latents:
            return np.empty((0, self.latent_dim), dtype=self.dtype)
        return np.stack(self.latents)

    def __len__(self):
        return len(self.latents)

    @property
    def bytes_per_token(self):
        return self.latent_dim * self.dtype.itemsize


def mla_decode_step(x_t, Wq, Wc, Wuk, Wuv, Wo, cache, n_heads):
    """Append one latent and decode against only the compressed cache."""
    x_t = np.asarray(x_t)
    d = x_t.shape[0]
    if x_t.ndim != 1 or d % n_heads:
        raise ValueError("x_t must be a vector and model_dim divisible by n_heads")
    dh = d // n_heads
    if Wq.shape != (d, d) or Wo.shape != (d, d):
        raise ValueError("Wq and Wo must both have shape (model_dim, model_dim)")
    if Wc.shape[0] != d or Wc.shape[1] != cache.latent_dim:
        raise ValueError("Wc dimensions must match the input and cache latent_dim")
    if Wuk.shape != (cache.latent_dim, d) or Wuv.shape != (cache.latent_dim, d):
        raise ValueError("Wuk and Wuv dimensions must match the cache and model")
    cache.append(x_t @ Wc)
    latent = cache.get()
    keys = (latent @ Wuk).reshape(len(cache), n_heads, dh)
    values = (latent @ Wuv).reshape(len(cache), n_heads, dh)
    query = (x_t @ Wq).reshape(n_heads, dh)
    scores = np.einsum("lhd,hd->hl", keys, query) / np.sqrt(dh)
    weights = softmax(scores, axis=-1)
    context = np.einsum("hl,lhd->hd", weights, values).reshape(d)
    return context @ Wo


def kv_cache_bytes_per_token(n_kv_heads, head_dim, dtype=np.float64):
    """Bytes for conventional K and V tensors for one token in one layer."""
    return 2 * n_kv_heads * head_dim * np.dtype(dtype).itemsize


if __name__ == "__main__":
    n, d, n_heads = 9, 32, 8
    dh = d // n_heads
    x = np.random.randn(n, d)
    Wq = np.random.randn(d, d) * 0.1
    Wk = np.random.randn(d, d) * 0.1
    Wv = np.random.randn(d, d) * 0.1
    Wo = np.random.randn(d, d) * 0.1

    # An identity compressor is lossless: the latent retains every input dimension.
    Wc = np.eye(d)
    Wuk, Wuv = Wk.copy(), Wv.copy()
    expected = mla(x, Wq, Wc, Wuk, Wuv, Wo, n_heads)
    reference = mla_reference(x, Wq, Wc, Wuk, Wuv, Wo, n_heads)
    ref_err = np.max(np.abs(expected - reference))
    assert ref_err < 1e-10

    # With the same projections, lossless MLA must reproduce standard MHA.
    from mha import mha_vectorized

    mha_out = mha_vectorized(x, Wq, Wk, Wv, Wo, n_heads)
    mha_err = np.max(np.abs(expected - mha_out))
    assert mha_err < 1e-10

    # Incremental decoding stores only c_t, never the expanded per-head K and V.
    cache = LatentKVCache(d)
    decoded = np.zeros_like(expected)
    for t in range(n):
        decoded[t] = mla_decode_step(x[t], Wq, Wc, Wuk, Wuv, Wo, cache, n_heads)
    decode_err = np.max(np.abs(decoded - expected))
    assert decode_err < 1e-10
    assert len(cache) == n

    itemsize = np.dtype(np.float64).itemsize
    mla_bytes = cache.bytes_per_token
    mha_bytes = kv_cache_bytes_per_token(n_heads, dh)
    gqa_groups = 4
    gqa_bytes = kv_cache_bytes_per_token(gqa_groups, dh)
    assert mla_bytes == d * itemsize
    assert mha_bytes == 2 * d * itemsize
    assert gqa_bytes == 2 * gqa_groups * dh * itemsize

    print(f"mla PASS max_err={max(ref_err, mha_err, decode_err):.2e} (lossless MLA == reference == MHA; latent-cache decode == full causal attention)")
    print(f"KV cache bytes/token (one layer, float64): MLA={mla_bytes}, MHA={mha_bytes}, GQA(groups={gqa_groups})={gqa_bytes}")
