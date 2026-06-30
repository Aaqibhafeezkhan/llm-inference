import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gqa(x, Wq, Wk, Wv, Wo, n_heads, n_kv_groups):
    n, d = x.shape
    dh = d // n_heads
    heads_per_group = n_heads // n_kv_groups
    q = (x @ Wq).reshape(n, n_heads, dh).transpose(1, 0, 2)
    k = (x @ Wk).reshape(n, n_kv_groups, dh).transpose(1, 0, 2)
    v = (x @ Wv).reshape(n, n_kv_groups, dh).transpose(1, 0, 2)
    k = np.repeat(k, heads_per_group, axis=0)  # broadcast kv heads to query heads
    v = np.repeat(v, heads_per_group, axis=0)
    scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, np.einsum('hnd,hmd->hnm', q, k) * scale)
    attn = softmax(scores, axis=-1)
    out = np.einsum('hnm,hmd->hnd', attn, v).transpose(1, 0, 2).reshape(n, d)
    return out @ Wo


if __name__ == "__main__":
    from mha import mha_vectorized
    from mqa import mqa

    n, d, n_heads = 6, 16, 4
    dh = d // n_heads
    x = np.random.randn(n, d)
    Wq = np.random.randn(d, d) * 0.1
    Wo = np.random.randn(d, d) * 0.1

    # n_kv_groups == n_heads -> gqa degenerates to mha
    Wk_full = np.random.randn(d, d) * 0.1
    Wv_full = np.random.randn(d, d) * 0.1
    out_gqa_full = gqa(x, Wq, Wk_full, Wv_full, Wo, n_heads, n_kv_groups=n_heads)
    out_mha = mha_vectorized(x, Wq, Wk_full, Wv_full, Wo, n_heads)
    err1 = np.max(np.abs(out_gqa_full - out_mha))
    assert err1 < 1e-10
    print(f"gqa(groups=heads) == mha PASS max_err={err1:.2e}")

    # n_kv_groups == 1 -> gqa degenerates to mqa
    Wk_1 = np.random.randn(d, dh) * 0.1
    Wv_1 = np.random.randn(d, dh) * 0.1
    out_gqa_1 = gqa(x, Wq, Wk_1, Wv_1, Wo, n_heads, n_kv_groups=1)
    out_mqa = mqa(x, Wq, Wk_1, Wv_1, Wo, n_heads)
    err2 = np.max(np.abs(out_gqa_1 - out_mqa))
    assert err2 < 1e-10
    print(f"gqa(groups=1) == mqa PASS max_err={err2:.2e}")
