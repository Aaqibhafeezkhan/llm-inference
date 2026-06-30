import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def alibi_slopes(n_heads):
    start = 2.0 ** (-8.0 / n_heads)
    return np.array([start ** (i + 1) for i in range(n_heads)])


def alibi_bias(n, slopes):
    pos = np.arange(n)
    rel = pos[None, :] - pos[:, None]  # rel[i,j] = j - i, 0 at diag, negative into the past
    return slopes[:, None, None] * rel[None, :, :]  # (h, n, n)


def attention_with_alibi(q, k, v, n_heads):
    seqlen, dh = q.shape[1], q.shape[2]
    scale = 1.0 / np.sqrt(dh)
    scores = np.einsum('hnd,hmd->hnm', q, k) * scale
    scores = scores + alibi_bias(seqlen, alibi_slopes(n_heads))
    mask = np.triu(np.ones((seqlen, seqlen), dtype=bool), k=1)
    scores = np.where(mask[None], -np.inf, scores)
    attn = softmax(scores, axis=-1)
    return np.einsum('hnm,hmd->hnd', attn, v)


if __name__ == "__main__":
    n_heads, seqlen, dh = 4, 6, 8

    slopes = alibi_slopes(n_heads)
    assert np.all(np.diff(slopes) < 0)
    print(f"alibi PASS slopes decreasing across heads: {np.round(slopes, 4)}")

    bias = alibi_bias(seqlen, slopes)
    expect = slopes[0] * (1 - 4)  # head 0, query i=4 attending to key j=1, distance 3
    err = abs(bias[0, 4, 1] - expect)
    assert err < 1e-12
    print(f"alibi PASS bias formula matches at (h=0,i=4,j=1) err={err:.2e}")

    # further-back keys get a more negative bias, for a fixed head and query
    row = bias[0, 5, :6]
    assert np.all(np.diff(row) > 0)  # increasing as j goes from 0 (far) to 5 (near)
    print(f"alibi PASS bias monotonically less negative as distance shrinks: {np.round(row, 3)}")

    # same distance penalized more by a head with a larger slope
    d0 = slopes[0] * (0 - 5)
    d_last = slopes[-1] * (0 - 5)
    assert d0 < d_last
    print(f"alibi PASS larger-slope head penalizes distance-5 more ({d0:.3f} < {d_last:.3f})")

    q = np.random.randn(n_heads, seqlen, dh)
    k = np.random.randn(n_heads, seqlen, dh)
    v = np.random.randn(n_heads, seqlen, dh)
    out = attention_with_alibi(q, k, v, n_heads)
    assert out.shape == (n_heads, seqlen, dh) and not np.any(np.isnan(out))
    print(f"alibi PASS forward pass shape={out.shape} no nans")
