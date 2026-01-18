import numpy as np

np.random.seed(0)

BLOCK = 16
N_HEADS, DH = 4, 32
SCALE = 1.0 / np.sqrt(DH)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def alloc_block_table(lengths, block_size):
    table, free = [], 0
    for n in lengths:
        n_blocks = (n + block_size - 1) // block_size
        table.append(list(range(free, free + n_blocks)))
        free += n_blocks
    return table, free


def attention(q, k, v, causal=True):
    scores = np.einsum('thd,shd->hts', q, k) * SCALE
    if causal:
        tq, tk = q.shape[0], k.shape[0]
        mask = np.triu(np.ones((tq, tk), dtype=bool), k=tk - tq + 1)
        scores = np.where(mask[None], -np.inf, scores)
    p = softmax(scores, axis=-1)
    return np.einsum('hts,shd->thd', p, v)


if __name__ == "__main__":
    lengths = [5, 16, 37, 1, 63, 20]
    block_table, n_blocks = alloc_block_table(lengths, BLOCK)

    k_pool = np.zeros((n_blocks, BLOCK, N_HEADS, DH))
    v_pool = np.zeros((n_blocks, BLOCK, N_HEADS, DH))
    ks, vs, qs = [], [], []

    for si, n in enumerate(lengths):
        k = np.random.randn(n, N_HEADS, DH)
        v = np.random.randn(n, N_HEADS, DH)
        q = np.random.randn(n, N_HEADS, DH)
        ks.append(k); vs.append(v); qs.append(q)
        for t in range(n):
            blk, slot = block_table[si][t // BLOCK], t % BLOCK
            k_pool[blk, slot] = k[t]
            v_pool[blk, slot] = v[t]

    max_err = 0.0
    for si, n in enumerate(lengths):
        blocks = block_table[si]
        gk = np.concatenate([k_pool[b] for b in blocks], axis=0)[:n]
        gv = np.concatenate([v_pool[b] for b in blocks], axis=0)[:n]
        assert np.array_equal(gk, ks[si]) and np.array_equal(gv, vs[si])  # block table reconstructs exact KV
        out_paged = attention(qs[si], gk, gv)
        out_contig = attention(qs[si], ks[si], vs[si])
        max_err = max(max_err, float(np.max(np.abs(out_paged - out_contig))))

    assert max_err < 1e-10
    print(f"paged_attention: {n_blocks} blocks of {BLOCK} tokens for seq lens {lengths}")
    print(f"paged_attention PASS max_err={max_err:.2e} (paged-KV attention == contiguous attention)")
