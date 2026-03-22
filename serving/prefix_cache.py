import numpy as np

np.random.seed(2)

VOCAB, D_MODEL, N_HEADS, N_LAYERS, D_FF = 40, 24, 4, 2, 48
DH = D_MODEL // N_HEADS


def rand(*shape):
    return np.random.randn(*shape) * 0.02


E = rand(VOCAB, D_MODEL)
W_POS = rand(256, D_MODEL)
LAYERS = [dict(
    wq=rand(D_MODEL, D_MODEL), wk=rand(D_MODEL, D_MODEL), wv=rand(D_MODEL, D_MODEL), wo=rand(D_MODEL, D_MODEL),
    w1=rand(D_MODEL, D_FF), w2=rand(D_FF, D_MODEL),
    g1=np.ones(D_MODEL), b1=np.zeros(D_MODEL), g2=np.ones(D_MODEL), b2=np.zeros(D_MODEL),
) for _ in range(N_LAYERS)]
W_OUT = rand(D_MODEL, VOCAB)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def split_heads(x):
    return x.reshape(x.shape[0], N_HEADS, DH)


def causal_attend(q, k, v, q_offset):
    scale = 1.0 / np.sqrt(DH)
    scores = np.einsum('thd,shd->hts', q, k) * scale
    tq, tk = q.shape[0], k.shape[0]
    idx_q = np.arange(tq)[:, None] + q_offset
    idx_k = np.arange(tk)[None, :]
    scores = np.where((idx_k > idx_q)[None], -np.inf, scores)
    p = softmax(scores, axis=-1)
    return np.einsum('hts,shd->thd', p, v).reshape(tq, D_MODEL)


def run_layers(x, kv_prefix=None):
    q_offset = kv_prefix[0][0].shape[0] if kv_prefix else 0
    caches = []
    for li, layer in enumerate(LAYERS):
        q = split_heads(x @ layer['wq'])
        k = split_heads(x @ layer['wk'])
        v = split_heads(x @ layer['wv'])
        if kv_prefix:
            pk, pv = kv_prefix[li]
            k = np.concatenate([pk, k], axis=0)  # cached prefix K, untouched, plus fresh suffix K
            v = np.concatenate([pv, v], axis=0)
        a = causal_attend(q, k, v, q_offset)
        x = x + a @ layer['wo']
        x = layernorm(x, layer['g1'], layer['b1'])
        h = np.maximum(x @ layer['w1'], 0)
        x = x + h @ layer['w2']
        x = layernorm(x, layer['g2'], layer['b2'])
        caches.append((k, v))
    return x, caches


def embed(tokens, start):
    return E[tokens] + W_POS[start:start + len(tokens)]


def forward_full(tokens):
    x = embed(tokens, 0)
    x, caches = run_layers(x)
    return x @ W_OUT, caches


def forward_with_prefix_cache(prefix_tokens, suffix_tokens, prefix_cache):
    x = embed(suffix_tokens, start=len(prefix_tokens))
    x, caches = run_layers(x, kv_prefix=prefix_cache)
    return x @ W_OUT, caches


if __name__ == "__main__":
    prefix = [3, 7, 19, 2, 8, 30, 11]
    suffix_a = [4, 4, 9, 21]
    suffix_b = [1, 15, 6, 33, 2]

    logits_prefix, prefix_cache = forward_full(prefix)  # computed once, reused below

    logits_a_cached, _ = forward_with_prefix_cache(prefix, suffix_a, prefix_cache)
    logits_a_full, _ = forward_full(prefix + suffix_a)
    err_a = np.max(np.abs(logits_a_cached - logits_a_full[len(prefix):]))

    logits_b_cached, _ = forward_with_prefix_cache(prefix, suffix_b, prefix_cache)
    logits_b_full, _ = forward_full(prefix + suffix_b)
    err_b = np.max(np.abs(logits_b_cached - logits_b_full[len(prefix):]))

    err_prefix = np.max(np.abs(logits_a_full[:len(prefix)] - logits_prefix))  # causal: prefix logits never see suffix

    max_err = max(err_a, err_b, err_prefix)
    assert max_err < 1e-8
    print(f"shared prefix len={len(prefix)}, suffix_a={suffix_a}, suffix_b={suffix_b}")
    print(f"cached-prefix vs full-recompute max_err: suffix_a={err_a:.2e} suffix_b={err_b:.2e} prefix_logits={err_prefix:.2e}")
    print(f"prefix_cache PASS max_err={max_err:.2e} (reused prefix KV == recomputing the whole sequence)")
