import numpy as np

np.random.seed(42)

VOCAB, D_MODEL, N_HEADS, N_LAYERS, D_FF = 64, 32, 4, 2, 64
DH = D_MODEL // N_HEADS


def rand(*shape):
    return np.random.randn(*shape) * 0.02


E = rand(VOCAB, D_MODEL)
W_POS = rand(512, D_MODEL)
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


def forward_layer(x, layer, kv_cache=None):
    q = split_heads(x @ layer['wq'])
    k = split_heads(x @ layer['wk'])
    v = split_heads(x @ layer['wv'])
    q_offset = 0
    if kv_cache is not None:
        kc, vc = kv_cache
        q_offset = kc.shape[0]
        k = np.concatenate([kc, k], axis=0)
        v = np.concatenate([vc, v], axis=0)
    a = causal_attend(q, k, v, q_offset)
    x = x + a @ layer['wo']
    x = layernorm(x, layer['g1'], layer['b1'])
    h = np.maximum(x @ layer['w1'], 0)
    x = x + h @ layer['w2']
    x = layernorm(x, layer['g2'], layer['b2'])
    return x, k, v


def forward_full(tokens):
    x = E[tokens] + W_POS[:len(tokens)]
    caches = []
    for layer in LAYERS:
        x, k, v = forward_layer(x, layer)
        caches.append((k, v))
    return x @ W_OUT, caches


def forward_step(token, pos, kv_caches):
    x = E[[token]] + W_POS[pos:pos + 1]
    new_caches = []
    for layer, cache in zip(LAYERS, kv_caches):
        x, k, v = forward_layer(x, layer, cache)
        new_caches.append((k, v))
    return x @ W_OUT, new_caches


def sample(logits, temp=0.8, top_k=8):
    logits = logits / temp
    top_idx = np.argpartition(-logits, top_k - 1)[:top_k]
    p = softmax(logits[top_idx])
    return int(np.random.choice(top_idx, p=p))


def generate(prompt, n_new):
    logits, cache = forward_full(prompt)  # prefill: one pass over the prompt, builds the KV cache
    tokens = list(prompt)
    next_logits = logits[-1]
    for _ in range(n_new):
        tok = sample(next_logits)
        tokens.append(tok)
        logits_step, cache = forward_step(tok, len(tokens) - 1, cache)  # decode: one token, cache grows by one
        next_logits = logits_step[0]
    return tokens


def check_kv_cache_equivalence(tokens):
    logits_full, _ = forward_full(tokens)
    logits_first, cache = forward_full(tokens[:1])
    incremental = [logits_first[-1]]
    for i in range(1, len(tokens)):
        logits_step, cache = forward_step(tokens[i], i, cache)
        incremental.append(logits_step[0])
    incremental = np.stack(incremental, axis=0)
    return np.max(np.abs(incremental - logits_full))


if __name__ == "__main__":
    prompt = [3, 17, 9, 44]
    tokens = generate(prompt, 12)
    err = check_kv_cache_equivalence(tokens)
    print(f"prompt={prompt}")
    print(f"generated={tokens}")
    assert err < 1e-8
    print(f"engine PASS max_err={err:.2e} (incremental decode with KV cache == full recompute at every position)")
