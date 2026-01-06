import numpy as np

np.random.seed(0)


def quant_per_token_int8(x):
    # per-row (per-token) symmetric scale, x: (n_tokens, d)
    qmax = 127
    amax = np.maximum(np.abs(x).max(axis=1, keepdims=True), 1e-8)
    scale = amax / qmax
    q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def dequant(q, scale):
    return q.astype(np.float32) * scale


def attention(Q, K, V):
    d = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d)
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ V


n_q, n_kv, d = 32, 512, 64
Q = np.random.randn(n_q, d).astype(np.float32)
K = np.random.randn(n_kv, d).astype(np.float32)
V = np.random.randn(n_kv, d).astype(np.float32)

out_fp = attention(Q, K, V)

# KV cache stored int8 per-token, dequantized on read before the QK^T / softmax(.)@V matmuls
Kq, sk = quant_per_token_int8(K)
Vq, sv = quant_per_token_int8(V)
K_deq = dequant(Kq, sk)
V_deq = dequant(Vq, sv)

out_q = attention(Q, K_deq, V_deq)

rel_err = np.linalg.norm(out_q - out_fp) / np.linalg.norm(out_fp)
assert rel_err < 0.02, f"int8 KV cache attention rel_err too high: {rel_err}"
print(f"PASS kv_cache_quant int8 KV attention output rel_err={rel_err:.6f} n_kv={n_kv} d={d}")

# memory savings: int8 KV + fp32 per-token scale vs fp32 KV
bytes_fp32 = (K.size + V.size) * 4
bytes_int8 = (Kq.size + Vq.size) * 1 + (sk.size + sv.size) * 4
ratio = bytes_fp32 / bytes_int8
print(f"PASS kv_cache_quant memory ratio fp32/int8={ratio:.2f}x")
