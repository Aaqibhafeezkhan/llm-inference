import numpy as np

np.random.seed(0)

VOCAB, D_MODEL, N_HEADS = 64, 16, 4
DH = D_MODEL // N_HEADS
PROMPT = [3, 7, 19, 2, 8, 30, 11, 5, 17, 4, 9, 21, 1, 15, 6, 33]
OTHER_PROMPT = [12, 8, 2, 19]
OTHER_DECODE = [4, 7, 1, 9, 3, 5, 2, 8, 6, 11, 14, 10, 13, 15, 16]
CHUNK_SIZES = (1, 2, 4, 8)


def rand(*shape):
    return np.random.randn(*shape) * 0.1


E = rand(VOCAB, D_MODEL)
W_POS = rand(128, D_MODEL)
WQ = rand(D_MODEL, D_MODEL)
WK = rand(D_MODEL, D_MODEL)
WV = rand(D_MODEL, D_MODEL)
W_OUT = rand(D_MODEL, VOCAB)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attend(q, k, v, query_positions, key_positions):
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(DH)
    mask = key_positions[None, :] > query_positions[:, None]
    scores = np.where(mask[None, :, :], -np.inf, scores)
    probs = softmax(scores, axis=-1)
    return np.einsum("hts,shd->thd", probs, v).reshape(len(query_positions), D_MODEL)


def project(tokens, start):
    x = E[np.asarray(tokens)] + W_POS[start:start + len(tokens)]
    return (
        (x @ WQ).reshape(-1, N_HEADS, DH),
        (x @ WK).reshape(-1, N_HEADS, DH),
        (x @ WV).reshape(-1, N_HEADS, DH),
    )


def prefill_full(tokens):
    n = len(tokens)
    q, k, v = project(tokens, 0)
    positions = np.arange(n)
    hidden = attend(q, k, v, positions, positions)
    return hidden @ W_OUT, k, v


def decode_step(token, position, cached_k, cached_v):
    q, k_new, v_new = project([token], position)
    all_k = np.concatenate((cached_k, k_new), axis=0)
    all_v = np.concatenate((cached_v, v_new), axis=0)
    hidden = attend(q, all_k, all_v, np.array([position]), np.arange(position + 1))
    return hidden @ W_OUT, all_k, all_v


def prefill_chunked(tokens, chunk_size, other_cache):
    logits_parts = []
    interleaved_logits = []
    cached_k = np.empty((0, N_HEADS, DH))
    cached_v = np.empty((0, N_HEADS, DH))
    _, other_k, other_v = other_cache
    other_steps = 0
    max_decode_wait = 0

    for start in range(0, len(tokens), chunk_size):
        end = min(start + chunk_size, len(tokens))
        q, k_new, v_new = project(tokens[start:end], start)
        all_k = np.concatenate((cached_k, k_new), axis=0)
        all_v = np.concatenate((cached_v, v_new), axis=0)
        query_positions = np.arange(start, end)
        key_positions = np.arange(end)
        hidden = attend(q, all_k, all_v, query_positions, key_positions)
        logits_parts.append(hidden @ W_OUT)
        cached_k, cached_v = all_k, all_v

        if end < len(tokens):
            max_decode_wait = max(max_decode_wait, end - start)
            token = OTHER_DECODE[other_steps]
            other_logits, other_k, other_v = decode_step(
                token, len(OTHER_PROMPT) + other_steps, other_k, other_v
            )
            interleaved_logits.append(other_logits)
            other_steps += 1

    decoded_logits = (
        np.concatenate(interleaved_logits, axis=0)
        if interleaved_logits
        else np.empty((0, VOCAB))
    )
    return (
        np.concatenate(logits_parts),
        cached_k,
        cached_v,
        other_steps,
        max_decode_wait,
        decoded_logits,
    )


if __name__ == "__main__":
    full_logits, full_k, full_v = prefill_full(PROMPT)
    other_cache = prefill_full(OTHER_PROMPT)
    errors = []
    decode_waits = []

    assert len(PROMPT) > max(CHUNK_SIZES)

    for chunk_size in CHUNK_SIZES:
        (
            chunk_logits,
            chunk_k,
            chunk_v,
            interleaved,
            max_decode_wait,
            interleaved_logits,
        ) = prefill_chunked(PROMPT, chunk_size, other_cache)

        next_full, _, _ = decode_step(23, len(PROMPT), full_k, full_v)
        next_chunk, _, _ = decode_step(23, len(PROMPT), chunk_k, chunk_v)

        _, other_k, other_v = other_cache
        standalone_logits = []
        for step, token in enumerate(OTHER_DECODE[:interleaved]):
            logits, other_k, other_v = decode_step(
                token, len(OTHER_PROMPT) + step, other_k, other_v
            )
            standalone_logits.append(logits)

        expected_other_logits = (
            np.concatenate(standalone_logits, axis=0)
            if standalone_logits
            else np.empty((0, VOCAB))
        )
        other_error = (
            np.max(np.abs(expected_other_logits - interleaved_logits))
            if interleaved
            else 0.0
        )

        error = max(
            np.max(np.abs(full_k - chunk_k)),
            np.max(np.abs(full_v - chunk_v)),
            np.max(np.abs(full_logits - chunk_logits)),
            np.max(np.abs(next_full - next_chunk)),
            other_error,
        )
        errors.append(error)
        decode_waits.append(max_decode_wait)

        assert error < 1e-10, (chunk_size, error)
        assert interleaved == (len(PROMPT) - 1) // chunk_size
        assert max_decode_wait <= chunk_size
        assert np.allclose(interleaved_logits, expected_other_logits, atol=1e-10, rtol=0.0)

        print(
            f"chunk_size={chunk_size}: max_err={error:.2e}, "
            f"interleaved decode steps={interleaved}, "
            f"max decode wait={max_decode_wait} token slots"
        )

    max_err = max(errors)
    max_chunked_wait = max(decode_waits)
    print(
        f"chunked_prefill PASS max_err={max_err:.2e} "
        "(chunked KV, prefill logits, next-token logits and interleaved decode == references)"
    )
    print(
        f"decode wait without chunking: {len(PROMPT)} token slots; "
        f"measured with chunking: at most {max_chunked_wait} token slots"
    )
