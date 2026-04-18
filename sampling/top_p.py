import numpy as np

np.random.seed(0)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def top_p_probs(logits, p):
    probs = softmax(logits)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    cutoff = min(int(np.searchsorted(cum, p)) + 1, len(probs))
    keep = order[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum(), set(keep.tolist())


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.5, 3.0, -1.0, 0.2, 1.8])
    p_thresh = 0.7

    probs = softmax(logits)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    brute_cutoff = next(i + 1 for i in range(len(cum)) if cum[i] >= p_thresh)
    brute_set = set(order[:brute_cutoff].tolist())

    out_probs, kept = top_p_probs(logits, p_thresh)
    assert kept == brute_set
    assert cum[len(kept) - 1] >= p_thresh  # nucleus covers at least p
    assert len(kept) == 1 or cum[len(kept) - 2] < p_thresh  # smallest such set
    assert abs(out_probs.sum() - 1.0) < 1e-12
    print(f"top_p PASS p={p_thresh} kept={sorted(kept)} cumprob={cum[len(kept)-1]:.4f} (matches brute force)")

    for p_thresh in [0.1, 0.5, 0.9, 0.99]:
        probs = softmax(logits)
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        brute_cutoff = next(i + 1 for i in range(len(cum)) if cum[i] >= p_thresh)
        _, kept = top_p_probs(logits, p_thresh)
        assert kept == set(order[:brute_cutoff].tolist())
    print("top_p PASS holds for p in [0.1, 0.5, 0.9, 0.99]")
