import numpy as np

np.random.seed(0)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def min_p_probs(logits, min_p):
    probs = softmax(logits)
    threshold = min_p * probs.max()
    keep = probs >= threshold
    filtered = np.where(keep, probs, 0.0)
    return filtered / filtered.sum(), keep


if __name__ == "__main__":
    logits = np.array([4.0, 3.5, 1.0, 0.2, 3.9, -2.0])
    probs = softmax(logits)
    min_p = 0.3
    threshold = min_p * probs.max()

    out_probs, keep = min_p_probs(logits, min_p)
    assert np.array_equal(keep, probs >= threshold)
    assert np.all(probs[keep] >= threshold) and np.all(probs[~keep] < threshold)
    assert abs(out_probs.sum() - 1.0) < 1e-12
    print(f"min_p PASS min_p={min_p} threshold={threshold:.4f} kept={np.where(keep)[0].tolist()}")

    for min_p in [0.0, 0.05, 0.5, 0.95]:
        threshold = min_p * probs.max()
        out_probs, keep = min_p_probs(logits, min_p)
        assert np.array_equal(keep, probs >= threshold)
        assert abs(out_probs.sum() - 1.0) < 1e-12
    print("min_p PASS holds for min_p in [0.0, 0.05, 0.5, 0.95]")
