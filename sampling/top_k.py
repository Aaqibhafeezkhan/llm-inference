import numpy as np

np.random.seed(0)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def top_k_probs(logits, k):
    idx = np.argpartition(logits, -k)[-k:]
    filtered = np.full_like(logits, -np.inf)
    filtered[idx] = logits[idx]
    return softmax(filtered)


if __name__ == "__main__":
    logits = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0])
    k = 3
    p = top_k_probs(logits, k)
    nonzero = {int(i) for i in np.where(p > 0)[0]}
    top_k_idx = {int(i) for i in np.argsort(logits)[-k:]}

    assert nonzero == top_k_idx and len(nonzero) == k
    assert abs(p.sum() - 1.0) < 1e-12
    print(f"top_k PASS k={k} nonzero_set={sorted(nonzero)} == true_top_k_set={sorted(top_k_idx)}")

    for k in [1, 2, 5, 7]:
        p = top_k_probs(logits, k)
        assert {int(i) for i in np.where(p > 0)[0]} == {int(i) for i in np.argsort(logits)[-k:]}
    print("top_k PASS holds for k in [1, 2, 5, 7]")
