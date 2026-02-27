import numpy as np

np.random.seed(0)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def temperature_probs(logits, T):
    return softmax(logits / T)


def entropy(p):
    return -np.sum(p * np.log(p + 1e-12))


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1, -1.0])

    p_low, p_high = temperature_probs(logits, 0.1), temperature_probs(logits, 5.0)
    assert entropy(p_high) > entropy(p_low)
    print(f"temperature PASS entropy(T=5.0)={entropy(p_high):.4f} > entropy(T=0.1)={entropy(p_low):.4f}")

    p_sharp = temperature_probs(logits, 1e-4)
    assert np.argmax(p_sharp) == np.argmax(logits) and p_sharp[np.argmax(logits)] > 0.999
    print(f"temperature PASS T->0 collapses to argmax, top_prob={p_sharp[np.argmax(logits)]:.6f}")

    err = np.max(np.abs(temperature_probs(logits, 1.0) - softmax(logits)))
    assert err < 1e-12
    print(f"temperature PASS T=1.0 matches plain softmax max_err={err:.2e}")
