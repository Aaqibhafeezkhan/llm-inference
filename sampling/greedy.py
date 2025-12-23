import numpy as np

np.random.seed(0)


def greedy_sample(logits):
    return int(np.argmax(logits))


if __name__ == "__main__":
    logits = np.array([1.2, 5.7, 0.3, 5.7001, -2.0])
    tok = greedy_sample(logits)
    assert tok == 3
    print(f"greedy PASS tok={tok} logit={logits[tok]}")

    ok = True
    for _ in range(20):
        logits = np.random.randn(50)
        tok = greedy_sample(logits)
        ok &= logits[tok] == np.max(logits)
    assert ok
    print("greedy PASS 20 random trials, always returns the argmax token")
