import numpy as np

np.random.seed(0)


def apply_repetition_penalty(logits, seen_tokens, penalty):
    out = logits.copy()
    for t in set(seen_tokens):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty
    return out


if __name__ == "__main__":
    logits = np.array([2.0, -1.5, 0.5, -3.0, 4.0])
    seen = [0, 1, 4, 4, 4]  # repeats of 4 don't compound, penalty applied once per token
    penalty = 1.5

    out = apply_repetition_penalty(logits, seen, penalty)
    expect = logits.copy()
    expect[0] = logits[0] / penalty
    expect[1] = logits[1] * penalty
    expect[4] = logits[4] / penalty

    err = np.max(np.abs(out - expect))
    assert err < 1e-12
    assert out[2] == logits[2] and out[3] == logits[3]  # untouched, never seen
    print(f"repetition_penalty PASS max_err={err:.2e} unseen tokens unchanged")

    assert out[0] < logits[0]  # positive logit pulled toward zero
    assert out[1] < logits[1]  # negative logit pushed further negative
    assert out[4] < logits[4]
    print("repetition_penalty PASS all seen-token logits decreased (less likely to repeat)")
