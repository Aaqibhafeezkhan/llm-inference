import numpy as np
from itertools import product

np.random.seed(0)

V, D = 4, 4  # vocab size, sequence length -- small enough to brute force (V^D = 256 paths)

init_logp = np.random.randn(V) * 0.5
trans_logp = np.random.randn(D - 1, V, V) * 0.5  # trans_logp[t, prev_token, next_token]


def seq_score(seq):
    score = init_logp[seq[0]]
    for t in range(1, D):
        score += trans_logp[t - 1, seq[t - 1], seq[t]]
    return score


def brute_force_best():
    return max(product(range(V), repeat=D), key=seq_score)


def beam_search(beam_width):
    beams = sorted(((t,), init_logp[t]) for t in range(V))
    beams.sort(key=lambda b: -b[1])
    beams = beams[:beam_width]
    for t in range(1, D):
        candidates = [
            (seq + (nxt,), score + trans_logp[t - 1, seq[-1], nxt])
            for seq, score in beams
            for nxt in range(V)
        ]
        candidates.sort(key=lambda b: -b[1])
        beams = candidates[:beam_width]
    return max(beams, key=lambda b: b[1])


if __name__ == "__main__":
    brute_seq = brute_force_best()
    brute_score = seq_score(brute_seq)

    beam_seq, beam_score = beam_search(beam_width=V)
    assert beam_seq == brute_seq
    assert abs(beam_score - brute_score) < 1e-10
    print(f"beam_search PASS width={V} found seq={beam_seq} score={beam_score:.4f}, matches brute force over all {V**D} paths")

    greedy_seq, greedy_score = beam_search(beam_width=1)
    print(f"beam_search greedy(width=1) seq={greedy_seq} score={greedy_score:.4f}", end="")
    if greedy_seq != brute_seq:
        print(f" -- suboptimal vs brute force best {brute_seq} ({brute_score:.4f}), as expected from a narrow beam")
    else:
        print(" -- happened to match optimum here")

    for w in [2, 3, 4, 8]:
        seq, score = beam_search(beam_width=w)
        assert score <= brute_score + 1e-10
        if w >= V:
            assert seq == brute_seq
    print(f"beam_search PASS wider beams never exceed brute-force optimum, width>={V} always finds it")
