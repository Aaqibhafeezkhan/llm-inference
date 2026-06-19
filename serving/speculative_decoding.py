import numpy as np

np.random.seed(1)

VOCAB = 6
GAMMA = 4  # draft tokens proposed per round


def rand_dist(n):
    x = np.random.exponential(1.0, size=n)
    return x / x.sum()


p_list = [rand_dist(VOCAB) for _ in range(GAMMA + 1)]  # target dist per position, incl. bonus slot
q_list = [0.5 * p_list[i] + 0.5 * rand_dist(VOCAB) for i in range(GAMMA)]
q_list = [q / q.sum() for q in q_list]  # imperfect draft, correlated with target but not equal


def draft_verify_round():
    out = []
    for i in range(GAMMA):
        x = np.random.choice(VOCAB, p=q_list[i])
        alpha = min(1.0, p_list[i][x] / q_list[i][x]) if q_list[i][x] > 0 else 0.0
        if np.random.rand() < alpha:
            out.append((i, x, True))
            continue
        resid = np.maximum(p_list[i] - q_list[i], 0.0)
        resid = resid / resid.sum()
        x_corr = np.random.choice(VOCAB, p=resid)
        out.append((i, x_corr, False))
        return out  # reject stops the round, rest of the draft is discarded
    x_bonus = np.random.choice(VOCAB, p=p_list[GAMMA])  # all gamma accepted -> free bonus token
    out.append((GAMMA, x_bonus, True))
    return out


if __name__ == "__main__":
    N = 150000
    counts = [np.zeros(VOCAB) for _ in range(GAMMA + 1)]
    reached = np.zeros(GAMMA + 1)
    accepted_at = np.zeros(GAMMA)

    for _ in range(N):
        for i, x, acc in draft_verify_round():
            counts[i][x] += 1
            reached[i] += 1
            if i < GAMMA and acc:
                accepted_at[i] += 1

    max_dist_err = 0.0
    max_accept_err = 0.0
    for i in range(GAMMA + 1):
        emp = counts[i] / reached[i]
        err = np.max(np.abs(emp - p_list[i]))
        max_dist_err = max(max_dist_err, err)
        tag = "bonus" if i == GAMMA else f"draft[{i}]"
        line = f"pos {i} ({tag}): reached={int(reached[i]):>6d}  max|emp-target|={err:.4f}"
        if i < GAMMA:
            emp_alpha = accepted_at[i] / reached[i]
            theo_alpha = float(np.minimum(p_list[i], q_list[i]).sum())
            max_accept_err = max(max_accept_err, abs(emp_alpha - theo_alpha))
            line += f"  accept_rate emp={emp_alpha:.3f} theory(sum min(p,q))={theo_alpha:.3f}"
        print(line)

    avg_tokens_per_round = float(sum(reached)) / N
    print(f"avg tokens/round={avg_tokens_per_round:.3f} (range (0, {GAMMA + 1}])")

    assert max_dist_err < 0.02
    assert max_accept_err < 0.02
    print(f"speculative_decoding PASS max_dist_err={max_dist_err:.2e} max_accept_err={max_accept_err:.2e} "
          f"(emitted-token distribution == target p, acceptance rule == sum min(p,q))")
