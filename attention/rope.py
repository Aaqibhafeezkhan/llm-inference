import numpy as np

np.random.seed(0)


def rope_freqs(dim, base=10000.0):
    return 1.0 / (base ** (np.arange(0, dim, 2) / dim))


def rotate(x, pos, freqs):
    # pairs (x0,x1),(x2,x3),... each pair rotated by pos*freq_i
    angles = pos * freqs
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


if __name__ == "__main__":
    dim = 8
    freqs = rope_freqs(dim)
    q = np.random.randn(dim)
    k = np.random.randn(dim)

    # dot(rot(q,m), rot(k,n)) depends only on (m - n), the whole point of rope
    pairs = [(5, 3), (10, 8), (2, 0), (7, 5), (100, 98)]  # all m - n == 2
    dots = [rotate(q, m, freqs) @ rotate(k, n, freqs) for m, n in pairs]
    err = max(abs(d - dots[0]) for d in dots)
    assert err < 1e-8
    print(f"rope relative-position PASS max_err={err:.2e} dots={[round(d,4) for d in dots]}")

    diff_pair_dot = rotate(q, 9, freqs) @ rotate(k, 3, freqs)  # m - n == 6, different offset
    assert abs(diff_pair_dot - dots[0]) > 1e-6
    print(f"rope PASS different relative offset gives different dot ({diff_pair_dot:.4f} != {dots[0]:.4f})")

    qm = rotate(q, 37, freqs)
    norm_err = abs(np.linalg.norm(qm) - np.linalg.norm(q))
    assert norm_err < 1e-8
    print(f"rope PASS rotation preserves norm err={norm_err:.2e}")
