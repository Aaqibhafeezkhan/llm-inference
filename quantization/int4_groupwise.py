import numpy as np

np.random.seed(0)


def quant_int4_groupwise(w, group=64, bits=4):
    d_out, d_in = w.shape
    assert d_in % group == 0
    ng = d_in // group
    wg = w.reshape(d_out, ng, group)

    qmax = 2 ** bits - 1
    wmin = np.minimum(wg.min(axis=2, keepdims=True), 0.0)
    wmax = np.maximum(wg.max(axis=2, keepdims=True), 0.0)
    scale = np.maximum((wmax - wmin) / qmax, 1e-8)
    zp = np.clip(np.round(-wmin / scale), 0, qmax)
    q = np.clip(np.round(wg / scale + zp), 0, qmax).astype(np.uint8)
    return q, scale, zp  # q: (d_out, ng, group), scale/zp: (d_out, ng, 1)


def dequant_int4_groupwise(q, scale, zp):
    d_out, ng, group = q.shape
    wg = (q.astype(np.float32) - zp) * scale
    return wg.reshape(d_out, ng * group)


d_out, d_in, group = 96, 256, 64
w = np.random.randn(d_out, d_in).astype(np.float32) * 0.2

q, scale, zp = quant_int4_groupwise(w, group=group, bits=4)
wr = dequant_int4_groupwise(q, scale, zp)

err = np.abs(w - wr)
max_err = err.max()
mean_err = err.mean()
bound = scale.max() / 2
assert max_err <= bound + 1e-6, f"group-wise round-trip bound violated: {max_err} > {bound}"
print(f"PASS int4 groupwise (group={group}) dequant max_err={max_err:.6f} mean_err={mean_err:.6f} max_scale={scale.max():.6f}")

# compare against a single group spanning the whole row (group=d_in), i.e. coarser per-channel int4
q1, s1, z1 = quant_int4_groupwise(w, group=d_in, bits=4)
wr1 = dequant_int4_groupwise(q1, s1, z1)
err1 = np.abs(w - wr1).mean()
assert mean_err < err1, f"group-wise should beat per-channel-only: {mean_err} >= {err1}"
print(f"PASS int4 groupwise beats single-group(per-channel) baseline: grouped_mean_err={mean_err:.6f} < per_channel_mean_err={err1:.6f}")
