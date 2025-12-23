import numpy as np

np.random.seed(0)

N_LAYERS, N_HEADS, HEAD_DIM, ELEM_BYTES = 24, 16, 64, 2
BYTES_PER_TOKEN = 2 * N_LAYERS * N_HEADS * HEAD_DIM * ELEM_BYTES  # K and V, every layer, fp16
BUDGET = 512 * 1024 * 1024


class Request:
    def __init__(self, rid, arrive, prompt_len, max_new):
        self.rid = rid
        self.arrive = arrive
        self.tokens_needed = prompt_len + max_new
        self.bytes = self.tokens_needed * BYTES_PER_TOKEN
        self.remaining = self.tokens_needed
        self.admit_step = None
        self.finish_step = None
        self.rejected = False


SPECS = [
    (0, 0, 400, 200),
    (1, 0, 900, 600),
    (2, 1, 1200, 300),
    (3, 2, 300, 100),
    (4, 2, 2000, 1000),
    (5, 3, 500, 500),
    (6, 4, 800, 200),
    (7, 5, 40000, 4000),  # bigger than the whole budget alone
    (8, 6, 150, 50),
    (9, 7, 600, 300),
]


def run():
    reqs = [Request(*s) for s in SPECS]
    rejected = [r for r in reqs if r.bytes > BUDGET]
    for r in rejected:
        r.rejected = True  # rejected at submit time, never touches the queue
    queue = sorted([r for r in reqs if not r.rejected], key=lambda r: r.arrive)

    active = []
    reserved = 0
    step = 0
    peak = 0
    finished = 0

    while finished < len(queue):
        for r in queue:
            if r.admit_step is not None or r.arrive > step:
                continue
            if r.bytes > BUDGET - reserved:
                break  # head-of-line: preserve FCFS order, wait for memory to free
            r.admit_step = step
            reserved += r.bytes
            active.append(r)

        peak = max(peak, reserved)
        assert reserved <= BUDGET

        for r in active:
            r.remaining -= 1
        done_now = [r for r in active if r.remaining == 0]
        for r in done_now:
            r.finish_step = step
            reserved -= r.bytes
            finished += 1
        active = [r for r in active if r.remaining > 0]
        step += 1

    assert all(r.finish_step is not None for r in queue)
    assert all(r.rejected for r in rejected)
    assert reserved == 0

    cap_tokens = BUDGET // BYTES_PER_TOKEN
    print(f"bytes/token={BYTES_PER_TOKEN} ({BYTES_PER_TOKEN / 1024:.1f}KB), "
          f"budget={BUDGET / 1e6:.0f}MB, cap={cap_tokens} tokens concurrently")
    print(f"admitted={len(queue)} requests, rejected={[r.rid for r in rejected]} (exceeded budget alone)")
    print(f"peak_reserved={peak / 1e6:.1f}MB ({peak / BUDGET * 100:.1f}% of budget), total_steps={step}")
    print("scheduler PASS budget invariant held every step, oversized request rejected at admission")


if __name__ == "__main__":
    run()
