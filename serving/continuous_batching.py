import numpy as np

np.random.seed(0)


class Request:
    def __init__(self, rid, arrive, prompt_len, out_len):
        self.rid = rid
        self.arrive = arrive
        self.prompt_len = prompt_len
        self.out_len = out_len
        self.done = 0
        self.prefilled = False
        self.finish_step = None


SPECS = [
    (0, 0, 12, 6),
    (1, 0, 4, 24),
    (2, 1, 8, 10),
    (3, 1, 6, 3),
    (4, 2, 15, 18),
    (5, 2, 3, 4),
    (6, 3, 9, 12),
    (7, 4, 5, 2),
]


def make_requests():
    return [Request(*s) for s in SPECS]


def static_batching(cap):
    reqs = sorted(make_requests(), key=lambda r: r.arrive)
    step = 0
    for i in range(0, len(reqs), cap):
        batch = reqs[i:i + cap]
        step += 1  # one shared prefill step for the whole batch
        step += max(r.out_len for r in batch)  # every member locked to the slowest
        for r in batch:
            r.finish_step = step
    total_tokens = sum(r.out_len for r in reqs)
    return reqs, step, total_tokens


def continuous_batching(cap):
    reqs = sorted(make_requests(), key=lambda r: r.arrive)
    queue = list(reqs)
    active = []
    step = 0
    finished = 0
    while finished < len(reqs):
        while len(active) < cap and queue and queue[0].arrive <= step:
            active.append(queue.pop(0))
        for r in active:
            if not r.prefilled:
                r.prefilled = True  # this step's slot spent on prefill, no token out yet
                continue
            r.done += 1
            if r.done == r.out_len:
                r.finish_step = step + 1
                finished += 1
        active = [r for r in active if r.finish_step is None]  # freed slots backfill next iter
        step += 1
    total_tokens = sum(r.out_len for r in reqs)
    return reqs, step, total_tokens


if __name__ == "__main__":
    cap = 3
    static_reqs, static_steps, static_tokens = static_batching(cap)
    cont_reqs, cont_steps, cont_tokens = continuous_batching(cap)
    assert static_tokens == cont_tokens

    static_thr = static_tokens / static_steps
    cont_thr = cont_tokens / cont_steps
    finish_steps = sorted(r.finish_step for r in cont_reqs)

    print(f"requests: {[(r.rid, r.arrive, r.prompt_len, r.out_len) for r in cont_reqs]}, cap={cap}")
    print(f"static:     {static_steps} steps, throughput={static_thr:.3f} tok/step (batch waits for its slowest member)")
    print(f"continuous: {cont_steps} steps, throughput={cont_thr:.3f} tok/step, finish steps={finish_steps}")

    assert len(set(finish_steps)) > 1  # different-length requests finish at different steps
    assert cont_steps < static_steps
    assert cont_thr > static_thr
    print(f"continuous_batching PASS speedup={cont_thr / static_thr:.2f}x throughput over static batching")
