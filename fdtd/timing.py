"""Wall-clock timing of one compiled step (torch.compile / Inductor generates fused Triton kernels
for the elementwise and stencil work; the 1x1/kxk convolutions stay cuDNN calls)."""
import statistics, time
import torch
from .yee import COURANT


@torch.no_grad()
def seconds_per_step(stepper, n, reps=5, steps=50):
    step = torch.compile(stepper.step)
    state = 0.01 * torch.randn(1, 3, n, n, device="cuda")
    ez, hx, hy = state[:, 0].contiguous(), state[:, 1].contiguous(), state[:, 2].contiguous()
    for _ in range(10):                               # compile + warm the GPU
        ez, hx, hy = step(ez, hx, hy, 1.0)
    runs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            ez, hx, hy = step(ez, hx, hy, 1.0)
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) / steps)
    return statistics.median(runs)


def seconds_per_unit_time(stepper, n, fallback_n=None):
    """Median wall-clock to advance an n^2 grid by one simulated time unit (n/COURANT steps).
    If n^2 does not fit in memory, time fallback_n^2 and scale by cell count (reported)."""
    try:
        return seconds_per_step(stepper, n) * n / COURANT, n
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        m = fallback_n
        return seconds_per_step(stepper, m) * (n / m) ** 2 * n / COURANT, m
