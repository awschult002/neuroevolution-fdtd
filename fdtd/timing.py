"""Wall-clock timing of one compiled step, following docs/TEST_SPEC.md T-TIME-2 as far as practical.

torch.compile (Inductor) turns the Yee step into fused Triton kernels; for learned models the
elementwise/stencil work is fused but each conv stays a separate cuDNN call, so a learned step is
a handful of kernels, not one (a documented deviation from T-TIME-1).
Each run: 50 steps; SM clock + temperature read from nvidia-smi right after the run. A run whose
clock is >10% below the warm-up clock, or during which another process uses the GPU, is discarded and repeated (up to 10 extra tries).
Result = median of 5 valid runs.
"""
import os, statistics, subprocess, time
import torch
from .yee import COURANT


def gpu_clock_and_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def other_gpu_processes():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def warm_up(seconds=30.0):
    """Busy the GPU for `seconds`, return the SM clock reached."""
    a = torch.randn(2048, 2048, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        a = torch.tanh(a @ a * 1e-3)
        torch.cuda.synchronize()
    return gpu_clock_and_temp()[0]


@torch.no_grad()
def seconds_per_step(stepper, n, warm_clock, steps=50, valid_runs=5):
    step = torch.compile(stepper.step)
    state = 0.01 * torch.randn(1, 3, n, n, device="cuda")
    ez, hx, hy = state[:, 0].contiguous(), state[:, 1].contiguous(), state[:, 2].contiguous()
    for _ in range(20):                               # compile + re-warm
        ez, hx, hy = step(ez, hx, hy, 1.0)
    runs, log = [], []
    for _ in range(valid_runs + 10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            ez, hx, hy = step(ez, hx, hy, 1.0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps
        clock, temp = gpu_clock_and_temp()
        others = other_gpu_processes()
        valid = clock >= 0.9 * warm_clock and not others
        log.append({"s_per_step": dt, "sm_clock": clock, "temp_c": temp, "other_gpu_pids": others, "valid": valid})
        if valid:
            runs.append(dt)
        if len(runs) == valid_runs:
            break
    return statistics.median(runs) if runs else float("nan"), log


def seconds_per_unit_time(stepper, n, warm_clock, fallback_n=None):
    """Median wall-clock to advance an n^2 grid by one simulated time unit (n/COURANT steps).
    If n^2 does not fit in memory, time fallback_n^2 and scale by cell count (reported)."""
    try:
        s, log = seconds_per_step(stepper, n, warm_clock)
        return s * n / COURANT, n, log
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        s, log = seconds_per_step(stepper, fallback_n, warm_clock)
        return s * (n / fallback_n) ** 2 * n / COURANT, fallback_n, log
