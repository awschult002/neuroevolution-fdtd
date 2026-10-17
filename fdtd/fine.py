"""Run a fine Yee simulation and record its restriction at chosen coarse step numbers."""
import torch
from .yee import COURANT, yee_step
from .restrict import restrict_state


@torch.no_grad()
def run_fine_restricted(ez, inv_eps, factor, record_steps):
    """ez: (B, n, n) initial Ez on the fine grid; H is zero at t=0.
    Yee needs H at -dt/2, so we take a backward half step (second-order consistent start).
    Returns (B, len(record_steps), 3, n/f, n/f) coarse states (Ez at t, H at t+T/2)."""
    hx = 0.5 * COURANT * (torch.roll(ez, -1, dims=-1) - ez)
    hy = -0.5 * COURANT * (torch.roll(ez, -1, dims=-2) - ez)
    record_steps = sorted(record_steps)
    out = []
    for step in range(record_steps[-1] + 1):
        ez_start = ez
        h_a = h_b = None
        for k in range(factor):
            ez, hx, hy = yee_step(ez, hx, hy, inv_eps)
            if factor == 1:            # no restriction in time: H after the step is at t+T/2
                h_a = h_b = (hx, hy)
            elif k == factor // 2 - 1:
                h_a = (hx, hy)
            elif k == factor // 2:
                h_b = (hx, hy)
        if step in record_steps:
            hx_mid = 0.5 * (h_a[0] + h_b[0])
            hy_mid = 0.5 * (h_a[1] + h_b[1])
            out.append(restrict_state(ez_start, hx_mid, hy_mid, factor))
    return torch.stack(out, dim=1)
