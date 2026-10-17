"""Coarse rollouts and the metrics: relative L2 error and energy drift."""
import torch
from .yee import energy


def rel_l2(pred, true):
    """Mean over samples of ||pred-true|| / ||true|| over all three fields."""
    num = (pred - true).flatten(1).norm(dim=1)
    den = true.flatten(1).norm(dim=1)
    return (num / den).mean().item()


@torch.no_grad()
def rollout(stepper, state, eps, n_steps, record_steps=(), energy_every=0):
    """state: (B, 3, n, n). Returns dict with recorded states, energies, blow-up step."""
    inv_eps = 1.0 / eps
    step = torch.compile(stepper.step)    # fuses the elementwise work; ~3x faster than eager
    ez, hx, hy = state[:, 0], state[:, 1], state[:, 2]
    u0 = energy(ez, hx, hy, eps)
    records, energies = {}, []
    blew_up_at = None
    for n in range(n_steps + 1):
        if n in record_steps:
            records[n] = torch.stack([ez, hx, hy], dim=1)
        if energy_every and n % energy_every == 0:
            energies.append(energy(ez, hx, hy, eps).cpu())
        if n % 100 == 0 or n == n_steps:
            u = energy(ez, hx, hy, eps)
            if not torch.isfinite(u).all() or (u > 10 * u0).any():
                blew_up_at = n
                break
        if n == n_steps:
            break
        ez, hx, hy = step(ez, hx, hy, inv_eps)
    u_end = energy(ez, hx, hy, eps)
    drift = ((u_end - u0).abs() / u0)
    return {"records": records, "energies": energies, "blew_up_at": blew_up_at,
            "drift_mean": drift.mean().item(), "drift_max": drift.max().item()}
