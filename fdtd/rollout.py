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
    """state: (B, 3, n, n) = (Ez^0, Hx^(1/2), Hy^(1/2)).
    The staggered energy U_n needs H^(n-1/2), which the initial state does not carry, so the
    reference is U_1 and drift at step N is |U_N - U_1| / U_1.
    Returns recorded states, energies (every energy_every steps, from n=1), blow-up step, drift."""
    inv_eps = 1.0 / eps
    step = torch.compile(stepper.step)    # fuses the elementwise work; ~3x faster than eager
    ez, hx, hy = state[:, 0], state[:, 1], state[:, 2]
    records, energies = {}, []
    blew_up_at, u_ref, u = None, None, None
    for n in range(n_steps + 1):
        if n in record_steps:
            records[n] = torch.stack([ez, hx, hy], dim=1)
        if n == n_steps:
            break
        h_prev = (hx, hy)
        ez, hx, hy = step(ez, hx, hy, inv_eps)
        m = n + 1                                        # now hold Ez^m, H^(m+1/2); h_prev = H^(m-1/2)
        if m == 1 or m % 100 == 0 or m == n_steps or (energy_every and m % energy_every == 0):
            u = energy(ez, h_prev, (hx, hy), eps)
            if m == 1:
                u_ref = u
            if energy_every and m % energy_every == 0:
                energies.append(u.cpu())
            if not torch.isfinite(u).all() or (u.abs() > 10 * u_ref.abs()).any():
                blew_up_at = m
                break
    drift = (u - u_ref).abs() / u_ref.abs()
    return {"records": records, "energies": energies, "blew_up_at": blew_up_at,
            "drift_mean": drift.mean().item(), "drift_max": drift.max().item()}
