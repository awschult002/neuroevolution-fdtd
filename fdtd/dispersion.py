"""Dispersion check (T-DISP-1): phase error per wavelength of single plane waves at 4, 8 and 16
coarse cells per wavelength on the 64^2 vacuum grid, along x and at 45 degrees.
Initial state: the exact wave Ez = cos(k.x) at cell centres (t=0) and Hx = khat_y f, Hy = -khat_x f
at their faces at t = dt/2. Along 45 deg the periodic box only allows kx = ky = 2 pi m, so the
nearest m is used and the actual cells/wavelength is reported. We track the phase of
c(t) = sum Ez exp(-i k.x) every step (unwrapped) against the exact -|k| t.
Reported: phase error [rad] per wavelength travelled (positive = lagging) and |c(T)|/|c(0)|."""
import math
import torch
from .yee import COURANT

N, STEPS = 64, 256


@torch.no_grad()
def phase_error(stepper, cells_per_wavelength, diagonal=False):
    h = 1.0 / N
    dt = COURANT * h
    if diagonal:
        m = round(N / (cells_per_wavelength * math.sqrt(2)))
        kx = ky = 2 * math.pi * m
    else:
        kx, ky = 2 * math.pi / (cells_per_wavelength * h), 0.0
    k = math.hypot(kx, ky)
    c = (torch.arange(N, device="cuda", dtype=torch.float64) + 0.5) * h
    f = c + 0.5 * h
    X, Y = torch.meshgrid(c, c, indexing="ij")
    Xf, Yf = torch.meshgrid(f, f, indexing="ij")
    ez = torch.cos(kx * X + ky * Y).float()[None]
    hx = (ky / k * torch.cos(kx * X + ky * Yf - k * dt / 2)).float()[None]
    hy = (-kx / k * torch.cos(kx * Xf + ky * Y - k * dt / 2)).float()[None]
    basis = torch.exp(-1j * (kx * X + ky * Y))
    step = torch.compile(stepper.step)

    def coeff(e):
        return (e[0].double() * basis).sum()

    c0 = coeff(ez)
    phase, prev = 0.0, torch.angle(c0).item()
    for _ in range(STEPS):
        ez, hx, hy = step(ez, hx, hy, 1.0)
        a = torch.angle(coeff(ez)).item()
        phase += (a - prev + math.pi) % (2 * math.pi) - math.pi
        prev = a
    t = STEPS * dt
    lam = 2 * math.pi / k
    err = phase + k * t
    return {"cells_per_wavelength": lam / h, "phase_err_per_wavelength": err / (t / lam),
            "amplitude_ratio": (coeff(ez).abs() / c0.abs()).item(), "finite": bool(torch.isfinite(ez).all())}


def dispersion_table(stepper):
    out = {}
    for c in (4, 8, 16):
        out[f"x_{c}"] = phase_error(stepper, c)
        out[f"diag_{c}"] = phase_error(stepper, c, diagonal=True)
    return out
