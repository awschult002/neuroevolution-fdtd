"""Dispersion check (Tester condition c): phase error per wavelength of a single x-directed plane
wave at 4, 8 and 16 coarse cells per wavelength on the 64^2 grid.
Initial state: Ez = cos(k x) at cell centres (t=0), Hy = -cos(k x_face - k dt/2) (t=dt/2), Hx = 0,
i.e. the exact right-going wave sampled in the Yee staggering. We advance STEPS coarse steps,
track the phase of the Fourier coefficient c(t) = sum Ez exp(-i k x) every step (unwrapped), and
compare with the exact phase -k t. Reported: phase error [rad] per wavelength travelled, and the
amplitude ratio |c(T)|/|c(0)|."""
import math
import torch
from .yee import COURANT

N, STEPS = 64, 256


@torch.no_grad()
def phase_error(stepper, cells_per_wavelength):
    h = 1.0 / N
    dt = COURANT * h
    k = 2 * math.pi / (cells_per_wavelength * h)
    xc = (torch.arange(N, device="cuda", dtype=torch.float64) + 0.5) * h
    xf = (torch.arange(N, device="cuda", dtype=torch.float64) + 1.0) * h
    ez = torch.cos(k * xc)[:, None].expand(N, N).float()[None].contiguous()
    hy = (-torch.cos(k * xf - k * dt / 2))[:, None].expand(N, N).float()[None].contiguous()
    hx = torch.zeros_like(ez)
    basis = torch.exp(-1j * k * xc)
    step = torch.compile(stepper.step)

    def coeff(e):
        return (e[0].double().mean(dim=1) * basis).sum()

    c0 = coeff(ez)
    phase, prev = 0.0, torch.angle(c0).item()
    for _ in range(STEPS):
        ez, hx, hy = step(ez, hx, hy, 1.0)
        a = torch.angle(coeff(ez)).item()
        d = (a - prev + math.pi) % (2 * math.pi) - math.pi
        phase += d
        prev = a
    t = STEPS * dt
    wavelengths = t / (cells_per_wavelength * h)
    err = phase - (-k * t)
    amp = (coeff(ez).abs() / c0.abs()).item()
    return {"phase_err_per_wavelength": err / wavelengths, "amplitude_ratio": amp,
            "finite": bool(torch.isfinite(ez).all())}


def dispersion_table(stepper):
    return {str(c): phase_error(stepper, c) for c in (4, 8, 16)}
