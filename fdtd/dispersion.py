"""Dispersion check (T-DISP-1): phase error per wavelength of single plane waves at 4, 8 and 16
coarse cells per wavelength on the 64^2 vacuum grid, at 0, 22.5 and 45 degrees.
Initial state: the exact wave Ez = cos(k.x) at cell centres (t=0) and Hx = khat_y f, Hy = -khat_x f
at their faces at t = dt/2. Off-axis, the nearest periodic wave is used (actual values reported). We track the phase of
c(t) = sum Ez exp(-i k.x) every step (unwrapped) against the exact -|k| t.
Reported: phase error [rad] per wavelength travelled (positive = lagging) and |c(T)|/|c(0)|."""
import math
import torch
from .yee import COURANT

N, STEPS = 64, 256


@torch.no_grad()
def phase_error(stepper, cells_per_wavelength, angle_deg=0.0):
    """The periodic box allows kx = 2 pi m, ky = 2 pi n; the nearest (m, n) to the requested
    wavelength and angle is used and the actual values are reported."""
    h = 1.0 / N
    dt = COURANT * h
    kk = N / cells_per_wavelength
    m = round(kk * math.cos(math.radians(angle_deg)))
    n = round(kk * math.sin(math.radians(angle_deg)))
    kx, ky = 2 * math.pi * m, 2 * math.pi * n
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
    return {"cells_per_wavelength": lam / h, "angle_deg": math.degrees(math.atan2(ky, kx)),
            "phase_err_per_wavelength": err / (t / lam),
            "amplitude_ratio": (coeff(ez).abs() / c0.abs()).item(), "finite": bool(torch.isfinite(ez).all())}


def dispersion_table(stepper):
    out = {}
    for c in (4, 8, 16):
        for ang in (0.0, 22.5, 45.0):
            out[f"a{ang:g}_{c}"] = phase_error(stepper, c, ang)
    return out
