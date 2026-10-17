"""Reflection/transmission off a dielectric half-slab vs analytic Fresnel (TMz = s-polarisation).
Coarse grid 256 x 64 cells (4 x 1 units, h = 1/64). eps = 1 for x < X_I, eps = 4 for X_I <= x < 4
(X_I = 2 + 0.3h, so the interface cuts a cell; coarse eps is the cell average). A Gaussian-envelope
plane-wave packet (8 cells per wavelength, envelope sigma = 1 wavelength in x) starts at x = 1 moving
at angle theta, with ky = 2 pi m so it is periodic in y: sin(theta) = m/8, m = 0, 2, 4, 6.
After the reflected packet has separated (its centre 0.6 left of the interface) we measure
R = energy(x < X_I) / energy(0) and T = energy(x >= X_I) / energy(0) with 1/2(eps Ez^2 + H^2)
and compare with |r_s|^2, r_s = (cos ti - n cos tt) / (cos ti + n cos tt), n = 2."""
import math
import torch
from .yee import COURANT

EPS2, X0, LAMBDA = 4.0, 1.0, 8.0 / 64


def slab_eps(NX, NY, H, X_I, device="cuda"):
    edges = torch.arange(NX + 1, device=device, dtype=torch.float64) * H
    frac = ((edges[1:] - X_I).clamp(0, H) / H)       # fraction of each cell inside the slab
    eps_x = 1.0 + (EPS2 - 1.0) * frac
    return eps_x[:, None].expand(NX, NY).float().contiguous()[None]


def analytic_r(theta):
    n = math.sqrt(EPS2)
    ct = math.sqrt(1 - (math.sin(theta) / n) ** 2)
    r = (math.cos(theta) - n * ct) / (math.cos(theta) + n * ct)
    return r * r


@torch.no_grad()
def reflect(stepper_for_eps, m, scale=1):
    """stepper_for_eps(eps) -> object with .step(ez, hx, hy, inv_eps). scale > 1 refines the grid
    (same physical problem) and is only used to check the harness with Yee."""
    NX, NY, H = 256 * scale, 64 * scale, 1.0 / (64 * scale)
    X_I = 2.0 + 0.3 / 64
    eps = slab_eps(NX, NY, H, X_I)
    dt = COURANT * H
    k = 2 * math.pi / LAMBDA
    ky = 2 * math.pi * m
    kx = math.sqrt(k * k - ky * ky)
    theta = math.asin(ky / k)
    sig = LAMBDA
    cx = (torch.arange(NX, device="cuda", dtype=torch.float64) + 0.5) * H
    cy = (torch.arange(NY, device="cuda", dtype=torch.float64) + 0.5) * H
    X, Y = torch.meshgrid(cx, cy, indexing="ij")
    Xf, Yf = X + 0.5 * H, Y + 0.5 * H

    def f(x, y, shift):
        return torch.exp(-((x - X0) ** 2) / (2 * sig * sig)) * torch.cos(kx * x + ky * y - shift)

    ez = f(X, Y, 0).float()[None]
    hx = (ky / k * f(X, Yf, k * dt / 2)).float()[None]
    hy = (-kx / k * f(Xf, Y, k * dt / 2)).float()[None]
    stepper = stepper_for_eps(eps)
    step = torch.compile(stepper.step)
    inv_eps = 1.0 / eps

    def energies(ez, hx, hy):
        u = 0.5 * (eps * ez * ez + hx * hx + hy * hy)[0]
        left = X < X_I
        return u[left].sum().item(), u[~left].sum().item()

    u0 = sum(energies(ez, hx, hy))
    t_end = (X_I - X0) / math.cos(theta) + 0.6 / math.cos(theta)
    for _ in range(int(round(t_end / dt))):
        ez, hx, hy = step(ez, hx, hy, inv_eps)
    left, right = energies(ez, hx, hy)
    return {"theta_deg": math.degrees(theta), "R": left / u0, "T": right / u0, "R_fresnel": analytic_r(theta),
            "R_abs_err": abs(left / u0 - analytic_r(theta))}


def fresnel_table(stepper_for_eps):
    return [reflect(stepper_for_eps, m) for m in (0, 2, 4, 6)]
