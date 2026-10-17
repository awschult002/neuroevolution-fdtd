"""2D TMz Yee FDTD on the periodic unit square, fp32.

Grid of n x n cells, spacing h = 1/n, arrays indexed [batch, x, y].
  Ez[i, j] at ((i+1/2)h, (j+1/2)h)   cell centres,  time  t_n
  Hx[i, j] at ((i+1/2)h, (j+1)h)     y-faces,       time  t_n + dt/2
  Hy[i, j] at ((i+1)h,   (j+1/2)h)   x-faces,       time  t_n + dt/2
Units: c = eps0 = mu0 = 1; eps is relative permittivity at Ez points.
Every grid uses the same Courant number dt/h = 0.5 (stable limit is 1/sqrt(2)),
so dt = 0.5/n: the 64^2 coarse dt is 4x the 256^2 dt and 8x the 512^2 dt.
"""
import torch

COURANT = 0.5


def time_step(n):
    return COURANT / n


def yee_step(ez, hx, hy, inv_eps):
    """One leapfrog step: Ez(t_n) -> Ez(t_n+1) using H(t_n+1/2), then H -> t_n+3/2."""
    curl_h = (hy - torch.roll(hy, 1, dims=-2)) - (hx - torch.roll(hx, 1, dims=-1))
    ez = ez + COURANT * inv_eps * curl_h
    hx = hx - COURANT * (torch.roll(ez, -1, dims=-1) - ez)
    hy = hy + COURANT * (torch.roll(ez, -1, dims=-2) - ez)
    return ez, hx, hy


def energy(ez, h_prev, h_next, eps):
    """The ONE discrete energy used by every method and by the 1% gate (staggered leapfrog energy):
        U_n = 1/2 * sum(eps * Ez^n^2 + H^(n-1/2) . H^(n+1/2)) * h^2      (mu = 1)
    h_prev = (Hx, Hy) at n-1/2, h_next = (Hx, Hy) at n+1/2. For any leapfrog of the form
    E += C*inv_eps*A H, H -= C*A^T E with A fixed in time (Yee included) U_n is conserved to roundoff."""
    return 0.5 * (eps * ez * ez + h_prev[0] * h_next[0] + h_prev[1] * h_next[1]).mean(dim=(-2, -1))


def naive_energy(ez, hx, hy, eps):
    """1/2 sum(eps Ez^2 + Hx^2 + Hy^2) with H half a step off. Reported for information only."""
    return 0.5 * (eps * ez * ez + hx * hx + hy * hy).mean(dim=(-2, -1))
