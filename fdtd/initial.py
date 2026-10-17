"""Initial conditions, defined analytically and point-sampled at Ez cell centres."""
import math
import torch


def random_pulses(rng):
    n_pulses = rng.randint(1, 4)
    pulses = []
    for _ in range(n_pulses):
        sigma = rng.uniform(3.0, 8.0) / 64.0          # 3..8 coarse cells
        amp = rng.uniform(0.5, 1.0) * rng.choice((-1.0, 1.0))
        pulses.append((rng.random(), rng.random(), sigma, amp))
    return {"kind": "pulses", "pulses": pulses}


def hard_wave_packet(rng):
    """Modulated pulse with wavelength = 4 coarse cells (near the coarse resolution limit)."""
    return {"kind": "wave", "x": rng.random(), "y": rng.random(), "sigma": 4.0 / 64.0,
            "wavelength": 4.0 / 64.0, "angle": rng.uniform(0.0, math.pi)}


def dielectric_case(rng):
    """Out-of-distribution case: pulses plus an eps=4 block (aligned to coarse cells)."""
    ic = random_pulses(rng)
    ic["kind"] = "dielectric"
    ic["block"] = (24 / 64.0, 40 / 64.0, 24 / 64.0, 40 / 64.0, 4.0)
    return ic


def _periodic_delta(coord, centre):
    return torch.remainder(coord - centre + 0.5, 1.0) - 0.5


def sample(ic, n, device):
    """Returns (ez, eps), each (n, n) float32."""
    c = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n
    x, y = torch.meshgrid(c, c, indexing="ij")
    ez = torch.zeros(n, n, device=device, dtype=torch.float64)
    if ic["kind"] in ("pulses", "dielectric"):
        for px, py, sigma, amp in ic["pulses"]:
            dx, dy = _periodic_delta(x, px), _periodic_delta(y, py)
            ez += amp * torch.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    else:
        dx, dy = _periodic_delta(x, ic["x"]), _periodic_delta(y, ic["y"])
        phase = 2 * math.pi * (dx * math.cos(ic["angle"]) + dy * math.sin(ic["angle"])) / ic["wavelength"]
        ez = torch.exp(-(dx * dx + dy * dy) / (2 * ic["sigma"] ** 2)) * torch.cos(phase)
    eps = torch.ones(n, n, device=device, dtype=torch.float64)
    if ic["kind"] == "dielectric":
        x0, x1, y0, y1, value = ic["block"]
        eps[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)] = value
    return ez.float(), eps.float()
