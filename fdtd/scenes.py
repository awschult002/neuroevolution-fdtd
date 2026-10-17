"""Material scenes (D2): dielectric blocks in vacuum plus Gaussian pulses, mu = 1.
Shapes are defined analytically and point-sampled on the 512^2 truth grid; coarser grids use the
cell average of that eps (for TMz, Ez is tangential to every interface, so the cell-averaged eps is
the standard subpixel-smoothed value).
Training/validation shapes: axis-aligned rectangles. Unseen test shapes: rotated rectangles and L-shapes."""
import math
import torch
from .initial import random_pulses


def _rect(rng):
    return {"type": "rect", "cx": rng.random(), "cy": rng.random(), "hx": rng.uniform(4, 16) / 64,
            "hy": rng.uniform(4, 16) / 64, "eps": rng.uniform(1.5, 8.0)}


def random_scene(rng, shapes="rect"):
    blocks = []
    for _ in range(rng.randint(2, 4)):
        b = _rect(rng)
        if shapes == "unseen":
            if rng.random() < 0.5:
                b["type"], b["angle"] = "rotated", rng.uniform(math.radians(15), math.radians(75))
            else:
                b["type"] = "L"                         # rectangle minus its upper-right quadrant
        blocks.append(b)
    pulses = random_pulses(rng)["pulses"]
    pulses = [(x, y, max(s, 4 / 64), a) for x, y, s, a in pulses]
    return {"kind": "scene", "blocks": blocks, "pulses": pulses}


def extra_test_scenes(rng):
    """T-MAT-3: geometries outside training. 2 scenes with contrast above the training range
    (eps 10-12), 2 with a thin strip exactly one coarse cell wide, 2 whose block edges sit exactly
    mid-cell on the 64^2 grid."""
    out = []
    for kind in ("high_contrast", "high_contrast", "thin", "thin", "midcell", "midcell"):
        sc = random_scene(rng)
        if kind == "high_contrast":
            for b in sc["blocks"]:
                b["eps"] = rng.uniform(10.0, 12.0)
        elif kind == "thin":
            sc["blocks"][0].update(hx=0.5 / 64, hy=rng.uniform(10, 20) / 64)
        else:
            for b in sc["blocks"]:
                b["cx"] = (round(b["cx"] * 64) + 0.5) / 64
                b["hx"] = round(b["hx"] * 64) / 64       # edges at cx +- hx = half-integer cells
                b["cy"] = (round(b["cy"] * 64) + 0.5) / 64
                b["hy"] = round(b["hy"] * 64) / 64
        sc["test_kind"] = kind
        out.append(sc)
    return out


def _delta(c, centre):
    return torch.remainder(c - centre + 0.5, 1.0) - 0.5


def sample_scene(scene, n, device):
    """Returns (ez, eps) point-sampled at Ez cell centres of an n x n grid."""
    c = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n
    x, y = torch.meshgrid(c, c, indexing="ij")
    eps = torch.ones(n, n, device=device, dtype=torch.float64)
    for b in scene["blocks"]:
        dx, dy = _delta(x, b["cx"]), _delta(y, b["cy"])
        if b["type"] == "rotated":
            ca, sa = math.cos(b["angle"]), math.sin(b["angle"])
            dx, dy = ca * dx + sa * dy, -sa * dx + ca * dy
        inside = (dx.abs() < b["hx"]) & (dy.abs() < b["hy"])
        if b["type"] == "L":
            inside &= ~((dx > 0) & (dy > 0))
        eps[inside] = b["eps"]
    ez = torch.zeros(n, n, device=device, dtype=torch.float64)
    for px, py, s, a in scene["pulses"]:
        dx, dy = _delta(x, px), _delta(y, py)
        ez += a * torch.exp(-(dx * dx + dy * dy) / (2 * s * s))
    return ez.float(), eps.float()


def average(field, f):
    b, n, _ = field.shape
    return field.reshape(b, n // f, f, n // f, f).mean(dim=(2, 4))


def interface_mask(eps_fine, f, band=2):
    """T-MAT-4 edge band: coarse cells within +-band coarse cells of any cell whose fine eps varies."""
    mx = torch.nn.functional.max_pool2d(eps_fine[:, None], f)
    mn = -torch.nn.functional.max_pool2d(-eps_fine[:, None], f)
    var = (mx - mn) > 1e-6
    var = torch.nn.functional.max_pool2d(torch.nn.functional.pad(var.float(), (band,) * 4, mode="circular"),
                                         2 * band + 1, stride=1)
    return var[:, 0] > 0
