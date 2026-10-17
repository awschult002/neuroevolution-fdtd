"""Material-aware stepper (D2). Update form (Oracle):
    M_eps dE/dt = C H,    M_mu dH/dt = -C^T E,      M_mu = 1 here,
    leapfrog:  Ez += dt M_eps^-1 (A1 Hy + A2 Hx);   Hy -= dt A1^T Ez;   Hx -= dt A2^T Ez
C is a spatially varying gather operator (per-cell stencils A1, A2 of radius r) and the H step always
uses its exact transpose, so with C and M fixed in time U = 1/2(E^T M_eps E + H^(n-1/2).H^(n+1/2))
is conserved to roundoff. The network NEVER produces E and H updates separately.

Setup (once per scene): for every coarse cell take the log-eps patch of radius p around it, find the
unique patches (exact hashing of the network INPUT, no clustering), run the MLP once per unique
pattern, and store a per-cell class index (uint8, uint16 if > 255 classes). Per pattern the MLP gives
  - "C": a consistent correction to the base stencils (base = tuned fixed vacuum stencil of radius r)
  - "M": a positive factor on the cell-averaged eps: M_eps = eps_avg * softplus(o)/softplus(0)
The last layer is zero-initialised, so an untrained network equals baseline (ii)."""
import itertools, math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ACTIVATIONS, apply_transpose, consistency_projector, patches
from .yee import COURANT

SPACE = {
    "eps_radius": (0, 1, 2),
    "layers": (1, 2, 3),
    "width": (4, 8, 16, 32),
    "activation": ("tanh", "relu", "silu"),
    "stencil_radius": (1, 2, 3),
    "outputs": ("C", "M", "CM"),
}


@dataclass(frozen=True)
class MatDesign:
    eps_radius: int
    layers: int
    width: int
    activation: str
    stencil_radius: int
    outputs: str

    def name(self):
        return (f"p{self.eps_radius}_l{self.layers}_w{self.width}_{self.activation}"
                f"_r{self.stencil_radius}_{self.outputs}")


def all_designs():
    return [MatDesign(*v) for v in itertools.product(*SPACE.values())]


def mutate(design, rng):
    knob = rng.choice(list(SPACE))
    return replace(design, **{knob: rng.choice([v for v in SPACE[knob] if v != getattr(design, knob)])})


def tied_step(ez, hx, hy, a, inv_m, r):
    """a: (B, 2, k*k, h, w) per-cell stencils; inv_m: (B, h, w)."""
    ez = ez + COURANT * inv_m * ((a[:, 0] * patches(hy, r)).sum(1) + (a[:, 1] * patches(hx, r)).sum(1))
    hy = hy - COURANT * apply_transpose(a[:, 0], ez, r)
    hx = hx - COURANT * apply_transpose(a[:, 1], ez, r)
    return ez, hx, hy


class MaterialNet(nn.Module):
    def __init__(self, design, base_kernel):
        """base_kernel: (2, k*k) tuned fixed vacuum stencils of radius design.stencil_radius."""
        super().__init__()
        self.design = design
        p, w, k2 = design.eps_radius, design.width, (2 * design.stencil_radius + 1) ** 2
        act = ACTIVATIONS[design.activation]
        layers = [nn.Linear((2 * p + 1) ** 2, w), act()]
        for _ in range(design.layers - 1):
            layers += [nn.Linear(w, w), act()]
        last = nn.Linear(w, 2 * k2 + 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.mlp = nn.Sequential(*layers, last)
        self.register_buffer("base", base_kernel.clone().float())
        self.register_buffer("proj", consistency_projector(design.stencil_radius))

    def classes(self, eps):
        """Exact pattern hashing of the network input. Returns (unique patterns, per-cell index)."""
        b, h, w = eps.shape
        pat = patches(torch.log(eps), self.design.eps_radius).permute(0, 2, 3, 1).reshape(b * h * w, -1)
        uniq, inv = torch.unique(pat, dim=0, return_inverse=True)
        dtype = torch.uint8 if len(uniq) <= 255 else torch.int16
        return uniq, inv.view(b, h, w).to(dtype)

    def table(self, uniq):
        """Per-class stencils (n, 2, k*k) and M factors (n,)."""
        out = self.mlp(uniq)
        k2 = self.base.shape[1]
        da = out[:, :2 * k2].view(-1, 2, k2)
        if "C" not in self.design.outputs:
            da = da * 0
        da = torch.einsum("cij,ncj->nci", self.proj, da)
        factor = F.softplus(out[:, -1] + math.log(math.e - 1))
        if "M" not in self.design.outputs:
            factor = factor * 0 + 1
        return self.base + da, factor

    def setup(self, eps):
        """Once per scene batch: per-cell stencils a (B, 2, k*k, h, w) and inv_m (B, h, w)."""
        uniq, idx = self.classes(eps)
        a_tab, f_tab = self.table(uniq)
        i = idx.long()
        a = a_tab[i].permute(0, 3, 4, 1, 2)
        inv_m = 1.0 / (eps * f_tab[i])
        return a, inv_m, len(uniq), idx.element_size()


class BoundMaterialStepper:
    """A MaterialNet set up for one batch of scenes; step() ignores the inv_eps argument."""

    def __init__(self, a, inv_m, r):
        self.a, self.inv_m, self.r = a, inv_m, r

    def step(self, ez, hx, hy, inv_eps):
        return tied_step(ez, hx, hy, self.a, self.inv_m, self.r)

    @torch.no_grad()
    def sigma_max(self, iters=20, seed=0):
        """Power iteration on G = M^-1/2 C (M_mu = 1): largest singular value, per scene (grid units)."""
        g = torch.Generator(device=self.a.device).manual_seed(seed)
        x = torch.randn(self.inv_m.shape, device=self.a.device, generator=g)
        s = self.inv_m.sqrt()
        sig = None
        for _ in range(iters):
            x = x / x.flatten(1).norm(dim=1)[:, None, None]
            y0 = apply_transpose(self.a[:, 0], s * x, self.r)           # C^T M^-1/2 x  (Hy, Hx parts)
            y1 = apply_transpose(self.a[:, 1], s * x, self.r)
            z = s * ((self.a[:, 0] * patches(y0, self.r)).sum(1) + (self.a[:, 1] * patches(y1, self.r)).sum(1))
            sig = z.flatten(1).norm(dim=1).sqrt()                       # ||G G^T x|| -> sigma^2
            x = z
        return sig                                                       # (B,)
