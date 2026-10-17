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
The last layer is zero-initialised, so an untrained network equals baseline (ii). Bulk cells (uniform
eps patch) always get the base stencil and M = eps exactly; the network only acts near interfaces."""
import itertools, math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ACTIVATIONS, apply_transpose, consistency_projector, patches
from .yee import COURANT

SPACE = {
    "eps_radius": (1, 2, 3),
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


def _shift_flip(a, dim):
    """b[i] = a[2r-1-i] along dim (offset d -> -1-d), assuming a[2r] = 0 (support mask)."""
    return torch.roll(torch.flip(a, dims=(dim,)), -1, dims=(dim,))


def transform_stencils(s, t, mx, my):
    """Stencils s (n, 2, k, k) = (A1 on Hy, A2 on Hx) for a pattern P, mapped to the pattern
    My^my(Mx^mx(T^t(P)))^-1 ... concretely: returns My^my Mx^mx T^t applied (in that order: T first)
    to the stencil pair, where for the geometry operations
      T  (swap x,y):  A1' = -A2^T,  A2' = -A1^T
      Mx (x -> -x) :  A1'(dx,dy) = -A1(-1-dx, dy),  A2'(dx,dy) = A2(-dx, dy)
      My (y -> -y) :  A1'(dx,dy) = A1(dx, -dy),     A2'(dx,dy) = -A2(dx, -1-dy)
    (derived from Hx = -dEz/dy, Hy = dEz/dx; checked by the equivariance test in s0b)."""
    a1, a2 = s[:, 0], s[:, 1]
    if t:
        a1, a2 = -a2.transpose(-2, -1), -a1.transpose(-2, -1)
    if mx:
        a1, a2 = -_shift_flip(a1, -2), torch.flip(a2, dims=(-2,))
    if my:
        a1, a2 = torch.flip(a1, dims=(-1,)), -_shift_flip(a2, -1)
    return torch.stack([a1, a2], dim=1)


def transform_stencils_forward(s, t, mx, my):
    """Stencil counterpart of the pattern operation T^t(Mx^mx(My^my(P))): My first, then Mx, then T."""
    return transform_stencils(transform_stencils(transform_stencils(s, 0, 0, my), 0, mx, 0), t, 0, 0)


def pattern_transforms(pat, p):
    """(8, N, q*q): T^t(Mx^mx(My^my(P))) for g = 4t + 2mx + my."""
    q = 2 * p + 1
    x = pat.view(-1, q, q)
    out = []
    for t in (0, 1):
        for mx in (0, 1):
            for my in (0, 1):
                y = x
                if my:
                    y = torch.flip(y, dims=(2,))
                if mx:
                    y = torch.flip(y, dims=(1,))
                if t:
                    y = y.transpose(1, 2)
                out.append(y.reshape(-1, q * q))
    return torch.stack(out)


def canonicalize(pat, p):
    """pat: (N, (2p+1)^2) eps patches. Returns canonical patches under D4 and, per cell, the
    group element g = 4t + 2mx + my with canonical = T^t(Mx^mx(My^my(pattern)))."""
    q = 2 * p + 1
    key_vec = torch.linspace(1.0, 2.0, q * q, device=pat.device, dtype=torch.float64)
    cands = pattern_transforms(pat, p)                            # (8, N, q*q)
    keys = (cands.double() * key_vec).sum(-1)                     # deterministic order key
    g = keys.argmin(dim=0)
    canon = cands[g, torch.arange(pat.shape[0], device=pat.device)]
    return canon, g


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
        r, k = design.stencil_radius, 2 * design.stencil_radius + 1
        support = torch.ones(2, k, k)
        support[0, -1, :] = 0.0     # A1 at dx = +r has no mirror partner (Hy offsets are dx + 1/2)
        support[1, :, -1] = 0.0     # A2 at dy = +r likewise
        self.register_buffer("support", support.view(2, k * k))

    def classes(self, eps):
        """Exact hashing of the network INPUT, canonicalised under D4 (rotations/reflections).
        Returns unique canonical patterns, per-cell canonical index, per-cell group element g, and the
        per-cell runtime class index over unique (pattern, g) pairs (uint8, int16 if > 255)."""
        b, h, w = eps.shape
        pat = patches(torch.log(eps), self.design.eps_radius).permute(0, 2, 3, 1).reshape(b * h * w, -1)
        canon, g = canonicalize(pat, self.design.eps_radius)
        uniq, inv = torch.unique(canon, dim=0, return_inverse=True)
        pair = inv * 8 + g
        _, cls = torch.unique(pair, return_inverse=True)
        n_cls = int(cls.max().item()) + 1
        dtype = torch.uint8 if n_cls <= 255 else torch.int16
        return uniq, inv.view(b, h, w), g.view(b, h, w), cls.view(b, h, w).to(dtype), n_cls

    def table(self, uniq):
        """Per-class stencils (n, 2, k*k) and M factors (n,)."""
        out = self.mlp(uniq)
        # bulk rule: a uniform eps patch (vacuum or inside one material) gets exactly the base stencil
        interface = ((uniq.max(dim=1).values - uniq.min(dim=1).values) > 0).float()[:, None]
        out = out * interface
        k2 = self.base.shape[1]
        da = out[:, :2 * k2].view(-1, 2, k2)
        if "C" not in self.design.outputs:
            da = da * 0
        da = torch.einsum("cij,ncj->nci", self.proj, da * self.support) * self.support
        factor = F.softplus(out[:, -1] + math.log(math.e - 1))
        if "M" not in self.design.outputs:
            factor = factor * 0 + 1
        return da, factor            # correction in the canonical frame; base added per cell

    def oriented_tables(self, uniq):
        """(8, n, 2, k*k) corrections for every group element, plus (n,) M factors."""
        da, f_tab = self.table(uniq)
        k = 2 * self.design.stencil_radius + 1
        da = da.view(-1, 2, k, k)
        # A canonical pattern that is itself symmetric (stabiliser h with h(P) = P) must get a stencil
        # with the same symmetry, or equivariance breaks: average over the stabiliser.
        stab = (pattern_transforms(uniq, self.design.eps_radius) == uniq[None]).all(-1).to(da.dtype)   # (8, n)
        sym = sum(stab[g][:, None, None, None] * transform_stencils_forward(da, g >> 2, (g >> 1) & 1, g & 1)
                  for g in range(8))
        da = sym / stab.sum(0)[:, None, None, None]
        tabs = torch.stack([transform_stencils(da, g >> 2, (g >> 1) & 1, g & 1).reshape(-1, 2, k * k) for g in range(8)])
        return tabs, f_tab

    def setup(self, eps):
        """Once per scene batch: per-cell stencils a (B, 2, k*k, h, w) and inv_m (B, h, w)."""
        uniq, inv, g, cls, n_cls = self.classes(eps)
        tabs, f_tab = self.oriented_tables(uniq)
        a = (self.base + tabs[g, inv]).permute(0, 3, 4, 1, 2)
        inv_m = 1.0 / (eps * f_tab[inv])
        return a, inv_m, n_cls, cls.element_size()


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


class IndexedMaterialStepper:
    """Runtime form used for timing: per-cell class index (uint8/int16) + weight table; the stencil
    is gathered from the table inside the compiled step instead of being stored per cell."""

    def __init__(self, net, eps):
        with torch.no_grad():
            uniq, inv, g, cls, n_cls = net.classes(eps)
            tabs, f_tab = net.oriented_tables(uniq)
            # runtime table: one row per (pattern, orientation) class actually present
            pair = (inv * 8 + g).flatten()
            first = torch.zeros(n_cls, dtype=torch.long, device=eps.device)
            first[cls.flatten().long()] = torch.arange(pair.numel(), device=eps.device)
            self.a_tab = (net.base + tabs[g.flatten()[first], inv.flatten()[first]]).contiguous()
            self.inv_m_tab = f_tab[inv.flatten()[first]].contiguous()
        self.r = net.design.stencil_radius
        self.idx = cls
        self.inv_eps = 1.0 / eps
        self.n_classes = n_cls

    def step(self, ez, hx, hy, inv_eps):
        i = self.idx.long()
        a = self.a_tab[i].permute(0, 3, 4, 1, 2)
        inv_m = self.inv_eps / self.inv_m_tab[i]
        return tied_step(ez, hx, hy, a, inv_m, self.r)
