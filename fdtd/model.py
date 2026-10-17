"""Search space and the coarse-grid steppers (plain Yee and learned).

Learned stepper, one coarse step, state = (Ez, Hx, Hy) on the 64^2 Yee grid:
  features = MLP applied per cell to the (2r+1)^2 neighbourhood of the state
             (first layer is a (2r+1)x(2r+1) conv, the rest are 1x1 convs).
  output "stencil": features give 4 local stencils of size (2r+1)^2, added to the
             Yee stencils:  Ez += C*inv_eps*(S0.Hy + S1.Hx);  Hx += C*S2.Ez;  Hy += C*S3.Ez
             (Bar-Sinai style: the net predicts the coefficients, the update stays linear in fields).
  output "tied": energy-structured variant of "stencil". Only the two curl stencils of the Ez
             update are learned (A1 acting on Hy, A2 acting on Hx); the H update uses their
             transposes:  Ez += C*inv_eps*(A1 Hy + A2 Hx);  Hy -= C*A1^T Ez;  Hx -= C*A2^T Ez.
             Plain Yee is exactly this with A1 = Dx^-, A2 = -Dy^-. With coefficients that are fixed in
             time this leapfrog conserves the discrete quantity
             1/2 sum(eps Ez^(n+1)^2 + H^(n+1/2) . H^(n+3/2)) exactly (up to roundoff); when the
             coefficients depend on the state that guarantee is lost (see s0_conservation_check.py).
  output "direct": plain Yee step plus C * (3 predicted corrections).
The last layer is zero-initialised, so every model starts as exact coarse Yee.
"""
import itertools
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yee import COURANT, yee_step

SPACE = {
    "radius": (1, 2, 3),
    "layers": (1, 2, 3),
    "width": (4, 8, 16, 32),
    "activation": ("tanh", "relu", "silu"),
    "output": ("tied", "stencil", "direct"),
}
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}


@dataclass(frozen=True)
class Design:
    radius: int
    layers: int
    width: int
    activation: str
    output: str

    def name(self):
        return f"r{self.radius}_l{self.layers}_w{self.width}_{self.activation}_{self.output}"


HAND_PICKED = Design(radius=2, layers=2, width=16, activation="relu", output="stencil")


def all_designs():
    return [Design(*values) for values in itertools.product(*SPACE.values())]


def random_design(rng):
    return Design(*(rng.choice(values) for values in SPACE.values()))


def mutate(design, rng):
    """Change exactly one knob to a different value."""
    knob = rng.choice(list(SPACE))
    choices = [v for v in SPACE[knob] if v != getattr(design, knob)]
    return replace(design, **{knob: rng.choice(choices)})


def patches(field, r):
    """(B, h, w) -> (B, k*k, h, w); entry (dx+r)*k + (dy+r) is field[i+dx, j+dy], periodic."""
    b, h, w = field.shape
    k = 2 * r + 1
    padded = F.pad(field[:, None], (r, r, r, r), mode="circular")
    return F.unfold(padded, k).view(b, k * k, h, w)


def apply_transpose(coeffs, field, r):
    """(A^T f)[j] = sum_k a_k(j-k) f[j-k], the adjoint of (A g)[i] = sum_k a_k(i) g[i+k].
    coeffs: (B, k*k, n, n) in the same offset order as patches().
    This is a GATHER: output cell j reads neighbour i = j-k's field AND neighbour i's own weight
    a_k(i) (torch.roll of the product a_k * f), never its own. No scatter-add / atomics, so it is
    deterministic. Checked as an exact adjoint (<Ah, f> = <h, A^T f>) in s0b_material_conservation.py."""
    k = 2 * r + 1
    out = torch.zeros_like(field)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            a = coeffs[:, (dx + r) * k + (dy + r)]
            out = out + torch.roll(a * field, shifts=(dx, dy), dims=(-2, -1))
    return out


def broken_transpose(coeffs, field, r):
    """NEGATIVE CONTROL ONLY: cell j applies its OWN stencil a_k(j) to f[j-k]. Equals the true
    transpose when the stencil is the same everywhere (vacuum) and is wrong where it varies."""
    k = 2 * r + 1
    out = torch.zeros_like(field)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out = out + coeffs[:, (dx + r) * k + (dy + r)] * torch.roll(field, shifts=(dx, dy), dims=(-2, -1))
    return out


def yee_stencils(r):
    k = 2 * r + 1
    s = torch.zeros(4, k, k)
    s[0, r + 0, r] = 1.0;  s[0, r - 1, r] = -1.0      # Hy -> Ez :  Hy[i] - Hy[i-1]
    s[1, r, r + 0] = -1.0; s[1, r, r - 1] = 1.0       # Hx -> Ez : -(Hx[j] - Hx[j-1])
    s[2, r, r + 1] = -1.0; s[2, r, r + 0] = 1.0       # Ez -> Hx : -(Ez[j+1] - Ez[j])
    s[3, r + 1, r] = 1.0;  s[3, r + 0, r] = -1.0      # Ez -> Hy :   Ez[i+1] - Ez[i]
    return s.view(4, k * k)


class YeeStepper(nn.Module):
    def step(self, ez, hx, hy, inv_eps):
        return yee_step(ez, hx, hy, inv_eps)


class LearnedStepper(nn.Module):
    def __init__(self, design):
        super().__init__()
        self.design = design
        r, k, w = design.radius, 2 * design.radius + 1, design.width
        act = ACTIVATIONS[design.activation]
        layers = [nn.Conv2d(3, w, k), act()]
        for _ in range(design.layers - 1):
            layers += [nn.Conv2d(w, w, 1), act()]
        n_out = {"stencil": 4 * k * k, "tied": 2 * k * k, "direct": 3}[design.output]
        last = nn.Conv2d(w, n_out, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)
        self.register_buffer("yee", yee_stencils(r)[None, :, :, None, None])

    def step(self, ez, hx, hy, inv_eps):
        r = self.design.radius
        state = F.pad(torch.stack([ez, hx, hy], dim=1), (r, r, r, r), mode="circular")
        out = self.net(state)
        if self.design.output == "direct":
            ez, hx, hy = yee_step(ez, hx, hy, inv_eps)
            return ez + COURANT * out[:, 0], hx + COURANT * out[:, 1], hy + COURANT * out[:, 2]
        b, _, n, _ = out.shape
        if self.design.output == "tied":
            return self.tied_update(ez, hx, hy, inv_eps, self.yee[:, :2] + out.view(b, 2, -1, n, n))
        s = self.yee + out.view(b, 4, -1, n, n)
        ez = ez + COURANT * inv_eps * ((s[:, 0] * patches(hy, r)).sum(1) + (s[:, 1] * patches(hx, r)).sum(1))
        pe = patches(ez, r)
        hx = hx + COURANT * (s[:, 2] * pe).sum(1)
        hy = hy + COURANT * (s[:, 3] * pe).sum(1)
        return ez, hx, hy


    def tied_update(self, ez, hx, hy, inv_eps, a):
        """a: (B, 2, k*k, n, n) stencils A1 (on Hy) and A2 (on Hx) of the Ez update."""
        r = self.design.radius
        ez = ez + COURANT * inv_eps * ((a[:, 0] * patches(hy, r)).sum(1) + (a[:, 1] * patches(hx, r)).sum(1))
        hy = hy - COURANT * apply_transpose(a[:, 0], ez, r)
        hx = hx - COURANT * apply_transpose(a[:, 1], ez, r)
        return ez, hx, hy


def periodic_fold(y, r):
    """Wrap the r-wide border of a full conv_transpose2d output back onto the periodic grid."""
    y = y.clone()
    y[..., r:2 * r, :] += y[..., -r:, :]
    y[..., -2 * r:-r, :] += y[..., :r, :]
    y = y[..., r:-r, :]
    y[..., :, r:2 * r] += y[..., :, -r:]
    y[..., :, -2 * r:-r] += y[..., :, :r]
    return y[..., :, r:-r]


def consistency_projector(r):
    """Projector onto stencil changes that keep the moment conditions of Yee's curl stencils:
    A1 (on Hy, at x-offset dx+1/2):  sum w = 0,  sum w (dx+1/2) = 1,  sum w dy = 0
    A2 (on Hx, at y-offset dy+1/2):  sum w = 0,  sum w (dy+1/2) = -1, sum w dx = 0
    i.e. sum_j w_j (j+1/2) = 1/dx (grid units). Yee satisfies them, so learned = Yee + P delta does too."""
    k = 2 * r + 1
    off = torch.arange(-r, r + 1, dtype=torch.float64)
    dx, dy = torch.meshgrid(off, off, indexing="ij")
    rows = []
    for normal, tangent in ((dx, dy), (dy, dx)):
        c = torch.stack([torch.ones(k * k, dtype=torch.float64), (normal + 0.5).flatten(), tangent.flatten()])
        rows.append(torch.eye(k * k, dtype=torch.float64) - c.T @ torch.linalg.solve(c @ c.T, c))
    return torch.stack(rows).float()                    # (2, k*k, k*k)


class FixedLinearStepper(nn.Module):
    """Coefficients fixed in time (the 'guaranteed' arm): two constant curl stencils A1, A2 of
    radius r, consistent (see consistency_projector). E step: conv2d of (Hy, Hx) with the kernel;
    H step: conv_transpose2d of Ez with the SAME kernel (the exact adjoint), so the staggered
    energy is conserved to roundoff. No penalty term."""

    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        k = 2 * radius + 1
        self.register_buffer("yee", yee_stencils(radius)[:2].clone())
        self.register_buffer("proj", consistency_projector(radius))
        self.delta = nn.Parameter(torch.zeros(2, k * k))

    def kernel(self):
        k = 2 * self.radius + 1
        w = self.yee + torch.einsum("cij,cj->ci", self.proj, self.delta)
        return w.view(1, 2, k, k)

    def step(self, ez, hx, hy, inv_eps):
        r, w = self.radius, self.kernel()
        h = F.pad(torch.stack([hy, hx], dim=1), (r, r, r, r), mode="circular")
        ez = ez + COURANT * inv_eps * F.conv2d(h, w)[:, 0]
        t = periodic_fold(F.conv_transpose2d(ez[:, None], w), r)
        return ez, hx - COURANT * t[:, 1], hy - COURANT * t[:, 0]

    def sigma_max(self, n=64):
        """max_k |C_hat(k)| of the curl operator on the periodic n^2 grid, by FFT (grid units)."""
        r, w = self.radius, self.kernel().detach()[0].double().cpu()
        total = torch.zeros(n, n, dtype=torch.float64)
        for c in range(2):
            full = torch.zeros(n, n, dtype=torch.float64)
            for a in range(2 * r + 1):
                for b in range(2 * r + 1):
                    full[(a - r) % n, (b - r) % n] = w[c, a, b]
            total += torch.fft.fft2(full).abs() ** 2
        return total.sqrt().max().item()

    def stable(self):
        """Leapfrog stability at the coarse (4x) dt: dt*sigma_max < 2 sqrt(eps mu), eps = mu = 1."""
        return COURANT * self.sigma_max() < 2.0

    def rescale_to_stable(self, margin=0.98):
        """Scale the kernel down so COURANT*sigma_max = margin*2. Returns the factor (1.0 if none)."""
        cn = COURANT * self.sigma_max()
        if cn < 2.0:
            return 1.0
        f = margin * 2.0 / cn
        with torch.no_grad():                            # w -> f*w, written back into delta
            w = self.kernel()[0].view(2, -1) * f
            self.delta.copy_(w - self.yee)              # note: scaling breaks the consistency moment
        return f


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


class D4FixedStepper(FixedLinearStepper):
    """D4-constrained fixed paired stencil. A1 (d/dx acting on Hy, which sits at x-offset dx+1/2) is
    odd about its half-cell axis, w(dx, dy) = -w(-1-dx, dy), and even along y, w(dx, dy) = w(dx, -dy);
    A2 (= -d/dy on Hx) is its 90-degree rotation, A2(dx, dy) = -A1(dy, dx). So each A1 entry is a
    paired difference p * (f[+] - f[-]). Free parameters: p(dx, dy) for dx = 0..r-1, dy = 0..r
    (r(r+1) of them), with the consistency moment sum w (dx+1/2) = 1 enforced by projection
    (sum w = 0 and sum w*dy = 0 hold by symmetry). Same conv2d / conv_transpose2d step as the parent."""

    def __init__(self, radius):
        super().__init__(radius)
        r, k = radius, 2 * radius + 1
        pairs = [(dx, dy) for dx in range(r) for dy in range(r + 1)]
        basis = torch.zeros(k * k, len(pairs))
        cons = torch.zeros(len(pairs))
        for j, (dx, dy) in enumerate(pairs):
            for sy in {dy, -dy}:
                basis[(dx + r) * k + (sy + r), j] += 1.0
                basis[(-1 - dx + r) * k + (sy + r), j] -= 1.0
            cons[j] = (2 * dx + 1) * (1 if dy == 0 else 2)
        p0 = torch.zeros(len(pairs))
        p0[0] = 1.0                                     # Yee: p(0, 0) = 1
        self.register_buffer("basis", basis)
        self.register_buffer("p0", p0)
        self.register_buffer("pproj", torch.eye(len(pairs)) - torch.outer(cons, cons) / cons.dot(cons))
        del self.delta
        self.dp = nn.Parameter(torch.zeros(len(pairs)))

    def kernel(self):
        k = 2 * self.radius + 1
        a1 = (self.basis @ (self.p0 + self.pproj @ self.dp)).view(k, k)
        a2 = -a1.T
        return torch.stack([a1, a2])[None]

    def rescale_to_stable(self, margin=0.98):
        if COURANT * self.sigma_max() < 2.0:
            return 1.0
        raise RuntimeError("D4 stencil violates T-STAB-3; rejected")


class RollFixedStepper(nn.Module):
    """Runtime form of a trained fixed paired stencil, for timing: the same operator written as a sum
    of shifted fields with constant weights (zero weights skipped), so torch.compile/Inductor fuses
    each half-step into one Triton kernel instead of calling cuDNN conv + conv_transpose.
    E step: Ez += C inv_eps sum w1[d] Hy[i+d] + w2[d] Hx[i+d];  H step: Hy[j] -= C sum w1[d] Ez[j-d]."""

    def __init__(self, fixed_stepper):
        super().__init__()
        r = fixed_stepper.radius
        w = fixed_stepper.kernel().detach().cpu()[0]
        self.terms = [[(dx, dy, float(w[c, dx + r, dy + r])) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                       if float(w[c, dx + r, dy + r]) != 0.0] for c in range(2)]

    def step(self, ez, hx, hy, inv_eps):
        acc = 0.0
        for c, field in ((0, hy), (1, hx)):
            for dx, dy, w in self.terms[c]:
                acc = acc + w * torch.roll(field, shifts=(-dx, -dy), dims=(-2, -1))
        ez = ez + COURANT * inv_eps * acc
        th, tx = 0.0, 0.0
        for dx, dy, w in self.terms[0]:
            th = th + w * torch.roll(ez, shifts=(dx, dy), dims=(-2, -1))
        for dx, dy, w in self.terms[1]:
            tx = tx + w * torch.roll(ez, shifts=(dx, dy), dims=(-2, -1))
        return ez, hx - COURANT * tx, hy - COURANT * th
