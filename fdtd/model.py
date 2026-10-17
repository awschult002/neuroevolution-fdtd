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
    """(B, n, n) -> (B, k*k, n, n); entry (dx+r)*k + (dy+r) is field[i+dx, j+dy], periodic."""
    b, n, _ = field.shape
    k = 2 * r + 1
    padded = F.pad(field[:, None], (r, r, r, r), mode="circular")
    return F.unfold(padded, k).view(b, k * k, n, n)


def apply_transpose(coeffs, field, r):
    """(A^T f)[j] = sum_k a_k(j-k) f[j-k], the adjoint of (A g)[i] = sum_k a_k(i) g[i+k].
    coeffs: (B, k*k, n, n) in the same offset order as patches()."""
    k = 2 * r + 1
    out = torch.zeros_like(field)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            a = coeffs[:, (dx + r) * k + (dy + r)]
            out = out + torch.roll(a * field, shifts=(dx, dy), dims=(-2, -1))
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


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
