"""Explicit cost estimator: FLOPs and bytes per coarse cell update, then a roofline time.

Counting rules (multiply-add = 2 FLOPs, activation = 1 FLOP per element):
  first conv  : 2 * 3 * k^2 * W  (+W bias)
  hidden 1x1  : (L-1) * (2 * W^2 + W)
  output 1x1  : 2 * W * n_out + n_out, n_out = 4k^2 (stencil) or 3 (direct)
  activations : L * W
  stencil mode: add Yee stencil (4k^2) + apply 4 stencils (2 * 4k^2) + 3 updates (2 each)
  direct mode : Yee (12) + 3 corrections (2 each)
Bytes assume an ideal fused kernel: read Ez,Hx,Hy, write Ez,Hx,Hy once (24 B, fp32), plus
inv_eps (4 B) for Yee; halos come from cache; weights are negligible.
Time per cell = max(FLOPs / PEAK_FLOPS, bytes / PEAK_BW) with Quadro T2000 nominal peaks.
"""
from .yee import COURANT

PEAK_FLOPS = 3.6e12
PEAK_BW = 128e9
YEE_FLOPS = 12
YEE_BYTES = 28


def flops_per_cell(design):
    k2 = (2 * design.radius + 1) ** 2
    w, layers = design.width, design.layers
    n_out = 4 * k2 if design.output == "stencil" else 3
    f = 2 * 3 * k2 * w + w
    f += (layers - 1) * (2 * w * w + w)
    f += 2 * w * n_out + n_out
    f += layers * w
    if design.output == "stencil":
        f += 4 * k2 + 2 * 4 * k2 + 6
    else:
        f += YEE_FLOPS + 6
    return f


def bytes_per_cell(design):
    return 24


def seconds_per_cell(flops, nbytes):
    return max(flops / PEAK_FLOPS, nbytes / PEAK_BW)


def seconds_per_unit_time(flops, nbytes, n):
    """Estimated wall-clock to advance an n^2 grid by one unit of simulated time."""
    steps = n / COURANT
    return seconds_per_cell(flops, nbytes) * n * n * steps
