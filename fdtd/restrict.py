"""The single restriction operator (fine Yee grid -> coarse Yee grid) used for every method.

factor f = n_fine / n_coarse.
Space:
  Ez: mean of the f x f fine cell centres inside the coarse cell (cell average, centred).
  Hx: lives on y-faces. The coarse face y=(J+1)H coincides with fine face j = fJ+f-1;
      take the mean of the f fine Hx values along that face (face average).
  Hy: same with x and y swapped (fine face i = fI+f-1).
Time: coarse H must sit at t + T/2 (T = coarse dt = f * fine dt). Fine H after fine
substep k sits at t + (k+1/2) dt_f, so we average fine H after substeps f/2-1 and f/2.
That averaging is done by the caller (see run_fine_restricted); here we only do space.
"""
import torch


def restrict_ez(ez, f):
    b, n, _ = ez.shape
    m = n // f
    return ez.reshape(b, m, f, m, f).mean(dim=(2, 4))


def restrict_hx(hx, f):
    b, n, _ = hx.shape
    m = n // f
    return hx.reshape(b, m, f, m, f)[:, :, :, :, f - 1].mean(dim=2)


def restrict_hy(hy, f):
    b, n, _ = hy.shape
    m = n // f
    return hy.reshape(b, m, f, m, f)[:, :, f - 1, :, :].mean(dim=-1)


def restrict_state(ez, hx, hy, f):
    """Returns (B, 3, m, m) with channels Ez, Hx, Hy."""
    return torch.stack([restrict_ez(ez, f), restrict_hx(hx, f), restrict_hy(hy, f)], dim=1)
