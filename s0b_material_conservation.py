"""Stage 0b (T-STAB-4 on materials + negative control). 20,000 coarse steps, staggered energy with
the model's own M_eps (U = 1/2(E^T M E + H^(n-1/2).H^(n+1/2))), max drift over 3 pulse ICs.
Weights: a MaterialNet (p=1, r=2, outputs CM) with a RANDOM last layer, i.e. random fixed per-class
stencils and M factors, set up once on the scene.
  correct_block  : true transpose (gather of neighbour weights), dielectric-block scene -> roundoff
  broken_block   : each cell uses its own stencil for the H step, block scene -> must drift
  broken_vacuum  : same broken variant in vacuum (all cells one class) -> roundoff
Also an exact adjoint test <C h, e> = <h, C^T e> for the per-cell operator.
Writes runs/conservation_materials.json."""
import json, torch
from fdtd.matmodel import MaterialNet, MatDesign
from fdtd.model import apply_transpose, broken_transpose, patches
from fdtd.mateval import base_kernel
from fdtd.yee import COURANT

STEPS, R = 20000, 2


def make(eps):
    torch.manual_seed(0)
    net = MaterialNet(MatDesign(1, 2, 16, "tanh", R, "CM"), base_kernel(R)).cuda()
    torch.nn.init.normal_(net.mlp[-1].weight, std=0.02)
    torch.nn.init.normal_(net.mlp[-1].bias, std=0.02)
    with torch.no_grad():
        a, inv_m, n_cls, _ = net.setup(eps)
    return a, inv_m, n_cls


@torch.no_grad()
def run(a, inv_m, transpose, state):
    ez, hx, hy = state[:, 0], state[:, 1], state[:, 2]
    m = 1.0 / inv_m
    u1 = None
    out = {}
    for n in range(1, STEPS + 1):
        h_prev = (hx, hy)
        ez = ez + COURANT * inv_m * ((a[:, 0] * patches(hy, R)).sum(1) + (a[:, 1] * patches(hx, R)).sum(1))
        hy = hy - COURANT * transpose(a[:, 0], ez, R)
        hx = hx - COURANT * transpose(a[:, 1], ez, R)
        if n == 1 or n in (5000, STEPS):
            u = 0.5 * (m * ez * ez + h_prev[0] * hx + h_prev[1] * hy).mean(dim=(-2, -1))
            if n == 1:
                u1 = u
            else:
                out[n] = ((u - u1).abs() / u1).max().item()
    return out


def main():
    torch.manual_seed(1)
    state = torch.zeros(3, 3, 64, 64, device="cuda")
    c = (torch.arange(64, device="cuda") + 0.5) / 64
    x, y = torch.meshgrid(c, c, indexing="ij")
    for b, (px, py) in enumerate(((0.3, 0.3), (0.5, 0.7), (0.75, 0.4))):
        state[b, 0] = torch.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * 0.06 ** 2))
    block = torch.ones(3, 64, 64, device="cuda")
    block[:, 20:44, 25:40] = 4.0
    block[:, 44, 25:40] = 2.5                           # a partially filled (subpixel) row
    vac = torch.ones(3, 64, 64, device="cuda")
    res = {}
    a, inv_m, n_cls = make(block)
    h, e = torch.randn(3, 64, 64, device="cuda", dtype=torch.float64), torch.randn(3, 64, 64, device="cuda", dtype=torch.float64)
    ad = a.double()
    lhs = ((ad[:, 0] * patches(h.float(), R).double()).sum(1) * e).sum()
    rhs = (h * apply_transpose(ad[:, 0], e, R)).sum()
    res["adjoint_rel_err"] = abs((lhs - rhs) / lhs).item()
    res["block_classes"] = n_cls
    res["correct_block"] = run(a, inv_m, apply_transpose, state)
    res["broken_block"] = run(a, inv_m, broken_transpose, state)
    a, inv_m, n_cls = make(vac)
    res["vacuum_classes"] = n_cls
    res["broken_vacuum"] = run(a, inv_m, broken_transpose, state)
    res["correct_vacuum"] = run(a, inv_m, apply_transpose, state)
    print(json.dumps(res, indent=1))
    json.dump(res, open("runs/conservation_materials.json", "w"), indent=1)


if __name__ == "__main__":
    main()
