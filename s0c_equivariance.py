"""Stage 0c: D4 equivariance of the canonicalised material net. Mirror (x), mirror (y) and transpose
the whole scene and state; one step of the mirrored problem must equal the mirrored step.
Also checks the indexed runtime stepper against the training-form stepper. Writes runs/equivariance.json."""
import json, torch
from fdtd.matmodel import MaterialNet, MatDesign, BoundMaterialStepper, IndexedMaterialStepper
from fdtd.model import yee_stencils


def mirror_x(ez, hx, hy):
    n = ez.shape[-2]
    i = (n - 1 - torch.arange(n)) % n
    j = (n - 2 - torch.arange(n)) % n
    return ez[:, i], hx[:, i], -hy[:, j]


def mirror_y(ez, hx, hy):
    n = ez.shape[-1]
    i = (n - 1 - torch.arange(n)) % n
    j = (n - 2 - torch.arange(n)) % n
    return ez[:, :, i], -hx[:, :, j], hy[:, :, i]


def transpose(ez, hx, hy):
    return ez.transpose(1, 2), -hy.transpose(1, 2), -hx.transpose(1, 2)


def main():
    torch.manual_seed(0)
    net = MaterialNet(MatDesign(2, 2, 16, "tanh", 2, "CM"), yee_stencils(2)[:2]).double().cuda()
    torch.nn.init.normal_(net.mlp[-1].weight, std=0.05)
    torch.nn.init.normal_(net.mlp[-1].bias, std=0.05)
    eps = torch.ones(1, 32, 32, dtype=torch.float64, device="cuda")
    eps[0, 5:17, 8:13] = 4.0
    eps[0, 17, 8:13] = 2.2
    eps[0, 20:24, 20:30] = 6.0
    state = [torch.randn(1, 32, 32, dtype=torch.float64, device="cuda") for _ in range(3)]
    res = {}
    with torch.no_grad():
        for name, op, eop in (("mirror_x", mirror_x, lambda e: e[:, torch.arange(31, -1, -1)]),
                              ("mirror_y", mirror_y, lambda e: e[:, :, torch.arange(31, -1, -1)]),
                              ("transpose", transpose, lambda e: e.transpose(1, 2))):
            a, inv_m, _, _ = net.setup(eps)
            out = BoundMaterialStepper(a, inv_m, 2).step(*state, None)
            a2, inv_m2, _, _ = net.setup(eop(eps))
            out2 = BoundMaterialStepper(a2, inv_m2, 2).step(*op(*state), None)
            res[name] = max((x - y).abs().max().item() for x, y in zip(op(*out), out2))
        a, inv_m, _, _ = net.setup(eps)
        ref = BoundMaterialStepper(a, inv_m, 2).step(*state, None)
        ind = IndexedMaterialStepper(net, eps).step(*state, None)
        res["indexed_vs_bound"] = max((x - y).abs().max().item() for x, y in zip(ref, ind))
    print(json.dumps(res, indent=1))
    json.dump(res, open("runs/equivariance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
