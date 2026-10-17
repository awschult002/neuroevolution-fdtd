"""Stage 0 (Tester b / Oracle 6): unit check of energy conservation, 20,000 coarse steps on 64^2,
5 validation pulse ICs. Three runs:
  yee            plain Yee
  fixed_random   FixedLinearStepper(r=2), random constant consistent stencils (Yee + P*0.05*noise),
                 conv2d / conv_transpose2d with one kernel: expected drift at fp32 roundoff
  field_mlp      tied LearnedStepper(r=2, 2x16 tanh) with a random untrained MLP (coefficients depend
                 on the fields, recomputed every step): expected to drift
Drift = |U_N - U_1| / U_1 (max over ICs) for the staggered energy (the gate) and, for information,
the naive energy. Writes runs/conservation.json."""
import json, torch
from fdtd.model import Design, FixedLinearStepper, LearnedStepper, YeeStepper
from fdtd.yee import energy, naive_energy
from fdtd.evaluate import load_split

STEPS = 20000


def fixed_random(seed):
    torch.manual_seed(seed)
    m = FixedLinearStepper(2).cuda()
    with torch.no_grad():
        m.delta.normal_(std=0.05)
    return m


def field_mlp(seed):
    torch.manual_seed(seed)
    m = LearnedStepper(Design(2, 2, 16, "tanh", "tied")).cuda()
    torch.nn.init.normal_(m.net[-1].weight, std=0.05 / 4)
    return m


@torch.no_grad()
def run(stepper, state):
    ez, hx, hy = state[:, 0], state[:, 1], state[:, 2]
    ones = torch.ones_like(ez)
    step = torch.compile(stepper.step)
    out, u1, v1 = {}, None, None
    for m in range(1, STEPS + 1):
        h_prev = (hx, hy)
        ez, hx, hy = step(ez, hx, hy, 1.0)
        if m == 1 or m in (5000, STEPS):
            u = energy(ez, h_prev, (hx, hy), ones)
            v = naive_energy(ez, hx, hy, ones)
            if m == 1:
                u1, v1 = u, v
            else:
                out[m] = {"staggered": ((u - u1).abs() / u1).max().item(),
                          "naive": ((v - v1).abs() / v1).max().item(),
                          "finite": bool(torch.isfinite(u).all())}
    return out


def main():
    state = load_split("val")["yee512"][:5, 0]
    res = {"yee": run(YeeStepper(), state)}
    m = fixed_random(0)
    res["fixed_random"] = dict(run(m, state), courant_sigma=0.5 * m.sigma_max())
    res["field_mlp"] = run(field_mlp(0), state)
    for k, v in res.items():
        print(k, json.dumps(v), flush=True)
    json.dump(res, open("runs/conservation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
