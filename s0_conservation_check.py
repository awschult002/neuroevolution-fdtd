"""Stage 0 (Tester condition b): does the tied update conserve energy?
20,000 coarse steps on 64^2, 5 pulse ICs from the validation set.
  1. plain Yee
  2. tied stencil, radius 2, random FIXED spatially varying coefficients (Yee + 0.05*noise)
  3. tied stencil whose coefficients come from a random untrained MLP (state-dependent,
     recomputed once per step; last layer scaled so the perturbation is ~0.05)
Reports, at 5k and 20k steps, relative drift |U-U0|/U0 of
  - U_shared: the one shared energy formula (fdtd/yee.py), used by the 1% gate
  - U_mod   : 1/2 sum(eps Ez^(n+1)^2 + H^(n+1/2).H^(n+3/2)), the quantity the tied leapfrog conserves
              (proof: with E += C A H, H -= C A^T E, the cross terms cancel step to step)
Writes runs/conservation.json."""
import json, torch
from fdtd.model import Design, LearnedStepper, YeeStepper
from fdtd.yee import energy
from fdtd.evaluate import load_split

STEPS, R = 20000, 2


class FixedTied:
    def __init__(self, n, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.model = LearnedStepper(Design(R, 1, 4, "tanh", "tied")).cuda()
        k2 = (2 * R + 1) ** 2
        self.a = self.model.yee[:, :2] + 0.05 * torch.randn(1, 2, k2, n, n, device="cuda", generator=g)

    def step(self, ez, hx, hy, inv_eps):
        return self.model.tied_update(ez, hx, hy, inv_eps, self.a.expand(ez.shape[0], -1, -1, -1, -1))


def mlp_tied(seed):
    torch.manual_seed(seed)
    m = LearnedStepper(Design(R, 2, 16, "tanh", "tied")).cuda()
    last = m.net[-1]
    torch.nn.init.normal_(last.weight, std=0.05 / 4)
    torch.nn.init.zeros_(last.bias)
    return m


@torch.no_grad()
def run(stepper, state):
    ez, hx, hy = state[:, 0], state[:, 1], state[:, 2]
    ones = torch.ones_like(ez)
    step = torch.compile(stepper.step)
    h_prev = None
    out = {}
    u_shared0 = energy(ez, hx, hy, ones)
    u_mod0 = None
    for n in range(STEPS + 1):
        ez_n, hx_n, hy_n = ez, hx, hy
        ez, hx, hy = step(ez, hx, hy, 1.0)
        # state at n holds Ez^n, H^(n+1/2); after the step Ez^(n+1), H^(n+3/2)
        u_mod = 0.5 * (ez * ez + hx_n * hx + hy_n * hy).mean(dim=(-2, -1))
        if n == 0:
            u_mod0 = u_mod
        if n in (5000, STEPS):
            u_sh = energy(ez_n, hx_n, hy_n, ones)
            out[n] = {"shared": ((u_sh - u_shared0).abs() / u_shared0).max().item(),
                      "mod": ((u_mod - u_mod0).abs() / u_mod0).max().item(),
                      "finite": bool(torch.isfinite(u_sh).all())}
    return out


def main():
    val = load_split("val")
    state = val["yee512"][:5, 0]
    res = {"yee": run(YeeStepper(), state),
           "tied_fixed_random": run(FixedTied(64, 0), state),
           "tied_mlp_random": run(mlp_tied(0), state)}
    for k, v in res.items():
        print(k, json.dumps(v))
    json.dump(res, open("runs/conservation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
