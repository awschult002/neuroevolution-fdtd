"""Stage 1: Yee sanity. Convergence of a pulse vs 1024^2 reference, energy drift, restriction check."""
import random, torch
from fdtd.initial import random_pulses, sample
from fdtd.fine import run_fine_restricted
from fdtd.rollout import rollout, rel_l2
from fdtd.model import YeeStepper

dev = "cuda"
ic = random_pulses(random.Random(1))
T = 64  # coarse (64^2) steps = 1.0 unit time
ref = run_fine_restricted(sample(ic, 1024, dev)[0][None], 1.0, 16, [0, T])
for n in (64, 128, 256, 512):
    st = run_fine_restricted(sample(ic, n, dev)[0][None], 1.0, n // 64, [0, T])
    print(f"Yee {n:4d}^2 -> 64^2 restricted, rel L2 vs 1024^2 at t=1: {rel_l2(st[:, 1], ref[:, 1]):.3e}")
start = ref[:, 0]
eps = torch.ones(1, 64, 64, device=dev)
r = rollout(YeeStepper(), start, eps, 5000)
print(f"coarse Yee 64^2, 5000 steps: drift {r['drift_mean']:.2e}, blew_up={r['blew_up_at']}")
r = rollout(YeeStepper(), start, eps, 64, record_steps=[64])
print(f"coarse Yee 64^2 vs 1024^2 truth at t=1: {rel_l2(r['records'][64], ref[:, 1]):.3e}")
