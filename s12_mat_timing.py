"""Stage 12 (D2): timing on an idle GPU (TEST_SPEC T-TIME-1..3). One validation scene sampled at
4096^2; coarse methods run at 1024^2 with 4x4-averaged eps, Yee also at 2048^2 and 4096^2.
Learned models use the runtime form (class index + table, IndexedMaterialStepper).
Writes runs/mat_timing.json."""
import json, platform, subprocess, sys
import torch
from fdtd.scenes import sample_scene, average
from fdtd.model import YeeStepper
from fdtd.matmodel import MatDesign, MaterialNet, IndexedMaterialStepper
from fdtd.mateval import base_kernel
from fdtd.timing import seconds_per_unit_time, warm_up, other_gpu_processes
from s9_mat_baselines import fixed


class WithEps:
    def __init__(self, stepper, inv_eps):
        self.stepper, self.inv_eps = stepper, inv_eps

    def step(self, ez, hx, hy, _):
        return self.stepper.step(ez, hx, hy, self.inv_eps)


def main():
    if other_gpu_processes():
        sys.exit(f"GPU not idle: {other_gpu_processes()}")
    scene = torch.load("data/mat_val.pt", weights_only=False)["scenes"][0]
    eps4096 = sample_scene(scene, 4096, "cuda")[1][None]
    eps = {n: average(eps4096, 4096 // n) if n < 4096 else eps4096 for n in (1024, 2048, 4096)}
    del eps4096
    warm = warm_up(30.0)
    out = {"_env": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": platform.python_version(), "warm_clock_mhz": warm,
                    "gpu_driver": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                    "ac_online": open("/sys/class/power_supply/AC/online").read().strip(),
                    "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}}

    def run(key, stepper, n):
        s, m, log = seconds_per_unit_time(stepper, n, warm, fallback_n=n // 2)
        out[key] = {"s_per_unit": s, "timed_at": m, "n": n, "runs": log}
        print(key, s, m, flush=True)
        json.dump(out, open("runs/mat_timing.json", "w"), indent=1)
        torch.cuda.empty_cache()

    for n in (1024, 2048, 4096):
        run(f"yee{n // 16}", WithEps(YeeStepper(), 1.0 / eps[n]), n)
    for r in (1, 2, 3):
        run(f"fixed_r{r}", WithEps(fixed(r), 1.0 / eps[1024]), 1024)
    final = json.load(open("runs/mat_final.json"))
    for e in final["front"]:
        d = MatDesign(**e["seeds"][0]["design"])
        net = MaterialNet(d, base_kernel(d.stencil_radius)).cuda()
        net.load_state_dict(torch.load(f"runs/mat_{e['name']}_s0.pt"))
        st = IndexedMaterialStepper(net.eval(), eps[1024])
        run(e["name"], st, 1024)
        out[e["name"]]["classes_1024"] = st.n_classes


if __name__ == "__main__":
    main()
