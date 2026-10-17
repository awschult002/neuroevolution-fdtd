"""Stage 12 (D2): timing on an actual material scene under T-TIME-1..3 (idle GPU, 30 s warm-up,
nvidia-smi clock/temp per run, runs with >10% clock drop or another GPU process discarded, median of 5,
fp32). One validation scene sampled at 4096^2; coarse methods run at 1024^2 with 4x4-averaged eps,
Yee at 1024^2, 2048^2 (eps 2x2-averaged) and 4096^2. All kernels are torch.compile'd shifted sums
(Yee, the frozen-bar stencil via RollFixedStepper) or, for networks, the class-table runtime form.
Bytes/cell/step (ideal, each array touched once): fields 24 + inv_eps 4 = 28 for Yee and the bar;
networks 24 + inv_eps 4 + class index (1 or 2) + weight table amortised over the grid.
Achieved GB/s = bytes/cell/step * cells * steps / time; % of the T2000's ~128 GB/s.
Writes runs/mat_timing.json."""
import json, os, platform, subprocess, sys
import torch
from fdtd.scenes import sample_scene, average
from fdtd.model import YeeStepper, RollFixedStepper
from fdtd.matmodel import MatDesign, MaterialNet, IndexedMaterialStepper
from fdtd.mateval import base_kernel
from fdtd.timing import seconds_per_unit_time, warm_up, other_gpu_processes
from fdtd.yee import COURANT
from s9_mat_baselines import candidate

PEAK_GBS = 128.0


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
    warm = warm_up(30.0)
    out = {"_env": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": platform.python_version(), "warm_clock_mhz": warm, "scene": "val[0]",
                    "gpu_driver": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                    "ac_online": open("/sys/class/power_supply/AC/online").read().strip(),
                    "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}}

    def run(key, stepper, n, nbytes, extra=None):
        s, m, log = seconds_per_unit_time(stepper, n, warm, fallback_n=n // 2)
        gbs = nbytes * n * n * (n / COURANT) / s / 1e9
        out[key] = {"s_per_unit": s, "timed_at": m, "n": n, "bytes_per_cell": nbytes, "gb_s": gbs,
                    "pct_peak": 100 * gbs / PEAK_GBS, "runs": log, **(extra or {})}
        print(key, n, round(s, 4), "s/unit", round(gbs, 1), "GB/s", flush=True)
        json.dump(out, open("runs/mat_timing.json", "w"), indent=1)
        torch.cuda.empty_cache()

    for n in (1024, 2048, 4096):
        run(f"yee{n // 16}", WithEps(YeeStepper(), 1.0 / eps[n]), n, 28)
    bar = json.load(open("runs/mat_bar.json"))["bar"]
    run(f"bar_{bar}", WithEps(RollFixedStepper(candidate(bar)), 1.0 / eps[1024]), 1024, 28)
    if os.path.exists("runs/mat_final.json"):
        for e in json.load(open("runs/mat_final.json"))["front"]:
            d = MatDesign(**e["seeds"][0]["design"])
            net = MaterialNet(d, base_kernel(d.stencil_radius)).cuda()
            net.load_state_dict(torch.load(f"runs/mat_{e['name']}_s0.pt"))
            st = IndexedMaterialStepper(net.eval(), eps[1024])
            k2 = (2 * d.stencil_radius + 1) ** 2
            nbytes = 28 + st.idx.element_size() + st.n_classes * (2 * k2 + 1) * 4 / 1024 ** 2
            run(e["name"], st, 1024, nbytes, {"classes_1024": st.n_classes})


if __name__ == "__main__":
    main()
