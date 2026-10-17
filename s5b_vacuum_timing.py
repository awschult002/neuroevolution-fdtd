"""Stage 5b (D1): vacuum timing under T-TIME-2. Yee at 1024^2, 2048^2, 4096^2 and the fixed paired
stencils r1-r3 at 1024^2, all at Courant 0.5 (so 1024^2 uses 4x the 4096^2 dt), fp32, torch.compile.
Fixed stencils are timed in the fused roll-sum form (RollFixedStepper, one Triton kernel per half
step) and, for reference, the conv2d/conv_transpose2d training form. Reported per method: median s per simulated time unit over 5 valid runs. Writes runs/vacuum_timing.json."""
import json, os, platform, subprocess, sys
import torch
from fdtd.model import YeeStepper, RollFixedStepper, D4FixedStepper
from fdtd.timing import seconds_per_unit_time, warm_up, other_gpu_processes
from s9_mat_baselines import fixed


def main():
    if other_gpu_processes():
        sys.exit(f"GPU not idle: {other_gpu_processes()}")
    warm = warm_up(30.0)
    out = {"_env": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": platform.python_version(), "warm_clock_mhz": warm,
                    "gpu_driver": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                    "ac_online": open("/sys/class/power_supply/AC/online").read().strip(),
                    "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}}
    def d4(r):
        m = D4FixedStepper(r)
        m.load_state_dict(torch.load(f"runs/d4_r{r}.pt", map_location="cpu"))
        return RollFixedStepper(m)

    jobs = [(f"yee{n}", YeeStepper(), n) for n in (1024, 2048, 4096)]
    jobs += [(f"fixed_r{r}_fused", RollFixedStepper(fixed(r)), 1024) for r in (1, 2, 3)]
    jobs += [(f"d4_r{r}_fused", d4(r), 1024) for r in (1, 2, 3) if os.path.exists(f"runs/d4_r{r}.pt")]
    jobs += [(f"fixed_r{r}_conv", fixed(r), 1024) for r in (1, 2, 3)]
    for key, stepper, n in jobs:
        s, m, log = seconds_per_unit_time(stepper, n, warm)
        out[key] = {"s_per_unit": s, "timed_at": m, "n": n, "runs": log}
        print(key, n, s, sum(r["valid"] for r in log), "/", len(log), flush=True)
        json.dump(out, open("runs/vacuum_timing.json", "w"), indent=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
