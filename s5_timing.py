"""Stage 5: measured cost. Times every feasible searched design (+ hand-picked) at 1024^2 coarse,
and Yee at 1024^2 / 2048^2 / 4096^2 (the same physical problem as 64^2 / 128^2 / 256^2 scaled 16x).
Writes runs/timing.json  {name: {"s_per_unit": .., "timed_at": n}}."""
import json, os, platform, subprocess, torch
from fdtd.model import Design, LearnedStepper, YeeStepper, HAND_PICKED
from fdtd.timing import seconds_per_unit_time, warm_up


def searched_designs():
    names = {}
    for method in ("random", "evolution"):
        for line in open(f"runs/search_{method}.jsonl"):
            r = json.loads(line)
            if r["feasible"]:
                names[r["name"]] = Design(**r["design"])
    names[HAND_PICKED.name()] = HAND_PICKED
    return names


def main():
    path = "runs/timing.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    warm_clock = warm_up(30.0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    out["_env"] = {"machine": platform.node(), "gpu_driver": smi, "cuda": torch.version.cuda,
                   "torch": torch.__version__, "python": platform.python_version(),
                   "ac_online": open("/sys/class/power_supply/AC/online").read().strip(),
                   "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                   "warm_clock_mhz": warm_clock}
    for n in (1024, 2048, 4096):
        key = f"yee{n // 16}"
        if key not in out:
            s, m, log = seconds_per_unit_time(YeeStepper(), n, warm_clock)
            out[key] = {"s_per_unit": s, "timed_at": m, "n": n, "runs": log}
            print(key, s, m, flush=True)
    for name, design in searched_designs().items():
        if name in out:
            continue
        s, m, log = seconds_per_unit_time(LearnedStepper(design).cuda().eval(), 1024, warm_clock, fallback_n=512)
        out[name] = {"s_per_unit": s, "timed_at": m, "n": 1024, "runs": log}
        print(name, s, m, flush=True)
        json.dump(out, open(path, "w"), indent=1)
        torch.cuda.empty_cache()
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
