"""Stage 5: measured cost. Times every feasible searched design (+ hand-picked) at 1024^2 coarse,
and Yee at 1024^2 / 2048^2 / 4096^2 (the same physical problem as 64^2 / 128^2 / 256^2 scaled 16x).
Writes runs/timing.json  {name: {"s_per_unit": .., "timed_at": n}}."""
import json, os, torch
from fdtd.model import Design, LearnedStepper, YeeStepper, HAND_PICKED
from fdtd.timing import seconds_per_unit_time


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
    for n in (1024, 2048, 4096):
        key = f"yee{n // 16}"
        if key not in out:
            s, m = seconds_per_unit_time(YeeStepper(), n)
            out[key] = {"s_per_unit": s, "timed_at": m, "n": n}
            print(key, out[key], flush=True)
    for name, design in searched_designs().items():
        if name in out:
            continue
        s, m = seconds_per_unit_time(LearnedStepper(design).cuda().eval(), 1024, fallback_n=512)
        out[name] = {"s_per_unit": s, "timed_at": m, "n": 1024}
        print(name, out[name], flush=True)
        json.dump(out, open(path, "w"), indent=1)
        torch.cuda.empty_cache()
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
