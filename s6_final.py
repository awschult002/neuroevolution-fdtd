"""Stage 6: freeze the search, pick the front, retrain front models with 3 seeds, stability gate,
held-out test (scored once). Usage: python s6_final.py [est|measured]
Front = non-dominated feasible designs (all methods + hand-picked) in (cost, val err1000);
cost = estimated (default) or measured (runs/timing.json) if the rank correlation came out < 0.8.
At most MAX_FRONT models (evenly spaced along the front). Writes runs/final.json."""
import json, sys, time
import torch
from fdtd.evaluate import load_split, score_stepper, train_and_score, TRAIN_STEPS
from fdtd.model import Design, LearnedStepper, YeeStepper
from fdtd.rollout import rollout
from fdtd.pareto import front

MAX_FRONT, SEEDS, STABILITY_STEPS, STABILITY_SAMPLES = 6, (0, 1, 2), 20000, 5


def load_candidates():
    recs = []
    for method in ("random", "evolution"):
        for line in open(f"runs/search_{method}.jsonl"):
            recs.append(dict(json.loads(line), method=method))
    hp = json.load(open("runs/baselines.json"))["hand_picked_val"]
    recs.append(dict(hp, method="hand"))
    return recs


def pick_front(recs, cost_key, timing):
    feas = {}
    for r in recs:
        if r["feasible"]:
            feas.setdefault(r["name"], r)
    names = list(feas)
    cost = (lambda r: timing[r["name"]]["s_per_unit"]) if cost_key == "measured" else (lambda r: r["est_s_per_unit_1024"])
    idx = front([(cost(feas[n]), feas[n]["err1000"]) for n in names])
    chosen = [names[i] for i in idx]
    if len(chosen) > MAX_FRONT:
        step = (len(chosen) - 1) / (MAX_FRONT - 1)
        chosen = [chosen[round(k * step)] for k in range(MAX_FRONT)]
    return [feas[n] for n in chosen]


def stability(stepper, val):
    sel = slice(0, STABILITY_SAMPLES)
    r = rollout(stepper, val["yee512"][sel, 0], val["eps"][sel], STABILITY_STEPS, energy_every=100)
    energies = torch.stack(r["energies"]).mean(dim=1).tolist()
    return {"blew_up_at": r["blew_up_at"], "drift_mean": r["drift_mean"], "drift_max": r["drift_max"],
            "energy_curve": energies}


def main():
    cost_key = sys.argv[1] if len(sys.argv) > 1 else "est"
    timing = json.load(open("runs/timing.json"))
    recs = load_candidates()
    chosen = pick_front(recs, cost_key, timing)
    print("front:", [r["name"] for r in chosen], flush=True)
    traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
    val, test = load_split("val"), load_split("test")
    out = {"cost_key": cost_key, "front": [], "gpu_s": 0.0}
    t_start = time.time()
    out["yee64_stability"] = stability(YeeStepper(), val)
    for r in chosen:
        design = Design(**r["design"])
        entry = {"name": design.name(), "method": r["method"], "params": r["params"], "flops": r["flops"],
                 "est_s_per_unit_1024": r["est_s_per_unit_1024"], "seeds": []}
        for seed in SEEDS:
            rec, model = train_and_score(design, traj, val, TRAIN_STEPS, seed=seed)
            model.eval()
            rec["stability"] = stability(model, val)
            rec["test"] = score_stepper(model, test)                                   # scored once
            rec["test_random"] = score_stepper(model, test, select=slice(0, 25))
            rec["test_hard"] = score_stepper(model, test, select=slice(25, 29))
            rec["test_dielectric"] = score_stepper(model, test, select=slice(29, 30))
            entry["seeds"].append(rec)
            print(design.name(), seed, {k: rec[k] for k in ("err100", "err1000", "err5000", "drift_mean")},
                  "stab", rec["stability"]["blew_up_at"], f'{rec["stability"]["drift_mean"]:.2e}', flush=True)
        out["front"].append(entry)
        out["gpu_s"] = time.time() - t_start
        json.dump(out, open("runs/final.json", "w"), indent=1)
    test_base = {"yee64": score_stepper(YeeStepper(), test, select=slice(0, 25)),
                 "yee64_hard": score_stepper(YeeStepper(), test, select=slice(25, 29)),
                 "yee64_dielectric": score_stepper(YeeStepper(), test, select=slice(29, 30))}
    out["test_baselines"] = test_base
    out["gpu_s"] = time.time() - t_start
    json.dump(out, open("runs/final.json", "w"), indent=1)


if __name__ == "__main__":
    main()
