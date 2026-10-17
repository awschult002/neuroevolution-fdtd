"""Stage 11 (D2): freeze search; front = non-dominated feasible designs in (bytes/cell, val interface
err@1000), at most 5 evenly spaced. Each front design retrained with seeds 0,1,2; per seed: val score,
20k-step stability (5 val scenes, staggered energy with true eps), held-out test scored once (random 25,
unseen shapes 10), dispersion (vacuum) and Fresnel slab test. Baselines (Yee 64, fixed r1-3) get the
same stability / dispersion / Fresnel runs. Writes runs/mat_final.json."""
import json, time
import torch
from fdtd.mateval import load_mat, score, train_and_score, bind
from fdtd.matmodel import MatDesign
from fdtd.model import YeeStepper
from fdtd.pareto import front
from fdtd.rollout import rollout
from fdtd.dispersion import dispersion_table
from fdtd.fresnel import fresnel_table
from s9_mat_baselines import fixed

MAX_FRONT, SEEDS = 5, (0, 1, 2)


def stability(stepper, val):
    r = rollout(stepper, val["yee512"][:5, 0], val["eps"][:5], 20000, energy_every=100)
    return {"blew_up_at": r["blew_up_at"], "drift_mean": r["drift_mean"], "drift_max": r["drift_max"],
            "energy_curve": torch.stack(r["energies"]).mean(dim=1).tolist()}


def extras(stepper_for_eps, val):
    vac = torch.ones(1, 64, 64, device="cuda")
    return {"stability": stability(stepper_for_eps(val["eps"][:5]), val),
            "dispersion": dispersion_table(stepper_for_eps(vac)),
            "fresnel": fresnel_table(stepper_for_eps)}


def main():
    t0 = time.time()
    recs = []
    for m in ("random", "evolution"):
        recs += [dict(json.loads(l), method=m) for l in open(f"runs/mat_search_{m}.jsonl")]
    feas = {}
    for r in recs:
        if r["feasible"]:
            feas.setdefault(r["name"], r)
    names = list(feas)
    idx = front([(feas[n]["bytes_per_cell"], feas[n]["ierr1000"]) for n in names])
    chosen = [names[i] for i in idx]
    if len(chosen) > MAX_FRONT:
        step = (len(chosen) - 1) / (MAX_FRONT - 1)
        chosen = [chosen[round(k * step)] for k in range(MAX_FRONT)]
    print("front", chosen, flush=True)
    train = torch.load("data/mat_train.pt", weights_only=False)
    train["traj"] = train["traj"].pin_memory()
    val, test = load_mat("val"), load_mat("test")
    out = {"front": [], "baselines": {}}
    out["baselines"]["yee64"] = extras(lambda eps: YeeStepper(), val)
    for r in (1, 2, 3):
        out["baselines"][f"fixed_r{r}"] = extras(lambda eps, r=r: fixed(r), val)
    for name in chosen:
        design = MatDesign(**feas[name]["design"])
        entry = {"name": name, "method": feas[name]["method"], "seeds": []}
        for seed in SEEDS:
            rec, net = train_and_score(design, train, val, seed=seed)
            if rec["rejected"]:
                entry["seeds"].append(rec)
                continue
            rec.update(extras(lambda eps: bind(net, eps), val))
            rec["test_random"] = score(bind(net, test["eps"][:25]), test, slice(0, 25))
            rec["test_unseen"] = score(bind(net, test["eps"][25:]), test, slice(25, 35))
            entry["seeds"].append(rec)
            print(name, seed, {k: rec.get(k) for k in ("err1000", "ierr1000", "drift_mean")},
                  "stab", rec["stability"]["blew_up_at"], flush=True)
            torch.save(net.state_dict(), f"runs/mat_{name}_s{seed}.pt")
        out["front"].append(entry)
        out["gpu_s"] = time.time() - t0
        json.dump(out, open("runs/mat_final.json", "w"), indent=1)
    out["gpu_s"] = time.time() - t0
    json.dump(out, open("runs/mat_final.json", "w"), indent=1)


if __name__ == "__main__":
    main()
