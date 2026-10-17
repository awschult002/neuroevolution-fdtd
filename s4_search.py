"""Stage 4: architecture search. Usage: python s4_search.py {random|evolution} BUDGET [TRAIN_STEPS]
Appends one JSON line per candidate to runs/search_<method>.jsonl (restartable).

random   : designs sampled uniformly without replacement.
evolution: regularized (aging) evolution, population 12, tournament 4; the child is a one-knob
           mutation of the tournament winner; the oldest member dies. Same budget as random.
Scalar fitness (lower is better) for the tournament only:
    log(err1000) + 0.25 * log(estimated cost);
    infeasible but finite (drift > 1%) = 100 + log(drift), so selection still pushes toward
    lower drift while nothing is feasible; blown up = +inf.
The Pareto fronts themselves are computed later from (err, cost) without this scalarisation.
"""
import json, math, os, random, sys, time
import torch
from fdtd.evaluate import load_split, train_and_score, TRAIN_STEPS
from fdtd.model import Design, all_designs, mutate

POPULATION, TOURNAMENT, COST_WEIGHT = 12, 4, 0.25


def fitness(rec):
    if not rec["feasible"]:
        if rec["blew_up_at"] is not None or rec["drift_mean"] != rec["drift_mean"]:
            return math.inf
        return 100.0 + math.log(rec["drift_mean"])
    return math.log(rec["err1000"]) + COST_WEIGHT * math.log(rec["est_s_per_unit_1024"])


def main():
    method, budget = sys.argv[1], int(sys.argv[2])
    train_steps = int(sys.argv[3]) if len(sys.argv) > 3 else TRAIN_STEPS
    path = f"runs/search_{method}.jsonl"
    done = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    rng = random.Random({"random": 1, "evolution": 2}[method])
    # (a restart skips already-evaluated designs but does not replay the exact random sequence)
    traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
    val = load_split("val")
    seen = {r["name"] for r in done}
    pool = [d for d in all_designs() if d.name() not in seen]
    while len(done) < budget:
        if method == "random" or len(done) < POPULATION:
            design = pool.pop(rng.randrange(len(pool)))
        else:
            population = done[-POPULATION:]
            parent = min(rng.sample(population, TOURNAMENT), key=fitness)
            design = mutate(Design(**parent["design"]), rng)
            for _ in range(20):
                if design.name() not in seen:
                    break
                design = mutate(Design(**parent["design"]), rng)
            if design in pool:
                pool.remove(design)
        t0 = time.time()
        for attempt in range(60):       # the GPU is shared with other jobs: wait out OOMs
            try:
                rec, _ = train_and_score(design, traj, val, train_steps)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on {design.name()}, retrying in 60 s", flush=True)
                time.sleep(60)
        else:
            raise RuntimeError("GPU memory never became available")
        rec["index"] = len(done)
        rec["wall_s"] = time.time() - t0
        done.append(rec)
        seen.add(design.name())
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{method} {len(done)}/{budget}] {design.name():28s} feasible={rec['feasible']} "
              f"err1000={rec['err1000']:.4f} drift={rec['drift_mean']:.2e} {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
