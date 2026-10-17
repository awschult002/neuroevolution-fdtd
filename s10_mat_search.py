"""Stage 10 (D2): search over material networks. Usage: python s10_mat_search.py {random|evolution} BUDGET
random   : uniform without replacement over the 972-design space.
evolution: regularized evolution, population 12, tournament 4, one-knob mutation, oldest dies.
           Designs rejected by T-STAB-3 (or diverged) are logged and never enter the population.
Fitness (tournament only): feasible -> log(val interface err@1000) + 0.1*log(bytes/cell);
drift > 1% -> 100 + log(drift).  Appends to runs/mat_search_<method>.jsonl (restartable)."""
import json, math, os, random, sys, time
import torch
from fdtd.mateval import load_mat, train_and_score
from fdtd.matmodel import MatDesign, all_designs, mutate

POPULATION, TOURNAMENT = 12, 4


def fitness(rec):
    if rec["feasible"]:
        return math.log(rec["ierr1000"]) + 0.1 * math.log(rec["bytes_per_cell"])
    if rec.get("drift_mean") is None or rec["drift_mean"] != rec["drift_mean"] or rec["blew_up_at"] is not None:
        return 1e9
    return 100.0 + math.log(rec["drift_mean"])


def main():
    method, budget = sys.argv[1], int(sys.argv[2])
    path = f"runs/mat_search_{method}.jsonl"
    done = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    rng = random.Random({"random": 11, "evolution": 12}[method] + len(done))
    train = torch.load("data/mat_train.pt", weights_only=False)
    train["traj"] = train["traj"].pin_memory()
    val = load_mat("val")
    seen = {r["name"] for r in done}
    pool = [d for d in all_designs() if d.name() not in seen]
    while len(done) < budget:
        alive = [r for r in done if r["rejected"] is None]
        if method == "random" or len(alive) < POPULATION:
            design = pool.pop(rng.randrange(len(pool)))
        else:
            parent = min(rng.sample(alive[-POPULATION:], TOURNAMENT), key=fitness)
            for _ in range(20):
                design = mutate(MatDesign(**parent["design"]), rng)
                if design.name() not in seen:
                    break
            if design in pool:
                pool.remove(design)
        for attempt in range(60):
            try:
                rec, _ = train_and_score(design, train, val)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on {design.name()}, retrying in 60 s", flush=True)
                time.sleep(60)
        rec["index"] = len(done)
        done.append(rec)
        seen.add(design.name())
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{method} {len(done)}/{budget}] {design.name():32s} rej={rec['rejected']} feas={rec['feasible']} "
              f"ierr1000={rec.get('ierr1000', float('nan')):.4f} err1000={rec.get('err1000', float('nan')):.4f} "
              f"drift={rec.get('drift_mean', float('nan')):.1e} B/cell={rec['bytes_per_cell']:.1f} cls={rec['max_classes']} "
              f"{rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
