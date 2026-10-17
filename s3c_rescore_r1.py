"""Stage 3c: re-score R1 random-search designs with the staggered energy. R1 weights were not kept,
so the 8 designs with the lowest R1 val err@1000 are retrained with the identical protocol
(seed 0, 500 steps) and re-scored. Writes runs/r1_rescored.json."""
import json, torch
from fdtd.evaluate import load_split, train_and_score
from fdtd.model import Design

old = [json.loads(l) for l in open("runs/v1_untied/search_random.jsonl")]
old = sorted([r for r in old if r["err1000"] == r["err1000"]], key=lambda r: r["err1000"])[:8]
traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
val = load_split("val")
out = []
for r in old:
    rec, _ = train_and_score(Design(**r["design"]), traj, val, 500)
    rec["r1_naive_drift"] = r["drift_mean"]
    out.append(rec)
    print(rec["name"], "err1000", round(rec["err1000"], 4), "staggered drift", rec["drift_mean"], "R1 naive drift", r["drift_mean"], flush=True)
json.dump(out, open("runs/r1_rescored.json", "w"), indent=1)
