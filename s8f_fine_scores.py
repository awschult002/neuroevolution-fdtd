"""Validation errors of the fine-Yee baselines vs the 2048^2 truth (edge band / bulk). Writes runs/mat_fine_val.json."""
import json
from fdtd.mateval import load_mat, rel_l2
d = load_mat("val")
out = {}
for n in (1024, 512, 256, 128):
    f = d[f"fine{n}"].cuda()
    for k, s in enumerate(d["steps"]):
        if s:
            t, m = d["yee512"][:, k], d["mask"]
            out[f"yee{n}_step{s}"] = {"err": rel_l2(f[:, k], t), "edge": rel_l2(f[:, k], t, m), "bulk": rel_l2(f[:, k], t, ~m)}
print(json.dumps(out, indent=1))
json.dump(out, open("runs/mat_fine_val.json", "w"), indent=1)
