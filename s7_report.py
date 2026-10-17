"""Stage 7: plots + REPORT.md tables + results/summary.json, from runs/*.json(l) and data/*.pt.
The verdict paragraph at the top of REPORT.md is written by hand after reading these numbers."""
import json, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from fdtd.pareto import front
from fdtd.rollout import rel_l2

base = json.load(open("runs/baselines.json"))
timing = json.load(open("runs/timing.json"))
final = json.load(open("runs/final.json"))
search = {m: [json.loads(l) for l in open(f"runs/search_{m}.jsonl")] for m in ("random", "evolution")}
hand = base["hand_picked_val"]


def med(xs):
    return statistics.median(xs)


def spread(xs):
    return f"{med(xs):.4f} [{min(xs):.4f}-{max(xs):.4f}]"


# ---------- Pareto plot: val err1000 vs measured s per simulated unit time (equal physical domain)
fig, ax = plt.subplots(figsize=(7, 5))
styles = {"random": ("tab:blue", "o"), "evolution": ("tab:orange", "^")}
for m, recs in search.items():
    pts = [(timing[r["name"]]["s_per_unit"], r["err1000"]) for r in recs if r["feasible"] and r["name"] in timing]
    c, mk = styles[m]
    ax.scatter(*zip(*pts), c=c, marker=mk, alpha=0.4, label=f"{m} ({len(pts)} feasible / {len(recs)})")
    f = sorted(pts[i] for i in front(pts))
    ax.step(*zip(*f), where="post", c=c, lw=2)
ax.scatter(timing[hand["name"]]["s_per_unit"], hand["err1000"], c="tab:green", marker="*", s=200,
           label="hand-picked (5x5 stencil MLP)", zorder=5)
yee = [(timing[f"yee{n}"]["s_per_unit"], base[f"yee{n}_val"]["err1000"]) for n in (64, 128, 256)]
ax.plot(*zip(*yee), "k-s", label="Yee 64/128/256 (timed 1024/2048/4096)")
for (x, y), n in zip(yee, (64, 128, 256)):
    ax.annotate(f"Yee {n}", (x, y), textcoords="offset points", xytext=(5, 5))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("measured wall-clock per simulated time unit [s] (T2000, 16x-scaled grids)")
ax.set_ylabel("validation rel. L2 error at 1000 coarse steps")
ax.set_title("Error vs measured cost (val set, single seed from search)")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("results/pareto.png", dpi=110)

# ---------- stability plot: energy vs step, 20k coarse steps
fig, ax = plt.subplots(figsize=(7, 4))
e = final["yee64_stability"]["energy_curve"]
ax.plot([100 * i for i in range(len(e))], [v / e[0] - 1 for v in e], "k", lw=2, label="Yee 64")
for entry in final["front"]:
    s = entry["seeds"][0]["stability"]["energy_curve"]
    ax.plot([100 * i for i in range(len(s))], [v / s[0] - 1 for v in s], label=entry["name"])
ax.axhline(0.01, c="r", ls="--", lw=0.8); ax.axhline(-0.01, c="r", ls="--", lw=0.8)
ax.set_xlabel("coarse step"); ax.set_ylabel("U(t)/U(0) - 1 (mean of 5 val samples)")
ax.set_title("Stability gate, 20,000 coarse steps (seed 0)"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/stability.png", dpi=110)

# ---------- tables
lines = []
lines.append("## Baselines (25 validation / 25 held-out random test ICs)\n")
lines.append("| method | split | err@100 | err@1000 | err@5000 | drift@5000 (mean) | s / unit time |")
lines.append("|---|---|---|---|---|---|---|")
for split in ("val", "test"):
    for n in (64, 128, 256):
        b = base[f"yee{n}_{split}"]
        drift = f"{b['drift_mean']:.2e}" if "drift_mean" in b else "-"
        lines.append(f"| Yee {n}^2 | {split} | {b['err100']:.4f} | {b['err1000']:.4f} | {b['err5000']:.4f} | {drift} | {timing[f'yee{n}']['s_per_unit']:.3f} |")
lines.append(f"| hand-picked | val | {hand['err100']:.4f} | {hand['err1000']:.4f} | {hand['err5000']:.4f} | {hand['drift_mean']:.2e} | {timing[hand['name']]['s_per_unit']:.3f} |")

lines.append("\n## Search summary\n")
lines.append("| method | candidates | feasible | early blow-up (<=500) | drift > 1% | best val err@1000 | GPU-hours |")
lines.append("|---|---|---|---|---|---|---|")
total_h = 0.0
for m, recs in search.items():
    feas = [r for r in recs if r["feasible"]]
    early = sum(r["early_reject"] for r in recs)
    drift = sum((not r["feasible"]) and (not r["early_reject"]) for r in recs)
    hrs = sum(r["wall_s"] for r in recs) / 3600
    total_h += hrs
    best = min((r["err1000"] for r in feas), default=float("nan"))
    lines.append(f"| {m} | {len(recs)} | {len(feas)} | {early} | {drift} | {best:.4f} | {hrs:.2f} |")

lines.append(f"\n## Front models, retrained with 3 seeds (front picked on {final['cost_key']} cost)\n")
lines.append("median [min-max] over seeds; val set; stability = 20k steps on 5 val ICs\n")
lines.append("| model | from | params | FLOPs/cell | err@100 | err@1000 | err@5000 | drift@5000 | 20k: blow-ups | 20k drift (median) | measured s/unit |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for e in final["front"]:
    s = e["seeds"]
    blow = sum(x["stability"]["blew_up_at"] is not None for x in s)
    lines.append(f"| {e['name']} | {e['method']} | {e['params']} | {e['flops']} | {spread([x['err100'] for x in s])} | "
                 f"{spread([x['err1000'] for x in s])} | {spread([x['err5000'] for x in s])} | "
                 f"{med([x['drift_mean'] for x in s]):.2e} | {blow}/3 | {med([x['stability']['drift_mean'] for x in s]):.2e} | "
                 f"{timing[e['name']]['s_per_unit']:.3f} |")
ys = final["yee64_stability"]
lines.append(f"| Yee 64 | - | 0 | 12 | | | | | {int(ys['blew_up_at'] is not None)}/1 | {ys['drift_mean']:.2e} | {timing['yee64']['s_per_unit']:.3f} |")

test = torch.load("data/test.pt", weights_only=False)
def fine_err(n, sel, k):
    return rel_l2(test[f"yee{n}"][sel, k], test["yee512"][sel, k])
subsets = {"random (25)": slice(0, 25), "hard lambda=4 cells (4)": slice(25, 29), "dielectric OOD (1)": slice(29, 30)}
lines.append("\n## Held-out test (scored once), err@1000 / err@5000, median over 3 seeds\n")
lines.append("| model | " + " | ".join(subsets) + " |")
lines.append("|---|" + "---|" * len(subsets))
tb = final["test_baselines"]
for name, keys in (("Yee 64", ("yee64", "yee64_hard", "yee64_dielectric")),):
    lines.append(f"| {name} | " + " | ".join(f"{tb[k]['err1000']:.4f} / {tb[k]['err5000']:.4f}" for k in keys) + " |")
for n in (128, 256):
    lines.append(f"| Yee {n} | " + " | ".join(f"{fine_err(n, s, 2):.4f} / {fine_err(n, s, 3):.4f}" for s in subsets.values()) + " |")
for e in final["front"]:
    cells = []
    for key in ("test_random", "test_hard", "test_dielectric"):
        a = [x[key]["err1000"] for x in e["seeds"]]; b = [x[key]["err5000"] for x in e["seeds"]]
        blow = sum(x[key]["blew_up_at"] is not None for x in e["seeds"])
        cells.append(f"{med(a):.4f} / {med(b):.4f}" + (f" ({blow} blew up)" if blow else ""))
    lines.append(f"| {e['name']} | " + " | ".join(cells) + " |")

# ---------- timing + rank correlation
front_names = [e["name"] for e in final["front"]]
learned = [n for n in timing if not n.startswith("yee")]
est = {}
for recs in list(search.values()) + [[hand]]:
    for r in recs:
        est[r["name"]] = r["est_s_per_unit_1024"]
off = sorted(n for n in learned if n not in front_names)[:5]
sub = front_names + off
rho_sub = spearmanr([est[n] for n in sub], [timing[n]["s_per_unit"] for n in sub]).correlation
rho_all = spearmanr([est[n] for n in learned], [timing[n]["s_per_unit"] for n in learned]).correlation
lines.append("\n## Timing (torch.compile'd step, fp32, warm GPU, median of 5 x 50 steps)\n")
lines.append("Coarse models at 1024^2, Yee at 1024^2/2048^2/4096^2 (same physical problem as 64/128/256 scaled 16x).\n")
lines.append("| model | estimated s/unit (roofline) | measured s/unit | timed at |")
lines.append("|---|---|---|---|")
for n in (64, 128, 256):
    t = timing[f"yee{n}"]
    lines.append(f"| Yee {n * 16}^2 | - | {t['s_per_unit']:.4f} | {t['timed_at']}^2 |")
for n in sub:
    lines.append(f"| {n}{' (front)' if n in front_names else ''} | {est[n]:.4f} | {timing[n]['s_per_unit']:.4f} | {timing[n]['timed_at']}^2 |")
lines.append(f"\nSpearman(estimated, measured): front + 5 off-front = **{rho_sub:.3f}** (n={len(sub)}); all {len(learned)} timed learned models = **{rho_all:.3f}**.")

final_h = final["gpu_s"] / 3600
lines.append(f"\n## GPU time\n\nSearch: {total_h:.2f} GPU-hours. Final retraining/evaluation: {final_h:.2f} GPU-hours.")
open("results/tables.md", "w").write("\n".join(lines) + "\n")
json.dump({"search_gpu_hours": total_h, "final_gpu_hours": final_h, "spearman_front_plus_5": rho_sub,
           "spearman_all": rho_all, "front": front_names,
           "front_val_err1000_median": {e["name"]: med([x["err1000"] for x in e["seeds"]]) for e in final["front"]},
           "timing_s_per_unit": {k: v["s_per_unit"] for k, v in timing.items()},
           "baselines_val_err1000": {f"yee{n}": base[f"yee{n}_val"]["err1000"] for n in (64, 128, 256)}},
          open("results/summary.json", "w"), indent=1)
print("\n".join(lines))
