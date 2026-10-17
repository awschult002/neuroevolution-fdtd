"""Stage 13: plots and tables for the D1 (vacuum) and D2 (materials) phases.
Writes results/pareto.png, results/stability.png, results/tables.md, results/summary.json.
The verdict in REPORT.md is written by hand from these numbers."""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from fdtd.pareto import front

J = lambda p: json.load(open(p))
cons, v2, r1 = J("runs/conservation.json"), J("runs/baselines_v2.json"), J("runs/r1_rescored.json")
mb, fin = J("runs/mat_baselines.json"), J("runs/mat_final.json")
tim = J("runs/mat_timing.json") if os.path.exists("runs/mat_timing.json") else {}
search = {m: [json.loads(l) for l in open(f"runs/mat_search_{m}.jsonl")] for m in ("random", "evolution")}
med = statistics.median
f4 = lambda x: f"{x:.4f}"
spread = lambda xs: f"{med(xs):.4f} [{min(xs):.4f}-{max(xs):.4f}]"
L = []

L += ["## T-STAB-4 conservation unit test (20,000 steps, 5 val ICs, max drift over ICs)\n",
      "| run | staggered @5k | staggered @20k | naive @20k (info) |", "|---|---|---|---|"]
for k, v in cons.items():
    L.append(f"| {k} | {v['5000']['staggered']:.1e} | {v['20000']['staggered']:.1e} | {v['20000']['naive']:.1e} |")

L += ["\n## Vacuum phase (D1), validation set, staggered energy\n",
      "| method | err@100 | err@1000 | err@5000 | drift@5000 | params |", "|---|---|---|---|---|---|"]
for k, label in (("yee64", "Yee 64"), ("hand_picked", "hand-picked Bar-Sinai net (R1)")):
    v = v2[k]["val"]
    L.append(f"| {label} | {f4(v['err100'])} | {f4(v['err1000'])} | {f4(v['err5000'])} | {v['drift_mean']:.1e} | {'0' if k == 'yee64' else '3188'} |")
for r in (1, 2, 3):
    v = v2[f"fixed_r{r}"]
    L.append(f"| fixed linear paired r={r} | {f4(v['val']['err100'])} | {f4(v['val']['err1000'])} | {f4(v['val']['err5000'])} | {v['val']['drift_mean']:.1e} | {v['params']} |")
L += ["\nR1 random-search designs re-scored with the staggered energy (retrained, same protocol):\n",
      "| design | err@1000 | staggered drift | R1 naive drift |", "|---|---|---|---|"]
for r in r1:
    L.append(f"| {r['name']} | {f4(r['err1000'])} | {r['drift_mean']:.3f} | {r['r1_naive_drift']:.3f} |")

L += ["\n## Material phase (D2) baselines: err@1000 / interface err@1000 / err@5000\n",
      "| method | val | test random (25) | test unseen shapes (10) |", "|---|---|---|---|"]
for k, label in (("yee64", "Yee 64 (avg eps)"), ("yee128", "Yee 128"), ("yee256", "Yee 256"),
                 ("fixed_r1", "fixed vacuum r1 (avg eps)"), ("fixed_r2", "fixed vacuum r2 (avg eps)"), ("fixed_r3", "fixed vacuum r3 (avg eps)")):
    cells = [f"{f4(mb[f'{k}_{t}']['err1000'])} / {f4(mb[f'{k}_{t}']['ierr1000'])} / {f4(mb[f'{k}_{t}']['err5000'])}" for t in ("val", "test_random", "test_unseen")]
    L.append(f"| {label} | " + " | ".join(cells) + " |")
best_fixed = min((1, 2, 3), key=lambda r: mb[f"fixed_r{r}_val"]["ierr1000"])
bar = mb[f"fixed_r{best_fixed}_val"]["ierr1000"]

L += ["\n## Material search (val)\n", "| method | candidates | T-STAB-3 rejects | feasible | best val interface err@1000 | beat fixed r%d (%.4f) | GPU-hours |" % (best_fixed, bar),
      "|---|---|---|---|---|---|---|"]
hours = {}
for m, recs in search.items():
    feas = [r for r in recs if r["feasible"]]
    hours[m] = sum(r["wall_s"] for r in recs) / 3600
    best = min((r["ierr1000"] for r in feas), default=float("nan"))
    L.append(f"| {m} | {len(recs)} | {sum(r['rejected'] is not None for r in recs)} | {len(feas)} | {f4(best)} | {sum(r['ierr1000'] < bar for r in feas)} | {hours[m]:.2f} |")

L += ["\n## Front models, 3 seeds (median [min-max]); val; stability = 20k steps on 5 val scenes\n",
      "| model | from | err@1000 | interface err@1000 | err@5000 | drift@5000 | 20k blow-ups | 20k drift | bytes/cell | classes |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for e in fin["front"]:
    s = [x for x in e["seeds"] if not x["rejected"]]
    if not s:
        L.append(f"| {e['name']} | {e['method']} | all seeds rejected | | | | | | | |")
        continue
    L.append(f"| {e['name']} | {e['method']} | {spread([x['err1000'] for x in s])} | {spread([x['ierr1000'] for x in s])} | {spread([x['err5000'] for x in s])} | "
             f"{med([x['drift_mean'] for x in s]):.1e} | {sum(x['stability']['blew_up_at'] is not None for x in s)}/{len(s)} | {med([x['stability']['drift_mean'] for x in s]):.1e} | "
             f"{med([x['bytes_per_cell'] for x in s]):.1f} | {max(x['max_classes'] for x in s)} |")
for k, v in fin["baselines"].items():
    st = v["stability"]
    L.append(f"| {k} | baseline | {f4(mb[k + '_val']['err1000'])} | {f4(mb[k + '_val']['ierr1000'])} | {f4(mb[k + '_val']['err5000'])} | {mb[k + '_val']['drift_mean']:.1e} | {int(st['blew_up_at'] is not None)}/1 | {st['drift_mean']:.1e} | 28.0 | - |")

L += ["\n## Held-out test (scored once): err@1000 / interface err@1000, median over seeds\n",
      "| model | random (25) | unseen shapes (10) |", "|---|---|---|"]
for k in ("yee64", "yee256", f"fixed_r{best_fixed}"):
    L.append(f"| {k} | {f4(mb[k + '_test_random']['err1000'])} / {f4(mb[k + '_test_random']['ierr1000'])} | {f4(mb[k + '_test_unseen']['err1000'])} / {f4(mb[k + '_test_unseen']['ierr1000'])} |")
for e in fin["front"]:
    s = [x for x in e["seeds"] if not x["rejected"]]
    if s:
        L.append(f"| {e['name']} | {f4(med([x['test_random']['err1000'] for x in s]))} / {f4(med([x['test_random']['ierr1000'] for x in s]))} | "
                 f"{f4(med([x['test_unseen']['err1000'] for x in s]))} / {f4(med([x['test_unseen']['ierr1000'] for x in s]))} |")

def disp_row(name, d):
    return f"| {name} | " + " | ".join(f"{d[f'{a}_{c}']['phase_err_per_wavelength']:+.4f}" for a in ("x", "diag") for c in (4, 8, 16)) + " |"
L += ["\n## T-DISP-1 phase error per wavelength [rad] (vacuum; diag uses the nearest periodic wave: 4.11, 7.54, 15.08 cells)\n",
      "| model | x 4 | x 8 | x 16 | 45° 4 | 45° 8 | 45° 16 |", "|---|---|---|---|---|---|---|"]
for k, v in fin["baselines"].items():
    L.append(disp_row(k, v["dispersion"]))
for e in fin["front"]:
    s = [x for x in e["seeds"] if not x["rejected"]]
    if s:
        L.append(disp_row(e["name"] + " (seed 0)", s[0]["dispersion"]))

L += ["\n## Fresnel slab test (eps 4 half-slab, interface mid-cell, 8 cells/wavelength): reflected energy R vs |r_s|^2\n"]
angles = [f"{x['theta_deg']:.1f}°" for x in fin["baselines"]["yee64"]["fresnel"]]
L += ["| model | " + " | ".join(angles) + " | mean abs err |", "|---" * (len(angles) + 2) + "|"]
L.append("| analytic | " + " | ".join(f"{x['R_fresnel']:.4f}" for x in fin["baselines"]["yee64"]["fresnel"]) + " | 0 |")
def fres_row(name, fr):
    return f"| {name} | " + " | ".join(f"{x['R']:.4f}" for x in fr) + f" | {statistics.mean(x['R_abs_err'] for x in fr):.4f} |"
for k, v in fin["baselines"].items():
    L.append(fres_row(k, v["fresnel"]))
for e in fin["front"]:
    s = [x for x in e["seeds"] if not x["rejected"]]
    if s:
        L.append(fres_row(e["name"] + " (seed 0)", s[0]["fresnel"]))

if tim:
    L += ["\n## Timing (T2000, fp32, torch.compile, median of 5 valid runs; s per simulated time unit)\n",
          "| method | grid | s / unit | valid runs | clock range MHz | temp range C |", "|---|---|---|---|---|---|"]
    for k, v in tim.items():
        if k.startswith("_"):
            continue
        runs = v["runs"]
        ok = [r for r in runs if r["valid"]]
        L.append(f"| {k} | {v['n']}^2 (timed {v['timed_at']}^2) | {v['s_per_unit']:.3f} | {len(ok)}/{len(runs)} | "
                 f"{min(r['sm_clock'] for r in runs)}-{max(r['sm_clock'] for r in runs)} | {min(r['temp_c'] for r in runs)}-{max(r['temp_c'] for r in runs)} |")
    L.append(f"\nEnvironment: `{json.dumps(tim['_env'])}`")

L.append(f"\n## GPU time\n\nMaterial search: random {hours['random']:.2f} h, evolution {hours['evolution']:.2f} h. Final retrain/eval: {fin.get('gpu_s', 0) / 3600:.2f} h.")
open("results/tables.md", "w").write("\n".join(L) + "\n")

# Pareto: val interface err@1000 vs estimated bytes/cell (all candidates), measured time (timed models)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for m, c in (("random", "tab:blue"), ("evolution", "tab:orange")):
    pts = [(r["bytes_per_cell"], r["ierr1000"]) for r in search[m] if r["feasible"]]
    if pts:
        ax.scatter(*zip(*pts), c=c, alpha=0.5, label=f"{m} ({len(pts)} feasible)")
        f = sorted(pts[i] for i in front(pts))
        ax.step(*zip(*f), where="post", c=c)
for k, mk in (("yee64", "ks"), ("fixed_r1", "g^"), ("fixed_r2", "g*"), ("fixed_r3", "gv")):
    ax.plot(28.0, mb[k + "_val"]["ierr1000"], mk, ms=10, label=k)
ax.axhline(mb["yee256_val"]["ierr1000"], c="k", ls=":", label="Yee 256 (error only)")
ax.set_xlabel("estimated bytes per cell per step (fields + index + table)"); ax.set_ylabel("val interface rel. L2 err @1000")
ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("All searched designs (estimated cost)")
ax = axes[1]
if tim:
    xs, ys, names = [], [], []
    for k in ("yee64", "yee128", "yee256"):
        if k in tim:
            xs.append(tim[k]["s_per_unit"]); ys.append(mb[k + "_val"]["ierr1000"]); names.append(k)
    ax.plot(xs, ys, "k-s", label="Yee 64/128/256 (1024/2048/4096^2)")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for r in (1, 2, 3):
        k = f"fixed_r{r}"
        ax.plot(tim[k]["s_per_unit"], mb[k + "_val"]["ierr1000"], "g^", ms=9)
        ax.annotate(k, (tim[k]["s_per_unit"], mb[k + "_val"]["ierr1000"]), xytext=(4, -10), textcoords="offset points", fontsize=7)
    for e in fin["front"]:
        s = [x for x in e["seeds"] if not x["rejected"]]
        if s and e["name"] in tim:
            c = "tab:blue" if e["method"] == "random" else "tab:orange"
            ax.plot(tim[e["name"]]["s_per_unit"], med([x["ierr1000"] for x in s]), "o", c=c)
            ax.annotate(e["name"], (tim[e["name"]]["s_per_unit"], med([x["ierr1000"] for x in s])), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("measured s per simulated time unit"); ax.set_ylabel("val interface err @1000 (front: median of 3 seeds)")
ax.grid(alpha=0.3, which="both"); ax.set_title("Measured cost (blue=random, orange=evolution front)")
fig.tight_layout(); fig.savefig("results/pareto.png", dpi=110)

fig, ax = plt.subplots(figsize=(7, 4))
for k, v in fin["baselines"].items():
    e = v["stability"]["energy_curve"]
    ax.plot([100 * (i + 1) for i in range(len(e))], [x / e[0] - 1 for x in e], lw=2, label=k)
for ent in fin["front"]:
    s = [x for x in ent["seeds"] if not x["rejected"]]
    if s:
        e = s[0]["stability"]["energy_curve"]
        ax.plot([100 * (i + 1) for i in range(len(e))], [x / e[0] - 1 for x in e], label=ent["name"], ls="--")
ax.axhline(0.01, c="r", lw=0.8); ax.axhline(-0.01, c="r", lw=0.8)
ax.set_xlabel("coarse step"); ax.set_ylabel("U_n/U_100 - 1 (staggered energy, true eps)")
ax.set_title("20,000-step stability, material scenes"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/stability.png", dpi=110)

summ = {"best_fixed_radius": best_fixed, "fixed_bar_val_ierr1000": bar, "search_gpu_hours": hours,
        "final_gpu_hours": fin.get("gpu_s", 0) / 3600,
        "front": {e["name"]: med([x["ierr1000"] for x in e["seeds"] if not x["rejected"]] or [float("nan")]) for e in fin["front"]},
        "timing_s_per_unit": {k: v["s_per_unit"] for k, v in tim.items() if not k.startswith("_")}}
if tim:
    timed = [k for k in tim if not k.startswith(("_", "yee128", "yee256"))]
    bytes_est = {"yee64": 28.0, "fixed_r1": 28.0, "fixed_r2": 28.0, "fixed_r3": 28.0}
    for e in fin["front"]:
        s = [x for x in e["seeds"] if not x["rejected"]]
        if s:
            bytes_est[e["name"]] = med([x["bytes_per_cell"] for x in s])
    timed = [k for k in timed if k in bytes_est]
    if len(timed) > 2:
        summ["spearman_bytes_vs_time"] = spearmanr([bytes_est[k] for k in timed], [tim[k]["s_per_unit"] for k in timed]).correlation
json.dump(summ, open("results/summary.json", "w"), indent=1)
print("\n".join(L)); print(json.dumps(summ, indent=1))
