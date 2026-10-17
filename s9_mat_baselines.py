"""Stage 9 (D2): material baselines. VALIDATION only by default (the bar is frozen from validation);
`python s9_mat_baselines.py test` scores the held-out test once at the very end.
 (i)  coarse Yee 64^2 with cell-averaged eps (T-MAT-2: the one coarse-eps rule for every method);
      Yee 128^2/256^2 (restricted, eps averaged from 512^2)
 (ii) tuned fixed vacuum stencils, free (s3b) and D4-constrained (s3d), r = 1, 2, 3, with the same
      cell-averaged eps (subpixel smoothing; Ez is tangential to every interface in TMz)
Errors: all cells (err), +-2-cell edge band (ierr), bulk (berr). The bar = the baseline-(ii) stencil
with the lowest validation edge-band err@1000; its family (free or D4) also provides the networks'
base stencils. Writes runs/mat_baselines.json (+ runs/mat_bar.json)."""
import json, sys, torch
from fdtd.mateval import load_mat, score, rel_l2
from fdtd.model import YeeStepper, FixedLinearStepper, D4FixedStepper


def fixed(r):
    m = FixedLinearStepper(r).cuda()
    k = 2 * r + 1
    w = torch.tensor(json.load(open("runs/baselines_v2.json"))[f"fixed_r{r}"]["kernel"])[0].view(2, k * k).cuda()
    with torch.no_grad():
        m.delta.copy_(w - m.yee)                 # w already satisfies the constraints
    return m.eval()


def d4(r):
    m = D4FixedStepper(r).cuda()
    m.load_state_dict(torch.load(f"runs/d4_r{r}.pt"))
    return m.eval()


def fine(split, n, sel):
    out = {}
    for k, s in enumerate(split["steps"]):
        if s > 0:
            t, p, m = split["yee512"][sel, k], split[f"yee{n}"][sel, k], split["mask"][sel]
            out[f"err{s}"], out[f"ierr{s}"], out[f"berr{s}"] = rel_l2(p, t), rel_l2(p, t, m), rel_l2(p, t, ~m)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    path = "runs/mat_baselines.json"
    res = json.load(open(path)) if which == "test" else {}
    if which == "val":
        sets = (("val", {"val": slice(None)}),)
    else:
        sets = (("test", {"test_random": slice(0, 25), "test_unseen": slice(25, 35), "test_extra": slice(35, 41)}),)
    for name, sels in sets:
        split = load_mat(name)
        for tag, sel in sels.items():
            res[f"yee64_{tag}"] = score(YeeStepper(), split, sel)
            for n in (128, 256):
                res[f"yee{n}_{tag}"] = fine(split, n, sel)
            for r in (1, 2, 3):
                res[f"fixed_r{r}_{tag}"] = score(fixed(r), split, sel)
                res[f"d4_r{r}_{tag}"] = score(d4(r), split, sel)
            for k, v in res.items():
                if k.endswith(tag):
                    print(k, json.dumps(v), flush=True)
    if which == "val":
        cands = [f"{f}_r{r}" for f in ("fixed", "d4") for r in (1, 2, 3)]
        best = min(cands, key=lambda c: res[c + "_val"]["ierr1000"])
        json.dump({"bar": best, "family": best.split("_")[0], "val_ierr1000": res[best + "_val"]["ierr1000"]},
                  open("runs/mat_bar.json", "w"), indent=1)
        print("BAR", best, res[best + "_val"]["ierr1000"], flush=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
