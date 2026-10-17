"""Stage 9 (D2): material baselines on val and test (random 25 / unseen shapes 10).
 (i)  coarse Yee 64^2 with cell-averaged eps; Yee 128^2/256^2 (restricted, eps averaged from 512^2)
 (ii) tuned fixed vacuum stencils r = 1, 2, 3 (from s3b) with cell-averaged eps
Errors overall and on interface cells (ierr). Writes runs/mat_baselines.json."""
import json, torch
from fdtd.mateval import load_mat, score, rel_l2
from fdtd.model import YeeStepper, FixedLinearStepper


def fixed(r):
    m = FixedLinearStepper(r).cuda()
    k = 2 * r + 1
    w = torch.tensor(json.load(open("runs/baselines_v2.json"))[f"fixed_r{r}"]["kernel"])[0].view(2, k * k).cuda()
    with torch.no_grad():
        m.delta.copy_(w - m.yee)                 # w already satisfies the constraints
    return m.eval()


def fine(split, n, sel):
    out = {}
    for k, s in enumerate(split["steps"]):
        if s > 0:
            t, p, m = split["yee512"][sel, k], split[f"yee{n}"][sel, k], split["mask"][sel]
            out[f"err{s}"], out[f"ierr{s}"] = rel_l2(p, t), rel_l2(p, t, m)
    return out


def main():
    res = {}
    for name, sels in (("val", {"val": slice(None)}), ("test", {"test_random": slice(0, 25), "test_unseen": slice(25, 35)})):
        split = load_mat(name)
        for tag, sel in sels.items():
            res[f"yee64_{tag}"] = score(YeeStepper(), split, sel)
            for n in (128, 256):
                res[f"yee{n}_{tag}"] = fine(split, n, sel)
            for r in (1, 2, 3):
                res[f"fixed_r{r}_{tag}"] = score(fixed(r), split, sel)
            for k, v in res.items():
                if k.endswith(tag):
                    print(k, json.dumps(v), flush=True)
    json.dump(res, open("runs/mat_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
