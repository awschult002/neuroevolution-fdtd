"""Stage 2b: vacuum reference moved to 2048^2 (same reason as T-MAT-1 for materials), error horizons
100 and 1000 coarse steps. Rewrites data/val.pt and data/test.pt: "yee512" (= the truth key) becomes
the 2048^2 restriction, steps = [0, 100, 1000], yee256/yee128 are sliced to those steps; the old 512^2
truth is kept as "truth512". (Training targets stay the 512^2 restriction over 128 steps.)"""
import torch
from fdtd.initial import sample
from fdtd.fine import run_fine_restricted

STEPS = [0, 100, 1000]


def main():
    for name in ("val", "test"):
        d = torch.load(f"data/{name}.pt", weights_only=False)
        if d.get("truth_n") == 2048:
            continue
        keep = [d["steps"].index(s) for s in STEPS]
        out = []
        for c in range(0, len(d["ics"]), 8):
            part = d["ics"][c:c + 8]
            f = [sample(ic, 2048, "cuda") for ic in part]
            ez, eps = torch.stack([a for a, _ in f]), torch.stack([b for _, b in f])
            out.append(run_fine_restricted(ez, 1.0 / eps, 32, STEPS, compiled=True).cpu())
            print(name, c, flush=True)
        d["truth512"] = d["yee512"][:, keep]
        d["yee512"] = torch.cat(out)
        for n in (256, 128):
            d[f"yee{n}"] = d[f"yee{n}"][:, keep]
        d["steps"], d["truth_n"] = STEPS, 2048
        torch.save(d, f"data/{name}.pt")


if __name__ == "__main__":
    main()
