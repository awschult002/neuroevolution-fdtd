"""Stage 8e (T-MAT-2): the fine-Yee baselines (1024/512/256/128) must use the same eps rule as every
coarse method: eps cell-averaged from the 2048^2 point samples (s8d had point-sampled them at n).
Rewrites fine{n} / yee{n} in data/mat_val.pt and data/mat_test.pt."""
import torch
from fdtd.scenes import sample_scene, average
from fdtd.fine import run_fine_restricted

STEPS = [0, 100, 1000]


def main():
    for name in ("val", "test"):
        d = torch.load(f"data/mat_{name}.pt", weights_only=False)
        for n in (1024, 512, 256, 128):
            out = []
            for c in range(0, len(d["scenes"]), 8):
                part = d["scenes"][c:c + 8]
                ez = torch.stack([sample_scene(s, n, "cuda")[0] for s in part])
                eps = torch.stack([average(sample_scene(s, 2048, "cuda")[1][None], 2048 // n)[0] for s in part])
                out.append(run_fine_restricted(ez, 1.0 / eps, n // 64, STEPS, compiled=True).cpu())
            d[f"fine{n}"] = torch.cat(out)
            if n in (256, 128):
                d[f"yee{n}"] = d[f"fine{n}"]
            print(name, n, flush=True)
        d["fine_eps_rule"] = "cell-averaged from 2048^2"
        torch.save(d, f"data/mat_{name}.pt")


if __name__ == "__main__":
    main()
