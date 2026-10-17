"""Stage 8b: (1) recompute the +-2-cell edge masks for all material splits, (2) append the six
T-MAT-3 extra held-out scenes (high contrast, one-cell-wide strip, mid-cell edges) to data/mat_test.pt
with their own 512/256/128 truth. Idempotent: skips the append if already done."""
import random, torch
from fdtd.scenes import extra_test_scenes, sample_scene, interface_mask
from s8_make_materials import simulate, coarse_info, EVAL_STEPS


def main():
    for name in ("train", "val", "test"):
        d = torch.load(f"data/mat_{name}.pt", weights_only=False)
        d["eps"], d["mask"] = coarse_info(d["scenes"])
        if name == "test" and len(d["scenes"]) == 35:
            extra = extra_test_scenes(random.Random(777))
            eps, mask = coarse_info(extra)
            d["scenes"] += extra
            d["eps"], d["mask"] = coarse_info(d["scenes"])
            for n in (512, 256, 128):
                d[f"yee{n}"] = torch.cat([d[f"yee{n}"], simulate(extra, n, EVAL_STEPS)])
                print("extra", n, flush=True)
        torch.save(d, f"data/mat_{name}.pt")
        print(name, "edge-band fraction", d["mask"].float().mean().item(), flush=True)


if __name__ == "__main__":
    main()
