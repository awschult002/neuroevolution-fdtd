"""Stage 8d (T-MAT-1 outcome): the 512^2 reference failed T-MAT-1 (512 vs 1024 edge-band difference
0.045 at 1000 coarse steps, about half the coarse models' error), so the material reference moves to
2048^2 (eps point-sampled at 2048^2, compiled Yee, restricted to 64^2 with the shared operator).
Material error horizons become 100 and 1000 coarse steps (5000-step drift/stability need no truth).
Also stores Yee 1024/512/256/128 restricted for val/test as fine baselines, and the T-MAT-1 numbers
(each of 512 and 1024 vs 2048, edge band and bulk, all validation scenes).
Rewrites data/mat_{train,val,test}.pt in place (scenes/eps/mask kept); writes runs/reference_check_2048.json."""
import json, time, torch
from fdtd.scenes import sample_scene, average, interface_mask
from fdtd.fine import run_fine_restricted
from fdtd.mateval import rel_l2

DEV, TRUTH = "cuda", 2048
STEPS = [0, 100, 1000]


def simulate(scenes, n, record, chunk):
    out = []
    for c in range(0, len(scenes), chunk):
        part = scenes[c:c + chunk]
        ez = torch.stack([sample_scene(s, n, DEV)[0] for s in part])
        eps = torch.stack([sample_scene(s, n, DEV)[1] for s in part])
        out.append(run_fine_restricted(ez, 1.0 / eps, n // 64, record, compiled=True).cpu())
        del ez, eps
        torch.cuda.empty_cache()
    return torch.cat(out)


def main():
    t0 = time.time()
    check = {}
    for name in ("val", "test", "train"):
        d = torch.load(f"data/mat_{name}.pt", weights_only=False)
        if d.get("truth_n") == TRUTH:
            continue
        if name == "train":
            d["traj"] = simulate(d["scenes"], TRUTH, list(range(129)), 8)
        else:
            d["steps"] = STEPS
            d["yee512"] = simulate(d["scenes"], TRUTH, STEPS, 8)        # key kept: "yee512" = the truth
            for n in (1024, 512, 256, 128):
                d[f"fine{n}"] = simulate(d["scenes"], n, STEPS, 8 if n >= 1024 else 25)
            for n in (256, 128):
                d[f"yee{n}"] = d[f"fine{n}"]
            if name == "val":
                m = d["mask"]
                for n in (1024, 512):
                    for k, s in enumerate(STEPS[1:], start=1):
                        check[f"{n}_vs_2048_step{s}"] = {
                            "edge": rel_l2(d[f"fine{n}"][:, k], d["yee512"][:, k], m),
                            "bulk": rel_l2(d[f"fine{n}"][:, k], d["yee512"][:, k], ~m)}
                json.dump(check, open("runs/reference_check_2048.json", "w"), indent=1)
                print(json.dumps(check, indent=1), flush=True)
        d["truth_n"] = TRUTH
        torch.save(d, f"data/mat_{name}.pt")
        print(name, "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
