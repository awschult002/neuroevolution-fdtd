"""Stage 8 (D2): material-scene dataset (superseded truth: s8d_truth2048.py replaces it with 2048^2). 200 rectangle scenes -> 150 train / 25 val / 25 test, plus
10 unseen-shape test scenes (rotated rectangles, L-shapes). Truth = Yee 512^2 with point-sampled eps,
restricted to 64^2; Yee 128^2 / 256^2 use eps averaged from the 512^2 samples. Coarse eps = 8x8 average.
Writes data/mat_train.pt, data/mat_val.pt, data/mat_test.pt."""
import random, time, torch
from fdtd.scenes import random_scene, sample_scene, average, interface_mask
from fdtd.fine import run_fine_restricted

DEV = "cuda"
TRAIN_STEPS, EVAL_STEPS, CHUNK = 128, [0, 100, 1000, 5000], 25


def fields(scenes):
    s = [sample_scene(sc, 512, DEV) for sc in scenes]
    return torch.stack([a for a, _ in s]), torch.stack([b for _, b in s])


def simulate(scenes, n, record):
    out = []
    for c in range(0, len(scenes), CHUNK):
        ez512, eps512 = fields(scenes[c:c + CHUNK])
        f = 512 // n
        ez = ez512 if f == 1 else torch.stack([sample_scene(sc, n, DEV)[0] for sc in scenes[c:c + CHUNK]])
        eps = eps512 if f == 1 else average(eps512, f)
        out.append(run_fine_restricted(ez, 1.0 / eps, n // 64, record).cpu())
    return torch.cat(out)


def coarse_info(scenes):
    _, eps512 = fields(scenes)
    return average(eps512, 8).cpu(), interface_mask(eps512, 8).cpu()


def main():
    rng = random.Random(4242)
    scenes = [random_scene(rng) for _ in range(200)]
    splits = {"train": scenes[:150], "val": scenes[150:175],
              "test": scenes[175:] + [random_scene(rng, "unseen") for _ in range(10)]}
    t0 = time.time()
    eps, mask = coarse_info(splits["train"])
    traj = simulate(splits["train"], 512, list(range(TRAIN_STEPS + 1)))
    torch.save({"traj": traj, "eps": eps, "mask": mask, "scenes": splits["train"]}, "data/mat_train.pt")
    print(f"train {tuple(traj.shape)} {time.time() - t0:.0f}s", flush=True)
    for name in ("val", "test"):
        eps, mask = coarse_info(splits[name])
        d = {"scenes": splits[name], "steps": EVAL_STEPS, "eps": eps, "mask": mask}
        for n in (512, 256, 128):
            d[f"yee{n}"] = simulate(splits[name], n, EVAL_STEPS)
            print(f"{name} yee{n} {time.time() - t0:.0f}s", flush=True)
        torch.save(d, f"data/mat_{name}.pt")


if __name__ == "__main__":
    main()
