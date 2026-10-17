"""Stage 2: dataset. 200 random pulse ICs -> 150 train / 25 val / 25 test, plus hard test cases.
Truth = Yee 512^2 restricted to 64^2. Val/test also get Yee 128^2 and 256^2 restricted the same way.
Writes data/train.pt, data/val.pt, data/test.pt."""
import random, time, torch
from fdtd.initial import random_pulses, hard_wave_packet, dielectric_case, sample
from fdtd.fine import run_fine_restricted
from fdtd.restrict import restrict_ez

DEV = "cuda"
TRAIN_STEPS = 128            # coarse steps of truth kept per training trajectory (2 unit times)
EVAL_STEPS = [0, 100, 1000, 5000]
CHUNK = 25


def simulate(ics, n, record):
    out = []
    for c in range(0, len(ics), CHUNK):
        fields = [sample(ic, n, DEV) for ic in ics[c:c + CHUNK]]
        ez = torch.stack([f[0] for f in fields])
        inv_eps = 1.0 / torch.stack([f[1] for f in fields])
        out.append(run_fine_restricted(ez, inv_eps, n // 64, record).cpu())
    return torch.cat(out)


def coarse_eps(ics):
    return torch.stack([restrict_ez(sample(ic, 512, DEV)[1][None], 8)[0] for ic in ics]).cpu()


def main():
    rng = random.Random(2026)
    ics = [random_pulses(rng) for _ in range(200)]
    splits = {"train": ics[:150], "val": ics[150:175],
              "test": ics[175:] + [hard_wave_packet(rng) for _ in range(4)] + [dielectric_case(rng)]}
    t0 = time.time()
    traj = simulate(splits["train"], 512, list(range(TRAIN_STEPS + 1)))
    torch.save({"traj": traj, "ics": splits["train"]}, "data/train.pt")
    print(f"train {tuple(traj.shape)} {time.time() - t0:.0f}s", flush=True)
    for name in ("val", "test"):
        d = {"ics": splits[name], "steps": EVAL_STEPS, "eps": coarse_eps(splits[name])}
        for n in (512, 256, 128):
            d[f"yee{n}"] = simulate(splits[name], n, EVAL_STEPS)
            print(f"{name} yee{n} {time.time() - t0:.0f}s", flush=True)
        torch.save(d, f"data/{name}.pt")


if __name__ == "__main__":
    main()
