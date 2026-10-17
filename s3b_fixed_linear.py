"""Stage 3b: re-score R1 baselines with the staggered energy, train the fixed-in-time linear
baseline (FixedLinearStepper, r = 1, 2, 3), FFT stability check at the coarse dt (rescale if
violated, logged), and the dispersion check for all of them. Writes runs/baselines_v2.json."""
import json, time, torch
from fdtd.evaluate import load_split, score_stepper, TRAIN_STEPS, TRAIN_BATCH
from fdtd.model import FixedLinearStepper, LearnedStepper, YeeStepper, HAND_PICKED
from fdtd.train import train_model
from fdtd.dispersion import dispersion_table


def main():
    traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
    val, test = load_split("val"), load_split("test")
    res = {}
    hand = LearnedStepper(HAND_PICKED).cuda().eval()
    hand.load_state_dict(torch.load("runs/hand_picked.pt"))
    for name, st in (("yee64", YeeStepper()), ("hand_picked", hand)):
        res[name] = {"val": score_stepper(st, val), "test_random": score_stepper(st, test, select=slice(0, 25)),
                     "dispersion": dispersion_table(st)}
        print(name, json.dumps(res[name]), flush=True)
    for r in (1, 2, 3):
        m = FixedLinearStepper(r).cuda()
        t0 = time.time()
        loss = train_model(m, traj, TRAIN_STEPS, batch=TRAIN_BATCH, seed=0)
        cn = 0.5 * m.sigma_max()
        factor = m.rescale_to_stable()
        m.eval()
        res[f"fixed_r{r}"] = {"final_loss": loss, "train_s": time.time() - t0, "courant_sigma_trained": cn,
                              "stability_action": "none" if factor == 1.0 else f"rescaled by {factor:.4f}",
                              "params": 2 * (2 * r + 1) ** 2 - 6,
                              "kernel": m.kernel().detach().cpu().tolist(),
                              "val": score_stepper(m, val), "test_random": score_stepper(m, test, select=slice(0, 25)),
                              "dispersion": dispersion_table(m)}
        print(f"fixed_r{r}", json.dumps({k: v for k, v in res[f'fixed_r{r}'].items() if k != 'kernel'}), flush=True)
    json.dump(res, open("runs/baselines_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
