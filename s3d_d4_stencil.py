"""Stage 3d: D4-constrained tuned stencil vs the free (44-weight at r=2) tuned stencil, vacuum.
Same training protocol as s3b. Compares val err@100/1000/5000 and phase error at 0/22.5/45 deg for
4/8/16 cells per wavelength (free r1-r3 and D4 r1-r3, plus Yee). Writes runs/d4_vs_free.json."""
import json, torch
from fdtd.evaluate import load_split, score_stepper, TRAIN_STEPS, TRAIN_BATCH
from fdtd.model import D4FixedStepper, YeeStepper
from fdtd.train import train_model
from fdtd.dispersion import dispersion_table
from s9_mat_baselines import fixed


def main():
    traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
    val = load_split("val")
    res = {"yee64": {"val": score_stepper(YeeStepper(), val), "dispersion": dispersion_table(YeeStepper()), "params": 0}}
    for r in (1, 2, 3):
        m = fixed(r)
        res[f"free_r{r}"] = {"val": score_stepper(m, val), "dispersion": dispersion_table(m), "params": 2 * (2 * r + 1) ** 2,
                             "courant_sigma": 0.5 * m.sigma_max()}
        d = D4FixedStepper(r).cuda()
        loss = train_model(d, traj, TRAIN_STEPS, batch=TRAIN_BATCH, seed=0)
        d.eval()
        res[f"d4_r{r}"] = {"val": score_stepper(d, val), "dispersion": dispersion_table(d), "params": len(d.dp),
                           "courant_sigma": 0.5 * d.sigma_max(), "final_loss": loss,
                           "kernel": d.kernel().detach().cpu().tolist()}
        torch.save(d.state_dict(), f"runs/d4_r{r}.pt")
        for k in (f"free_r{r}", f"d4_r{r}"):
            v = res[k]
            print(k, {a: round(b, 4) for a, b in v["val"].items() if a.startswith("err")}, "cs", round(v["courant_sigma"], 4),
                  " ".join(f"{a}:{b['phase_err_per_wavelength']:+.3f}" for a, b in v["dispersion"].items()), flush=True)
    json.dump(res, open("runs/d4_vs_free.json", "w"), indent=1)


if __name__ == "__main__":
    main()
