"""Stage 3: baselines on val and test: coarse Yee 64^2, Yee 128^2 / 256^2 (restricted), and the
hand-picked Bar-Sinai-style learned stencil (5x5, 2 hidden layers x 16, relu, predicts coefficients).
Writes runs/baselines.json and runs/hand_picked.pt."""
import json, sys, torch
from fdtd.evaluate import load_split, score_stepper, train_and_score, rel_l2, TRAIN_STEPS as DEFAULT_STEPS
from fdtd.model import YeeStepper, HAND_PICKED
from fdtd.yee import energy

TRAIN_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEPS


def fine_scores(split, n):
    out = {}
    for k, s in enumerate(split["steps"]):
        if s > 0:
            out[f"err{s}"] = rel_l2(split[f"yee{n}"][:, k], split["yee512"][:, k])
    return out


def main():
    traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
    res = {}
    for name in ("val", "test"):
        split = load_split(name)
        res[f"yee64_{name}"] = score_stepper(YeeStepper(), split)
        res[f"yee128_{name}"] = fine_scores(split, 128)
        res[f"yee256_{name}"] = fine_scores(split, 256)
        print(name, json.dumps({k: v for k, v in res.items() if k.endswith(name)}, indent=1), flush=True)
    val = load_split("val")
    rec, model = train_and_score(HAND_PICKED, traj, val, TRAIN_STEPS)
    res["hand_picked_val"] = rec
    print(json.dumps(rec, indent=1), flush=True)
    torch.save(model.state_dict(), "runs/hand_picked.pt")
    json.dump(res, open("runs/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
