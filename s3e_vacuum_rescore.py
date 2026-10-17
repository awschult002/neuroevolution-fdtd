"""Stage 3e: vacuum table re-scored against the 2048^2 truth (validation, err@100/1000; drift@5000 with
the staggered energy). Yee 64/128/256, fixed free r1-3, D4 r1-3, D4 dispersion fits, hand-picked R1 net,
and the best R1 random-search design (retrained, seed 0, same protocol). Also reports 512^2-truth vs
2048^2-truth difference. Writes runs/vacuum_table_2048.json."""
import json, torch
from fdtd.evaluate import load_split, score_stepper, train_and_score
from fdtd.model import YeeStepper, LearnedStepper, HAND_PICKED, Design
from fdtd.rollout import rel_l2
from s9_mat_baselines import candidate, CANDIDATES


def main():
    val = load_split("val")
    res = {"truth512_vs_2048": {f"step{s}": rel_l2(val["truth512"][:, k].cuda(), val["yee512"][:, k])
                                for k, s in enumerate(val["steps"]) if s}}
    res["yee64"] = score_stepper(YeeStepper(), val)
    for n in (128, 256):
        res[f"yee{n}"] = {f"err{s}": rel_l2(val[f"yee{n}"][:, k], val["yee512"][:, k]) for k, s in enumerate(val["steps"]) if s}
    for c in CANDIDATES:
        res[c] = score_stepper(candidate(c), val)
    hand = LearnedStepper(HAND_PICKED).cuda().eval()
    hand.load_state_dict(torch.load("runs/hand_picked.pt"))
    res["hand_picked"] = score_stepper(hand, val)
    best_r1 = min(json.load(open("runs/r1_rescored.json")), key=lambda r: r["err1000"])
    traj = torch.load("data/train.pt", weights_only=False)["traj"].pin_memory()
    rec, _ = train_and_score(Design(**best_r1["design"]), traj, val, 500)
    res[f"r1best_{best_r1['name']}"] = {k: rec[k] for k in ("err100", "err1000", "drift_mean", "drift_max", "blew_up_at")}
    for k, v in res.items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()}, flush=True)
    json.dump(res, open("runs/vacuum_table_2048.json", "w"), indent=1)


if __name__ == "__main__":
    main()
