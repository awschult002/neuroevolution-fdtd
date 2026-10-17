"""Train-and-score one design, and score any stepper on a val/test split."""
import time
import torch
from .model import LearnedStepper, parameter_count
from .rollout import rollout, rel_l2
from .train import train_model
from . import cost

DEV = "cuda"
DRIFT_LIMIT = 0.01
EARLY_BLOWUP = 500
TRAIN_BATCH = 8          # scaled down from 16 (shared 4 GB GPU)
TRAIN_STEPS = 500        # scaled down from 2000 (T2000 too slow for 2000 x ~80 candidates)


def load_split(name):
    d = torch.load(f"data/{name}.pt", weights_only=False)
    for key in ("yee512", "yee256", "yee128", "eps"):
        d[key] = d[key].to(DEV)
    return d


def score_stepper(stepper, split, n_steps=5000, select=None):
    """Errors vs restricted 512^2 truth at the recorded steps, plus energy drift at n_steps."""
    truth = split["yee512"] if select is None else split["yee512"][select]
    eps = split["eps"] if select is None else split["eps"][select]
    steps = [s for s in split["steps"] if 0 < s <= n_steps]
    r = rollout(stepper, truth[:, 0], eps, n_steps, record_steps=steps)
    out = {"blew_up_at": r["blew_up_at"], "drift_mean": r["drift_mean"], "drift_max": r["drift_max"]}
    for s in steps:
        k = split["steps"].index(s)
        out[f"err{s}"] = rel_l2(r["records"][s], truth[:, k]) if s in r["records"] else float("nan")
    return out


def feasible(score):
    return (score["blew_up_at"] is None and score["drift_mean"] < DRIFT_LIMIT
            and all(v == v for k, v in score.items() if k.startswith("err")))


def train_and_score(design, traj, val, train_steps, seed=0):
    torch._dynamo.reset()                 # fresh compile cache per candidate (avoids recompile limit)
    t0 = time.time()
    model = LearnedStepper(design).to(DEV)
    loss = train_model(model, traj, train_steps, batch=TRAIN_BATCH, seed=seed)
    torch.cuda.synchronize()
    train_s = time.time() - t0
    model.eval()
    score = score_stepper(model, val)
    rec = {"design": design.__dict__, "name": design.name(), "seed": seed, "final_loss": loss,
           "params": parameter_count(model), "flops": cost.flops_per_cell(design),
           "bytes": cost.bytes_per_cell(design),
           "est_s_per_unit_1024": cost.seconds_per_unit_time(cost.flops_per_cell(design),
                                                             cost.bytes_per_cell(design), 1024),
           "train_s": train_s, "eval_s": time.time() - t0 - train_s, **score}
    rec["early_reject"] = score["blew_up_at"] is not None and score["blew_up_at"] <= EARLY_BLOWUP
    rec["feasible"] = feasible(score) and loss == loss
    return rec, model
