"""Training and scoring for the material phase (D2)."""
import json, time
import torch
from .matmodel import MaterialNet, BoundMaterialStepper, tied_step
from .rollout import rollout
from .yee import COURANT

DEV = "cuda"
DRIFT_LIMIT = 0.01
TRAIN_STEPS, TRAIN_BATCH, UNROLL = 500, 8, 8
FIELD_BYTES = 24          # read Ez,Hx,Hy + write Ez,Hx,Hy, fp32


def load_mat(name):
    d = torch.load(f"data/mat_{name}.pt", weights_only=False)
    for k in ("eps", "mask", "yee512", "yee256", "yee128"):
        if k in d:
            d[k] = d[k].to(DEV)
    return d


def base_kernel(r):
    k = 2 * r + 1
    w = json.load(open("runs/baselines_v2.json"))[f"fixed_r{r}"]["kernel"]
    return torch.tensor(w)[0].view(2, k * k)


def rel_l2(pred, true, mask=None):
    if mask is not None:
        m = mask[:, None].float()
        pred, true = pred * m, true * m
    num = (pred - true).flatten(1).norm(dim=1)
    den = true.flatten(1).norm(dim=1).clamp_min(1e-12)
    return (num / den).mean().item()


def score(stepper, split, sel=slice(None), n_steps=5000):
    truth, eps, mask = split["yee512"][sel], split["eps"][sel], split["mask"][sel]
    steps = [s for s in split["steps"] if 0 < s <= n_steps]
    r = rollout(stepper, truth[:, 0], eps, n_steps, record_steps=steps)
    out = {"blew_up_at": r["blew_up_at"], "drift_mean": r["drift_mean"], "drift_max": r["drift_max"]}
    for s in steps:
        k = split["steps"].index(s)
        ok = s in r["records"]
        out[f"err{s}"] = rel_l2(r["records"][s], truth[:, k]) if ok else float("nan")
        out[f"ierr{s}"] = rel_l2(r["records"][s], truth[:, k], mask) if ok else float("nan")
    return out


def feasible(sc):
    return (sc["blew_up_at"] is None and sc["drift_mean"] < DRIFT_LIMIT
            and all(sc[k] == sc[k] for k in ("err1000", "err5000")))


def train_net(net, traj, eps, steps=TRAIN_STEPS, batch=TRAIN_BATCH, unroll=UNROLL, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    step = torch.compile(tied_step)
    r = net.design.stencil_radius
    n_traj, n_t = traj.shape[:2]
    loss_value = float("nan")
    for it in range(steps):
        i = torch.randint(n_traj, (batch,), generator=gen)
        t = torch.randint(n_t - unroll, (batch,), generator=gen)
        window = traj[i[:, None], t[:, None] + torch.arange(unroll + 1)].to(DEV, non_blocking=True)
        a, inv_m, _, _ = net.setup(eps[i].to(DEV))
        ez, hx, hy = window[:, 0, 0], window[:, 0, 1], window[:, 0, 2]
        loss = 0.0
        for u in range(1, unroll + 1):
            ez, hx, hy = step(ez, hx, hy, a, inv_m, r)
            target = window[:, u]
            loss = loss + ((torch.stack([ez, hx, hy], 1) - target) ** 2).mean() / (target ** 2).mean()
        loss = loss / unroll
        if not torch.isfinite(loss):
            return float("nan")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0 or it == steps - 1:
            loss_value = loss.item()
    return loss_value


@torch.no_grad()
def runtime_bytes(net, eps):
    """Bytes read+written per cell per step: fields + class index + the scene's weight table
    (amortised over the scene's cells). Mean over scenes."""
    k2 = net.base.shape[1]
    vals, classes = [], []
    for e in eps:
        uniq, idx = net.classes(e[None])
        n_cls = len(uniq)
        vals.append(FIELD_BYTES + idx.element_size() + n_cls * (2 * k2 + 1) * 4 / e.numel())
        classes.append(n_cls)
    return sum(vals) / len(vals), max(classes)


def bind(net, eps):
    with torch.no_grad():
        a, inv_m, _, _ = net.setup(eps)
    return BoundMaterialStepper(a, inv_m, net.design.stencil_radius)


def train_and_score(design, train, val, seed=0, steps=TRAIN_STEPS):
    t0 = time.time()
    net = MaterialNet(design, base_kernel(design.stencil_radius)).to(DEV)
    loss = train_net(net, train["traj"], train["eps"], steps=steps, seed=seed)
    torch.cuda.synchronize()
    train_s = time.time() - t0
    net.eval()
    stepper = bind(net, val["eps"])
    cn = (COURANT * stepper.sigma_max()).max().item()
    nbytes, n_cls = runtime_bytes(net, val["eps"])
    rec = {"design": design.__dict__, "name": design.name(), "seed": seed, "final_loss": loss,
           "params": sum(p.numel() for p in net.parameters()), "courant_sigma": cn,
           "bytes_per_cell": nbytes, "max_classes": n_cls, "train_s": train_s}
    if not cn < 2.0 or loss != loss:
        rec.update(rejected="T-STAB-3 violated" if loss == loss else "training diverged", feasible=False)
    else:
        rec.update(score(stepper, val), rejected=None)
        rec["feasible"] = feasible(rec)
    rec["wall_s"] = time.time() - t0
    return rec, net
