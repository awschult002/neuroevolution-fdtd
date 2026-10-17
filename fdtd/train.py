"""Train a learned stepper on unrolled rollouts of the restricted truth."""
import torch


def train_model(model, traj, steps, unroll=8, batch=16, lr=1e-3, seed=0):
    """traj: (n_traj, n_t, 3, n, n), kept in (pinned) CPU memory to leave VRAM free; each batch
    window of unroll+1 states is copied to the GPU. Returns final loss (nan if it diverged)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    step = torch.compile(model.step)
    n_traj, n_t = traj.shape[:2]
    loss_value = float("nan")
    for _ in range(steps):
        i = torch.randint(n_traj, (batch,), generator=gen)
        t = torch.randint(n_t - unroll, (batch,), generator=gen)
        window = traj[i[:, None], t[:, None] + torch.arange(unroll + 1)].to(device, non_blocking=True)
        s = window[:, 0]
        ez, hx, hy = s[:, 0], s[:, 1], s[:, 2]
        loss = 0.0
        for u in range(1, unroll + 1):
            ez, hx, hy = step(ez, hx, hy, 1.0)
            target = window[:, u]
            pred = torch.stack([ez, hx, hy], dim=1)
            loss = loss + ((pred - target) ** 2).mean() / (target ** 2).mean()
        loss = loss / unroll
        if not torch.isfinite(loss):
            return float("nan")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_value = loss.item() if _ % 100 == 0 or _ == steps - 1 else loss_value
    return loss_value
