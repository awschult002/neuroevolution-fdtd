"""Stage 8c (T-MAT-1): is the 512^2 truth good enough at interfaces? 4 validation scenes are solved at
1024^2 and 512^2 to 1000 coarse steps, both restricted to 64^2 with the shared operator. Reported:
rel. L2 difference 512 vs 1024 in the +-2-cell edge band and in the bulk, at 100 and 1000 steps,
next to the coarse fixed-r2 error. Target: reference error < 1/10 of the coarse models' edge error.
Writes runs/reference_check.json."""
import json, torch
from fdtd.scenes import sample_scene, average, interface_mask
from fdtd.fine import run_fine_restricted
from fdtd.mateval import load_mat, rel_l2

STEPS = [0, 100, 1000]


def main():
    val = load_mat("val")
    scenes = val["scenes"][:4]
    out = {}
    res = {}
    for n in (1024, 512):
        ez = torch.stack([sample_scene(s, n, "cuda")[0] for s in scenes])
        eps = torch.stack([sample_scene(s, n, "cuda")[1] for s in scenes])
        res[n] = run_fine_restricted(ez, 1.0 / eps, n // 64, STEPS).cuda()
        if n == 1024:
            mask = interface_mask(eps, 16, band=2)       # +-2 coarse cells, from the 1024^2 eps
    torch.save({"res512": res[512].cpu(), "res1024": res[1024].cpu(), "mask": mask.cpu()}, "runs/reference_check.pt")
    for k, s in enumerate(STEPS[1:], start=1):
        out[f"step{s}"] = {"edge": rel_l2(res[512][:, k], res[1024][:, k], mask),
                           "bulk": rel_l2(res[512][:, k], res[1024][:, k], ~mask),
                           "all": rel_l2(res[512][:, k], res[1024][:, k])}
    print(json.dumps(out, indent=1))
    json.dump(out, open("runs/reference_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
