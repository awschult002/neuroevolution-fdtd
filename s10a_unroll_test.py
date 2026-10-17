"""Stage 10a: training unroll 8 vs 32 (same 500 optimiser steps, same everything else) on three
representative material designs; validation edge-band err@100/1000 vs the frozen bar. The unroll with
the lower mean validation edge err@1000 is written to runs/unroll_choice.json and then used identically
by both search arms (logged deviation). Writes runs/unroll_test.json."""
import json, time, statistics, torch
import fdtd.mateval as E
from fdtd.matmodel import MaterialNet, MatDesign

DESIGNS = (MatDesign(1, 2, 16, "tanh", 2, "CM"), MatDesign(2, 2, 16, "silu", 2, "C"), MatDesign(1, 1, 8, "relu", 1, "M"))


def main():
    train = torch.load("data/mat_train.pt", weights_only=False)
    train["traj"] = train["traj"].pin_memory()
    val = E.load_mat("val")
    out = {"bar": json.load(open("runs/mat_bar.json"))["val_ierr1000"]}
    for d in DESIGNS:
        for unroll in (8, 32):
            torch._dynamo.reset()
            t0 = time.time()
            net = MaterialNet(d, E.base_kernel(d.stencil_radius)).cuda()
            loss = E.train_net(net, train["traj"], train["eps"], train["mask"], unroll=unroll)
            sc = E.score(E.bind(net.eval(), val["eps"]), val)
            out[f"{d.name()}_u{unroll}"] = dict(sc, final_loss=loss, wall_s=time.time() - t0)
            print(d.name(), unroll, {k: round(v, 4) for k, v in sc.items() if k[:4] in ("err1", "ierr", "berr")},
                  f"drift {sc['drift_mean']:.1e}", round(time.time() - t0), "s", flush=True)
            json.dump(out, open("runs/unroll_test.json", "w"), indent=1)
    mean = {u: statistics.mean(out[f"{d.name()}_u{u}"]["ierr1000"] for d in DESIGNS) for u in (8, 32)}
    choice = min(mean, key=mean.get)
    json.dump({"unroll": choice, "mean_val_ierr1000": mean}, open("runs/unroll_choice.json", "w"), indent=1)
    print("UNROLL", choice, mean, flush=True)


if __name__ == "__main__":
    main()
