"""Stage 10a: does a longer training unroll fix the 1000-step regression? Two designs, unroll 8 vs 32,
same number of optimiser steps (500), validation edge-band err@1000 vs the frozen bar.
Writes runs/unroll_test.json."""
import json, time, torch
import fdtd.mateval as E
from fdtd.matmodel import MaterialNet, MatDesign


def main():
    train = torch.load("data/mat_train.pt", weights_only=False)
    train["traj"] = train["traj"].pin_memory()
    val = E.load_mat("val")
    bar = json.load(open("runs/mat_bar.json"))
    out = {"bar": bar}
    for d in (MatDesign(1, 2, 16, "tanh", 2, "CM"), MatDesign(2, 2, 16, "silu", 2, "C")):
        for unroll in (8, 32):
            torch._dynamo.reset()
            t0 = time.time()
            net = MaterialNet(d, E.base_kernel(d.stencil_radius)).cuda()
            loss = E.train_net(net, train["traj"], train["eps"], train["mask"], unroll=unroll)
            sc = E.score(E.bind(net.eval(), val["eps"]), val)
            key = f"{d.name()}_u{unroll}"
            out[key] = dict(sc, final_loss=loss, wall_s=time.time() - t0)
            print(key, {k: round(v, 4) for k, v in sc.items() if k[:4] in ("err1", "ierr", "berr")}, round(time.time() - t0), "s", flush=True)
            json.dump(out, open("runs/unroll_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
