"""Stage 3f (Oracle): closed-form dispersion fit of the D4 r2 stencil, no rollouts.
D4 r2 has 6 weights p(dx, dy), dx in {0,1}, dy in {0,1,2}. Its Fourier symbol (grid units, h = 1):
  a1(kx, ky) = 2i * sum_p p(dx,dy) sin(kx (dx + 1/2)) * c(dy, ky),  c(0) = 1, c(dy>0) = 2 cos(ky dy)
  a2(kx, ky) = a1 rotated by 90 degrees (= a1(ky, kx) up to sign),  sigma(k) = sqrt(|a1|^2 + |a2|^2)
Leapfrog at Courant C = dt/h = 0.5:  sin(w dt / 2) = (C/2) sigma(k)  ->  w_num = (2/C) asin(C sigma / 2)
(in units where h = 1 and c = 1; exact w = |k|).
  L2 fit     : minimise  int_0^kmax int_0^pi/4 (w_num - |k|)^2 dtheta dk
  minimax fit: minimise  max_theta int_0^kmax (w_num - |k|)^2 dk
with kmax = 2 pi / 4 (4 cells per wavelength), subject to the consistency moment
sum p (2dx+1) m(dy) = 1 and the stability bound C * sigma_max < 2 (checked on a dense k grid).
Writes runs/d4_fit.json and runs/d4fit{l2,mm}_r2.pt (D4FixedStepper state dicts)."""
import json, math
import numpy as np
import torch
from scipy.optimize import minimize
from fdtd.model import D4FixedStepper

C = 0.5
PAIRS = [(dx, dy) for dx in range(2) for dy in range(3)]
MULT = np.array([(2 * dx + 1) * (1 if dy == 0 else 2) for dx, dy in PAIRS], dtype=float)
KMAX = 2 * math.pi / 4
NK, NT = 48, 24
kk = np.linspace(KMAX / NK, KMAX, NK)
tt = np.linspace(0, math.pi / 4, NT)
K, T = np.meshgrid(kk, tt, indexing="ij")
KX, KY = K * np.cos(T), K * np.sin(T)
g = np.linspace(-math.pi, math.pi, 97)
GX, GY = np.meshgrid(g, g, indexing="ij")


def a1(p, kx, ky):
    s = np.zeros_like(kx)
    for w, (dx, dy) in zip(p, PAIRS):
        s = s + w * np.sin(kx * (dx + 0.5)) * (1.0 if dy == 0 else 2 * np.cos(ky * dy))
    return 2 * s


def sigma(p, kx, ky):
    return np.sqrt(a1(p, kx, ky) ** 2 + a1(p, ky, kx) ** 2)


def omega(p, kx, ky):
    return (2 / C) * np.arcsin(np.clip(C * sigma(p, kx, ky) / 2, -1, 1))


def err_per_theta(p):
    return ((omega(p, KX, KY) - K) ** 2).mean(axis=0)          # (NT,)


def constraints():
    return [{"type": "eq", "fun": lambda p: MULT.dot(p[:6]) - 1.0},
            {"type": "ineq", "fun": lambda p: 1.98 - C * sigma(p[:6], GX, GY).max()}]


def fit(kind):
    p0 = np.zeros(6)
    p0[0] = 1.0                                                 # Yee
    if kind == "l2":
        res = minimize(lambda p: err_per_theta(p).mean(), p0, constraints=constraints(), method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-14})
        return res.x, res
    x0 = np.append(p0, err_per_theta(p0).max())
    cons = constraints() + [{"type": "ineq", "fun": lambda x: x[6] - err_per_theta(x[:6])}]
    res = minimize(lambda x: x[6], x0, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-16})
    return res.x[:6], res


def main():
    out = {}
    trained = D4FixedStepper(2)
    trained.load_state_dict(torch.load("runs/d4_r2.pt", map_location="cpu"))
    p_tr = (trained.p0 + trained.pproj @ trained.dp).detach().numpy().astype(float)
    p_yee = np.zeros(6); p_yee[0] = 1.0
    for name, p in (("yee", p_yee), ("d4_trained", p_tr)):
        e = err_per_theta(p)
        out[name] = {"p": p.tolist(), "l2": float(e.mean()), "worst_theta": float(e.max()),
                     "courant_sigma": float(C * sigma(p, GX, GY).max())}
    for kind in ("l2", "mm"):
        p, res = fit(kind)
        e = err_per_theta(p)
        out[f"d4fit{kind}"] = {"p": p.tolist(), "l2": float(e.mean()), "worst_theta": float(e.max()),
                               "courant_sigma": float(C * sigma(p, GX, GY).max()), "success": bool(res.success),
                               "message": str(res.message)}
        m = D4FixedStepper(2)
        with torch.no_grad():
            m.dp.copy_(torch.tensor(p, dtype=torch.float32) - m.p0)   # p satisfies the moment, so P(p - p0) = p - p0
        out[f"d4fit{kind}"]["courant_sigma_fft64"] = 0.5 * m.sigma_max()
        torch.save(m.state_dict(), f"runs/d4fit{kind}_r2.pt")
    for k, v in out.items():
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items() if a != "p"}, np.round(v["p"], 4).tolist())
    json.dump(out, open("runs/d4_fit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
