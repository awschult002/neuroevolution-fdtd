"""Stage 5c: achieved bandwidth of every vacuum timed kernel (runs/vacuum_timing.json). Ideal traffic
is 24 bytes/cell/step (Ez, Hx, Hy read + written once, fp32) for Yee and every fixed stencil, fused
or conv2d (weights are a handful of constants). GB/s = 24 * n^2 * steps_per_unit / s_per_unit,
steps_per_unit = n / COURANT. % of the T2000's ~128 GB/s. Also the conv2d / fused gap per radius.
Writes runs/vacuum_gbs.json."""
import json
from fdtd.yee import COURANT

PEAK_GBS, BYTES = 128.0, 24


def main():
    t = json.load(open("runs/vacuum_timing.json"))
    out = {}
    for k, v in t.items():
        if k.startswith("_"):
            continue
        n = v["timed_at"]
        s = v["s_per_unit"] * (n / v["n"]) ** 3 if n != v["n"] else v["s_per_unit"]
        gbs = BYTES * n * n * (n / COURANT) / s / 1e9
        out[k] = {"n": n, "s_per_unit": s, "gb_s": gbs, "pct_peak": 100 * gbs / PEAK_GBS}
        print(f"{k:16s} {n:5d} {s:8.3f} s/unit {gbs:6.1f} GB/s {100 * gbs / PEAK_GBS:5.1f}%")
    for r in (1, 2, 3):
        out[f"conv_over_fused_r{r}"] = t[f"fixed_r{r}_conv"]["s_per_unit"] / t[f"fixed_r{r}_fused"]["s_per_unit"]
        print(f"conv2d/fused r{r}: {out[f'conv_over_fused_r{r}']:.1f}x")
    json.dump(out, open("runs/vacuum_gbs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
