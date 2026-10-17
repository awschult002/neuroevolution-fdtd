# Plan: learned coarse-grid stencil for 2D FDTD

## Goal
Find out whether a small learned correction lets a coarse 2D TMz Yee FDTD grid match a fine grid's accuracy for less compute, and whether automated search (random search, then evolution) beats a hand-designed update.

**Current direction (D2, decided by Alex 2026-10-17):** the network's job is scenes with materials (dielectric blocks and interfaces). See [Deviations](#deviations-from-plan).

## Experiment
- **Grids:** coarse learned/tuned stencil vs plain Yee, scored against a fine reference.
- **Reference:** 2048² truth for material scenes (512² failed T-MAT-1); vacuum truth moving to 2048² too. Generated once and cached.
- **Comparisons, in order:** (1) plain coarse Yee; (2) **frozen bar:** D4-symmetric radius-2 tuned stencil (6 weights) with cell-averaged \(\varepsilon\); (3) evolved networks mapping local material patches to curl and material weights. The hand-picked Bar-Sinai-style net is kept for the R1 record only.
- **Material scenes:** random dielectric blocks; the test set has unseen sizes and contrasts (one above the training range), a mid-cell edge, a one-cell-wide feature.
- **Search:** random search first, then evolution on the same budget and protocol; PSO for continuous knobs; NEAT for networks. An 8- vs 32-step training-unroll test runs on both arms first.
- **Budget:** a few hundred candidates per method; a few GPU-hours.

## Scoring
- Error against the reference on **validation** scenes during search; held-out test geometries run once, at the end.
- Error within two cells of block edges reported separately; the win criterion uses edge error.
- Energy drift measured with the staggered leapfrog energy \(U_n=\tfrac12(\varepsilon\|E^n\|^2+\mu\,H^{n-\frac12}\cdot H^{n+\frac12})\), weighted by the true \(\varepsilon,\mu\), for every method.
- Phase error per wavelength at 4, 8 and 16 coarse cells per wavelength, at 0°, 22.5° and 45°.
- Reflection and transmission off a dielectric slab at several angles, against the analytic Fresnel answer.
- Rotation/mirror test: rotating or mirroring a block scene must rotate or mirror the solution to fp32 roundoff.
- Cost measured as bytes read per cell per step (including the class table) and time. Network size counts toward setup time only.
- **Stability gate:** a candidate must survive 20,000 steps.
- **Win condition:** a network must beat the frozen bar (0.0962 edge error at 1,000 steps, validation).

The numbered gates (T-REF, T-MAT-1, T-STAB-1..4, T-TIME, T-CLAIM-1, T-KILL-1, ...) are defined in [TEST_SPEC.md](TEST_SPEC.md). Findings live in [FINDINGS.md](FINDINGS.md).

## Fairness and timing rules
- Timing grids are well beyond the GPU's ~1 MB L2.
- **Compare at matched error, not matched grid:** the tuned coarse stencil is timed against the Yee grid of similar accuracy; finer Yee is also reported.
- Everything is timed in fp32 unless every method switches precision together.
- Laptop GPU: plugged in, 30-second warm-up, GPU clock and temperature logged with `nvidia-smi` throughout.
- A run is invalid if the clock drops more than 10%; it's discarded, not averaged in.
- Report the median of 5 valid runs.
- GPU work runs in a few large batches, not one command per candidate.
- Timing waits for an idle GPU.

## Kill criterion
If no searched network beats the frozen bar at interfaces at equal or lower measured cost, we stop and write up the negative result.

## Environment (logged with every result)
| Item | Value |
|---|---|
| Machine | `abaddon` (Alex's laptop), Debian |
| GPU | NVIDIA Quadro T2000 (Turing), 4 GB, ~128 GB/s, ~3.5 TFLOPS fp32, no tensor cores, ~1 MB L2 |
| Driver / CUDA | 550 / 12.4 |
| Python / PyTorch | venv on system Python 3.14, PyTorch 2.11 (CUDA build) |
| Work dir | `~/fdtd-neuroevo` |

Each result also records GPU clock, temperature, power state and commit hash.

## Results log

### R1: random search vs hand-picked baseline (2026-10-17, `afc9bef`, finding F-001, closed)
- 48 random designs plus the hand-picked Bar-Sinai net all failed the 1% energy gate. Re-scored with the staggered energy (`c783341`): best drift 3.6%, others 4.5–20%, hand-picked net 32.7%.
- Direct-correction models drift 150–440% and are dropped. The evolution run was stopped (all fitness infinite).

### R2: vacuum tuned-stencil baseline (2026-10-17, `c783341`)
- Conservation test, 20,000 steps: Yee and a paired random fixed stencil drift ~2e-7; field-dependent weights grow ~21×. The naive energy formula wobbles 0.7–1.7% even for exact Yee, so the old 1% gate was invalid.
- Free radius-2 stencil (44 weights): error 0.098 at 1,000 steps vs 0.240 coarse Yee and 0.086 Yee 128². Superseded by the D4 stencil in R4.

### R3: first material search (obsolete, archived)
Scored against the 512² reference, which failed T-MAT-1. Best 0.113 vs bar 0.103 after 15 of 40 designs. Not comparable to later results.

### R4: milestone (2026-10-17, `60c8ade`)
- **Speed at matched error:** tuned stencil at 1024² has edge error 0.096 vs Yee 2048² at 0.126, and is about **7.5× faster** (0.74 s vs 5.57 s per unit simulated time, median of 5, thermals clean). Yee 4096² is 2.6× more accurate but 60× slower. *Caveat: timing is in vacuum; repeat on a material scene pending.*
- **Transpose at interfaces:** no bug. H step is a gather; adjoint test agrees to 2e-8. On a dielectric-block scene, 20,000 steps: correct version drifts 7e-8; deliberately broken version drifts 2% on the block and passes in vacuum, so the check works.
- **Reference:** 512² material reference differs from 1024² by 0.045 at edges (about half the models' error), failing T-MAT-1. Material truth is now 2048², scored at 100 and 1,000 steps.
- **D4 symmetry:** radius-2 stencil drops from 44 weights to 6 and gets slightly more accurate. Phase error at 4 cells/wavelength: 0.013 on-axis (free fit 0.22), 0.15 at 45°.
- **Frozen bar:** D4 r2 with cell-averaged \(\varepsilon\), **0.0962 edge error at 1,000 steps** (validation).
- **Kernel form:** fused shifted-sum kernels reach ~70 GB/s; the `conv2d` form is 13× slower.
- **Next:** vacuum truth to 2048² (~40 GPU-minutes); 8- vs 32-step unroll test on both arms; restart the materials search from zero against 0.0962.

## Deviations from plan

### D1 (approved 2026-10-17): energy conserved by structure
- The H update uses the exact transpose of the E-update operator, so the curl pair is skew-symmetric. No second kernel tied by a penalty.
- Coefficients must be fixed in time: constants, or functions of \(\varepsilon,\mu\) only. Field-dependent coefficients break the guarantee.
- Vacuum stability limit checked with an FFT of each stencil at the 4× time step; violations rejected immediately.
- Long-wavelength consistency constraint \(\sum_j w_j\,(j+\tfrac12)=1/\Delta x\).
- Conservation unit test (T-STAB-4) runs before any search.

### D2 (decided by Alex 2026-10-17): move to materials
In vacuum a fixed stencil is close to optimal, leaving nothing for a network to learn. The search moves to scenes with dielectric blocks and interfaces. Materials don't change during a run, so learned weights are fixed in time and the energy guarantee holds.
- **Energy-safe form:** \(M_\varepsilon\dot E = C H\), \(M_\mu\dot H = -C^{\mathsf T}E\), with \(M_\varepsilon, M_\mu\) symmetric positive-definite.
- **What the network may learn:** a spatially varying curl \(C\) (H step uses its exact transpose) and effective material weights (softplus diagonal, or \(LL^{\mathsf T}\) block). It may not output E and H updates separately.
- **True transpose at interfaces:** the H step is a **gather** that reads each neighbor's class (not a scatter-add, whose fp32 atomics aren't reproducible). T-STAB-4 runs on a dielectric scene, with a deliberately broken variant that must drift.
- **Stability check:** ~20 power iterations on \(M_\varepsilon^{-1/2}CM_\mu^{-1/2}\) per scene; reject when \(\Delta t\,\sigma_{\max}\ge 2\).
- **Setup and storage:** the network runs once at setup, once per unique material pattern; never cluster output weights.

### D3 (2026-10-17): grid symmetry
- Tuned stencils are constrained to the square grid's D4 symmetry (odd stencils computed in pairs as \(w_k(f_{+k}-f_{-k})\)).
- Material patterns are canonicalized under rotation and reflection before hashing. Each cell stores the class index plus the 3-bit symmetry \(g\) (fits in one byte with a 5-bit class, or alongside a 16-bit index).
- At setup the canonical weights are mapped back with \(g^{-1}\): offsets rotated/reflected, H components transformed as vectors (90°: \(H_x\to H_y\), \(H_y\to -H_x\); reflection flips the normal H component; a mirrored odd stencil flips sign). \(E_z\) is unchanged.
- Skipping the back-transform still passes the energy check, so T-STAB-4 uses an asymmetric (L-shaped) block and the rotation/mirror test catches it.

## Claims and sources
- Learned discretizations for coarse grids: Bar-Sinai et al., PNAS 2019, [arXiv:1808.04930](https://arxiv.org/abs/1808.04930).
- Learned solvers on coarse grids for fluids: Kochkov et al., PNAS 2021, [arXiv:2102.01010](https://arxiv.org/abs/2102.01010).
- Dispersion-relation-preserving finite differences: Tam & Webb, JCP 1993, [doi:10.1006/jcph.1993.1142](https://doi.org/10.1006/jcph.1993.1142).
- Subpixel smoothing at interfaces: Farjadpour et al., Optics Letters 2006, [doi:10.1364/OL.31.002972](https://doi.org/10.1364/OL.31.002972).
- NEAT: Stanley & Miikkulainen, Evolutionary Computation 2002.
- Random search as a strong NAS baseline: Li & Talwalkar, 2019, [arXiv:1902.07638](https://arxiv.org/abs/1902.07638).
- WaveY-Net (learned Maxwell solver): link still needed.
