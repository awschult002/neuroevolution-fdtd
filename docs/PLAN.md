# Plan: learned coarse-grid stencil for 2D FDTD

## Goal
Find out whether a small learned correction lets a coarse 2D TMz Yee FDTD grid match a fine grid's accuracy for less compute, and whether automated search (random search, then evolution) beats a hand-designed update.

**Current direction (D2, decided by Alex 2026-10-17):** the network's job is scenes with materials (dielectric blocks and interfaces). See [Deviations](#deviations-from-plan).

## Experiment
- **Grids:** coarse 64² vs plain 256² Yee, both scored against a 512² reference (512² is checked against 1024² near block edges first; raised if not clearly better than the models).
- **Reference:** truth generated once (fp64 allowed here only) and cached.
- **Comparisons, in order:** (1) plain coarse Yee; (2) best fixed linear energy-safe stencil at radius 1, 2, 3 (dispersion-optimized, after Tam & Webb), with cell-averaged \(\varepsilon\) (subpixel smoothing) in material scenes; (3) evolved networks mapping local material patches to curl and material weights. The hand-picked Bar-Sinai-style net is kept for the R1 record only.
- **Material scenes:** random dielectric blocks; the test set has unseen sizes and contrasts (one above the training range), a mid-cell edge, a one-cell-wide feature.
- **Search:** random search first, then evolution on the same budget; PSO for continuous knobs; NEAT for networks.
- **Budget:** about 2,000 training steps per candidate on 64²; a few hundred candidates per method. Expected total: a few GPU-hours.

## Scoring
- Error against the reference on a held-out test set; error within two cells of block edges reported separately.
- Energy drift measured with the staggered leapfrog energy \(U_n=\tfrac12(\varepsilon\|E^n\|^2+\mu\,H^{n-\frac12}\cdot H^{n+\frac12})\), weighted by the true \(\varepsilon,\mu\), for every method.
- Phase error per wavelength on plane waves at 4, 8 and 16 coarse cells per wavelength.
- Reflection and transmission off a dielectric slab at several angles, against the analytic Fresnel answer.
- Cost measured as bytes read per cell per step (including the weight table) and time, not FLOPs. Network size counts toward setup time only.
- **Stability gate:** a candidate must survive 20,000 steps.
- **Win condition:** a network must beat comparison (2) at interfaces.

The numbered gates (T-REF, T-STAB-1..4, T-TIME, T-KILL, ...) are defined in [TEST_SPEC.md](TEST_SPEC.md). Findings live in [FINDINGS.md](FINDINGS.md).

## Fairness and timing rules
- Timing grids are 4096² fine vs 1024² coarse, well beyond the GPU's ~1 MB L2.
- Everything is timed in fp32 unless every method switches precision together.
- Laptop GPU: plugged in, 30-second warm-up, GPU clock and temperature logged with `nvidia-smi` throughout.
- A run is invalid if the clock drops more than 10%; it's discarded, not averaged in.
- Report the median of 5 valid runs.
- GPU work runs in a few large batches, not one command per candidate.
- Timing waits for an idle GPU (no other jobs sharing it).

## Kill criterion
If no searched network beats comparison (2) at interfaces at equal or lower measured cost, we stop and write up the negative result.

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

### R1: random search vs hand-picked baseline (2026-10-17, harness `afc9bef`, finding F-001)
- 48 random designs plus the hand-picked Bar-Sinai net: **all failed the 1% energy gate.** None blew up, but all gain energy steadily.
- Re-scored with the staggered energy (`c783341`): best drift 3.6% (read 3% with the naive formula), others 4.5–20%, hand-picked net 32.7%. **F-001 stands.**
- Stencil-coefficient models beat coarse Yee on error at 1,000 steps (best 0.134 vs 0.240); by 5,000 steps the gap mostly closes (0.62 vs 0.66).
- Direct-correction models drift 150–440% and are dropped.
- The evolution run was stopped: every fitness was infinite, so parent selection was random.

### R2: vacuum tuned-stencil baseline (2026-10-17, `c783341`)
- **Conservation test (T-STAB-4), 20,000 steps:** Yee and a paired stencil with random fixed weights drift ~2e-7; field-dependent weights grow ~21×. With the old naive energy formula even exact Yee wobbles 0.7–1.7%, so the old 1% gate was invalid.
- **Tuned radius-2 stencil (44 weights):** error 0.098 at 1,000 steps vs 0.240 coarse Yee and 0.086 Yee 128²; 0.410 at 5,000 steps. Phase error at 4 cells/wavelength 0.22 vs 0.50 rad. Beats every R1 network. Yee 256² is still ~5× more accurate.
- **Timing:** in progress.

### R3: material scenes (in progress)
- Bar: tuned stencil + cell-averaged \(\varepsilon\), interface error 0.103 at 1,000 steps (coarse Yee 0.323).
- Random search: 15 of 40 designs done, best 0.113. None beats the bar yet. Evolution follows on the same budget.

## Deviations from plan

### D1 (approved 2026-10-17): energy conserved by structure
- The H update uses the exact transpose of the E-update operator (`conv_transpose2d` with the same kernel as `conv2d`), so the curl pair is skew-symmetric. No second kernel tied by a penalty.
- Coefficients must be fixed in time: constants, or functions of \(\varepsilon,\mu\) only. Field-dependent coefficients break the guarantee and may only appear as an unconstrained arm that passes the drift test empirically.
- Vacuum: each candidate's stability limit is checked with an FFT of its stencil at the 4× time step (\(\Delta t\,\sigma_{\max} < 2\sqrt{\varepsilon\mu}\)); violations are rejected immediately.
- Long-wavelength consistency constraint \(\sum_j w_j\,(j+\tfrac12)=1/\Delta x\).
- Conservation unit test (T-STAB-4) runs before any search: fixed coefficients must stay at roundoff, field-dependent ones must drift.
- Fixed linear stencil baseline added at radius 1, 2, 3.

### D2 (decided by Alex 2026-10-17): move to materials
In vacuum, Maxwell is linear and uniform, so a fixed stencil is close to optimal and leaves nothing for a network to learn. After the vacuum stencil baseline, the search moves to scenes with dielectric blocks and interfaces. Materials don't change during a run, so learned weights are fixed in time and the energy guarantee holds.
- **Energy-safe form:** \(M_\varepsilon\dot E = C H\), \(M_\mu\dot H = -C^{\mathsf T}E\), with \(M_\varepsilon, M_\mu\) symmetric positive-definite, conserves \(U=\tfrac12(E^{\mathsf T}M_\varepsilon E+H^{\mathsf T}M_\mu H)\) exactly.
- **What the network may learn:** a spatially varying curl \(C\) (H step always uses its exact transpose), and effective material weights (diagonal positive via softplus, or a small symmetric block as \(LL^{\mathsf T}\)). It may **not** output the E and H updates separately.
- **True transpose at interfaces:** the E step gathers with each cell's own class; the H step must scatter-add the same weights (or gather reading each neighbor's class). Applying a cell's own stencil to both updates is only correct where stencils are uniform and silently leaks energy at interfaces. T-STAB-4 runs on a dielectric-block scene as well as vacuum.
- **Stability check:** FFT no longer applies; run ~20 power iterations on \(M_\varepsilon^{-1/2}CM_\mu^{-1/2}\) per scene and require \(\Delta t\,\sigma_{\max}<2\). Once per scene.
- **Setup and storage:** the network runs once at setup. Hash each cell's local material neighborhood (\(\varepsilon,\mu\) within the stencil radius), run the network once per unique pattern, store a 1-byte pattern index per cell (16-bit if a scene exceeds 255 patterns). Never cluster the output weights; that would change what the network computed.

## Claims and sources
- Learned discretizations for coarse grids: Bar-Sinai et al., *Learning data-driven discretizations for partial differential equations*, PNAS 2019, [arXiv:1808.04930](https://arxiv.org/abs/1808.04930).
- Learned solvers on coarse grids for fluids: Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, PNAS 2021, [arXiv:2102.01010](https://arxiv.org/abs/2102.01010).
- Dispersion-relation-preserving finite differences: Tam & Webb, JCP 1993, [doi:10.1006/jcph.1993.1142](https://doi.org/10.1006/jcph.1993.1142).
- Subpixel smoothing at interfaces: Farjadpour et al., Optics Letters 2006, [doi:10.1364/OL.31.002972](https://doi.org/10.1364/OL.31.002972).
- NEAT: Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies*, Evolutionary Computation 2002.
- Random search as a strong NAS baseline: Li & Talwalkar, *Random Search and Reproducibility for NAS*, 2019, [arXiv:1902.07638](https://arxiv.org/abs/1902.07638).
- WaveY-Net (learned Maxwell solver): link still needed.
