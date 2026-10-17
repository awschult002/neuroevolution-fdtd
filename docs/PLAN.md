# Plan: learned coarse-grid stencil for 2D FDTD

## Goal
Find out whether a small learned correction lets a coarse 2D TMz Yee FDTD grid match a fine grid's accuracy for less compute, and whether automated search (random search, then evolution) beats a hand-designed update.

**Current direction (D2, decided by Alex 2026-10-17):** the network's job is scenes with materials (dielectric blocks and interfaces). See [Deviations](#deviations-from-plan).

## Experiment
- **Grids:** coarse 64² vs plain 256² Yee, both scored against a 512² reference.
- **Reference:** 512² truth generated once (fp64 allowed here only) and cached.
- **Baselines:** plain coarse Yee; best fixed linear energy-safe stencil at radius 1, 2, 3 (dispersion-optimized, after Tam & Webb). The hand-picked Bar-Sinai-style net is kept for the R1 record only.
- **Search order:** random search first, then evolution; PSO for continuous knobs; NEAT/HyperNEAT for networks that map the local material layout to stencil weights.
- **Budget:** about 2,000 training steps per candidate on 64²; a few hundred candidates per method. Expected total: a few GPU-hours.

## Scoring
- Error against the 512² reference on a held-out test set, with interface regions reported separately.
- Energy drift measured with the staggered leapfrog energy \(U_n=\tfrac12(\varepsilon\|E^n\|^2+\mu\,H^{n-\frac12}\cdot H^{n+\frac12})\) for every method.
- Phase error per wavelength on plane waves at 4, 8 and 16 coarse cells per wavelength.
- Cost measured as bytes read per cell per step (including any weight table) and time, not FLOPs. Network size counts toward setup time only.
- **Stability gate:** a candidate must survive 20,000 steps.
- **Win condition:** a network must beat the best fixed linear stencil, not just coarse Yee.

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
If no searched candidate beats the best fixed linear stencil on error at equal or lower measured cost, we stop and write up the negative result.

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
- 48 random designs plus the hand-picked Bar-Sinai net: **all failed the 1% energy gate.** None blew up, but all gain energy steadily (hand-picked net: +32.7% over 5,000 steps).
- Stencil-coefficient models beat coarse Yee on error at 1,000 steps (best 0.134 vs 0.240), but the lowest drift among them is 3%, and by 5,000 steps the gap mostly closes (0.62 vs 0.66).
- Direct-correction models drift 150–440% and are dropped.
- Plain 256² Yee is 7–30× more accurate than any coarse model.
- The evolution run was stopped: every fitness was infinite, so parent selection was random.
- **Pending:** the 3% best drift was measured with the plain energy formula and is being re-measured with the staggered energy. The 32.7% and 150–440% drifts grew steadily and stand either way.

## Deviations from plan

### D1 (approved 2026-10-17): energy conserved by structure
- The H update uses the exact transpose of the E-update operator (`conv_transpose2d` with the same kernel as `conv2d`), so the curl pair is skew-symmetric. No second kernel tied by a penalty.
- Coefficients must be fixed in time: constants, or functions of \(\varepsilon,\mu\) only. Field-dependent coefficients break the guarantee and may only appear as an unconstrained arm that passes the drift test empirically.
- Each candidate's stability limit is checked with an FFT of its stencil at the 4× time step (\(\Delta t\,\sigma_{\max} < 2\sqrt{\varepsilon\mu}\)); violations are rejected immediately.
- Long-wavelength consistency constraint \(\sum_j w_j\,(j+\tfrac12)=1/\Delta x\).
- Conservation unit test (T-STAB-4) runs before any search: fixed coefficients must stay at roundoff, field-dependent ones must drift.
- Fixed linear stencil baseline added at radius 1, 2, 3.

### D2 (decided by Alex 2026-10-17): move to materials
In vacuum, Maxwell is linear and uniform, so a fixed stencil is close to optimal and leaves nothing for a network to learn. After the vacuum stencil baseline, the search moves to scenes with dielectric blocks and interfaces. Networks map the local material layout (\(\varepsilon,\mu\) around each cell) to that cell's stencil weights. Materials don't change during a run, so the weights are fixed in time and D1's guarantee holds.
- The network runs once at setup, not per step.
- Per-cell weights are stored as a 1-byte class index into a small table of unique weight sets; bulk cells share the vacuum stencil.

## Claims and sources
- Learned discretizations for coarse grids: Bar-Sinai et al., *Learning data-driven discretizations for partial differential equations*, PNAS 2019, [arXiv:1808.04930](https://arxiv.org/abs/1808.04930).
- Learned solvers on coarse grids for fluids: Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, PNAS 2021, [arXiv:2102.01010](https://arxiv.org/abs/2102.01010).
- Dispersion-relation-preserving finite differences: Tam & Webb, JCP 1993, [doi:10.1006/jcph.1993.1142](https://doi.org/10.1006/jcph.1993.1142).
- NEAT: Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies*, Evolutionary Computation 2002.
- Random search as a strong NAS baseline: Li & Talwalkar, *Random Search and Reproducibility for NAS*, 2019, [arXiv:1902.07638](https://arxiv.org/abs/1902.07638).
- WaveY-Net (learned Maxwell solver): link still needed.
