# Plan: learned coarse-grid stencil for 2D FDTD

## Goal
Find out whether a small learned correction lets a coarse 2D TMz Yee FDTD grid match a fine grid's accuracy for less compute, and whether automated search (random search, then evolution) beats a hand-picked network.

## Experiment
- **Grids:** coarse 64² learned stencil vs plain 256² Yee, both scored against a 512² reference.
- **Reference:** 512² truth generated once (fp64 allowed here only) and cached.
- **Baseline:** hand-picked Bar-Sinai-style learned stencil network.
- **Search order:** random search first, then evolution; PSO for continuous knobs (learning rate, stencil width, etc.); NEAT/HyperNEAT last, only if earlier stages justify it.
- **Budget:** about 2,000 training steps per candidate on 64²; a few hundred candidates per method. Expected total: a few GPU-hours.
- **First deliverable:** random search vs the hand-picked baseline, with a Pareto plot (error vs cost).

## Scoring
- Error against the 512² reference on a held-out test set.
- Energy drift over the run.
- Cost measured as time and bytes moved per step, not FLOPs (the step is memory-bound).
- **Stability gate:** a candidate must survive 20,000 steps without blowing up to count.

The numbered gates (T-REF, T-STAB, T-TIME, T-KILL, ...) are defined in [TEST_SPEC.md](TEST_SPEC.md).

## Fairness and timing rules
- Timing grids are 4096² fine vs 1024² coarse, well beyond the GPU's ~1 MB L2, so cache effects don't flatter small grids.
- Everything is timed in fp32 unless every method switches precision together.
- Laptop GPU: plugged in, 30-second warm-up, GPU clock and temperature logged with `nvidia-smi` throughout.
- A run is invalid if the clock drops more than 10%; it's discarded, not averaged in.
- Report the median of 5 valid runs.
- GPU work runs in a few large batches (about six), not one command per candidate.

## Kill criterion
If no searched candidate beats plain 256² Yee on error at lower measured cost, and none beats the hand-picked baseline, we stop and write up the negative result.

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

### R1: random search vs hand-picked baseline (2026-10-17, harness `afc9bef`)
- 48 random designs plus the hand-picked Bar-Sinai net: **all failed the 1% energy gate.** None blew up, but all gain energy steadily (hand-picked net: +32.7% over 5,000 steps).
- Stencil-coefficient models beat coarse Yee on error at 1,000 steps (best 0.134 vs 0.240), but the lowest drift among them is 3%, and by 5,000 steps the gap mostly closes (0.62 vs 0.66).
- Direct-correction models drift 150–440% and are dropped.
- Plain 256² Yee is 7–30× more accurate than any coarse model. **Kill criterion currently points to no win.**
- The evolution run is not a valid comparison yet: every fitness is infinite, so parent selection is random.
- Timing not run yet: the GPU was shared with `bitfeat` jobs.

## Deviations from plan

### D1 (proposed, awaiting Oracle/Senior review and Alex): structural energy conservation
Tie each learned E-update stencil to the negative adjoint of its H-update partner so the discrete curl pair stays skew-symmetric, conserving the shared discrete energy exactly as Yee does. Then rerun random search vs evolution on the same budget (~2 GPU-hours). Alternative: write up R1 as the negative result.

## Claims and sources
- Learned discretizations for coarse grids: Bar-Sinai et al., *Learning data-driven discretizations for partial differential equations*, PNAS 2019, [arXiv:1808.04930](https://arxiv.org/abs/1808.04930).
- Learned solvers on coarse grids for fluids: Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, PNAS 2021, [arXiv:2102.01010](https://arxiv.org/abs/2102.01010).
- NEAT: Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies*, Evolutionary Computation 2002.
- Random search as a strong NAS baseline: Li & Talwalkar, *Random Search and Reproducibility for NAS*, 2019, [arXiv:1902.07638](https://arxiv.org/abs/1902.07638).
- WaveY-Net (learned Maxwell solver): link still needed.
