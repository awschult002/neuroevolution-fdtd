# neuroevolution-fdtd
Searching small learned coarse-grid stencils for 2D FDTD (random search vs evolution vs hand-picked).

Question: can random search or a small evolutionary search find a learned 64^2 stepper for 2D TMz
Maxwell that beats a hand-picked Bar-Sinai-style stencil, and plain Yee, on error vs cost?
Plan and fairness rules: `docs/PLAN.md`, `docs/TEST_SPEC.md`. Results: `results/REPORT.md`.

## Code
| file | what |
|---|---|
| `fdtd/yee.py` | Yee TMz step (periodic, fp32, Courant 0.5) and the one energy formula |
| `fdtd/restrict.py` | the one staggering-aware restriction (fine -> coarse) used for every method |
| `fdtd/fine.py` | run a fine Yee grid, record restricted states at coarse step numbers |
| `fdtd/initial.py` | initial conditions: Gaussian pulses, lambda=4-cell wave packets, dielectric block |
| `fdtd/model.py` | search space, `YeeStepper`, `LearnedStepper` (stencil-coefficient or direct-correction output) |
| `fdtd/train.py` | training on 8-step unrolled rollouts |
| `fdtd/rollout.py` | coarse rollouts, relative L2 error, energy drift |
| `fdtd/evaluate.py` | train-and-score one design; feasibility rules (no blow-up, drift < 1%) |
| `fdtd/cost.py` | explicit FLOPs / bytes per cell estimator and roofline time |
| `fdtd/timing.py` | compiled-step timing with warm-up, clock logging, invalid-run rejection |
| `fdtd/pareto.py` | Pareto front |

## Stages (run from the repo root, GPU needed; logs go to `logs/`, big files to `data/` and `runs/`, both git-ignored)
1. `python3 s1_check_yee.py` - Yee convergence vs 1024^2 and energy sanity (seconds).
2. `python3 s2_make_data.py` - 200 ICs, 512^2 truth + 128^2/256^2 Yee, restricted to 64^2 (~30 min).
3. `python3 s3_baselines.py` - coarse/fine Yee scores and the hand-picked learned stencil.
4. `python3 s4_search.py random 48` then `python3 s4_search.py evolution 48` (`run_search.sh` does 3+4).
5. `python3 s5_timing.py` - measured cost of every feasible design and of Yee at 1024/2048/4096^2.
6. `python3 s6_final.py [est|measured]` - front, 3-seed retrain, 20k-step stability, held-out test.
7. `python3 s7_report.py` - `results/pareto.png`, `results/stability.png`, `results/tables.md`, `results/summary.json`.

Long stages are meant to be started with `nohup ... > logs/x.log 2>&1 &`.
