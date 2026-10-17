# Test and fairness spec

Owner: The Tester. No speed or accuracy claim from this project counts unless it passes every gate below. Each gate has an ID so results and reports can cite it.

**Scope (2026-10-17, Alex):** the search moves to scenes with dielectric blocks and interfaces. The vacuum tuned-stencil baseline still runs first. The T-MAT gates below apply to every material-scene result.

## Data and reference

- **T-REF-1** Truth comes from a 2D TMz Yee solve at 512², which is 2× finer than the 256² "fine" baseline. Both the 256² Yee run and every coarse model are scored against it, so no model gets credit for matching Yee's own dispersion error.
- **T-REF-2** All methods use one shared, documented cell-averaging operator that respects the Yee staggering (E_z at nodes, H at half-cell offsets). Naive striding is not allowed.
- **T-DATA-1** The split is 150 train, 25 validation, 25 held-out test. Validation is used for selection during the search. Test is touched once, after the search is frozen, and the result is reported as-is.
- **T-DATA-2** The held-out test also includes cases from outside the training distribution: pulses near the coarse grid's resolution limit (about 4 coarse cells per wavelength) and at least one dielectric-block case.

## Material scenes

- **T-MAT-1** Reference convergence: the 512² truth itself has staircase error at interfaces. Before it is used, check it against a 1024² solve on a few scenes. Interface-band error between 512² and 1024² must be well below the coarse models' error (target under one tenth), or the reference resolution goes up.
- **T-MAT-2** One shared, documented rule turns the fine ε map into the coarse one (for example harmonic or arithmetic averaging per coarse cell). Coarse Yee, the fixed stencil and every network see the same coarse ε map.
- **T-MAT-3** The held-out geometries are new: block shapes, sizes, positions and ε contrasts not in training, including at least one contrast above the training range, one block edge that falls mid-cell on the coarse grid, and one thin feature about 1 coarse cell wide.
- **T-MAT-4** Interface error is reported separately: relative L2 inside a band of ±2 coarse cells around every interface, next to the bulk error. A model wins at interfaces only if it beats the fixed stencil in that band.
- **T-MAT-5** Physical check on a flat interface: reflection and transmission coefficients for a plane wave at normal incidence, compared with Fresnel, for coarse Yee, the fixed stencil and the network.

## Accuracy and stability

- **T-ACC-1** Relative L2 field error against T-REF-1 at 100, 1,000 and 5,000 coarse steps. Comparisons are made at one fixed error target.
- **T-STAB-1** Each model runs 20,000 coarse steps with no NaN, Inf or blow-up. Any failure disqualifies the model.
- **T-STAB-2** Energy drift |U_n - U_0| / U_0 must stay under 1% at 5,000 steps, using the staggered leapfrog energy for every method: U_n = ½(∑ ε_i (E_i^n)² + ∑ μ_j H_j^{n-½} H_j^{n+½}), with per-cell ε and μ. It needs only the stored fields and materials, not the model's operator. The naive ½(‖E‖²+‖H‖²) is not allowed, because at 4× Δt it wobbles by about (Δt σ_max)²/8 and can fail exact Yee. Coarse Yee's drift on the same run is reported as the yardstick. *(Changed 2026-10-17, per The Oracle; R1 drift numbers in F-001 must be re-measured with this formula.)*
- **T-STAB-3** Stability bound, checked per candidate before training is scored: Δt · σ_max < 2√(εμ), where σ_max = max_k |Ĉ(k)| from an FFT of the stencil on the periodic grid, at the 4× coarse time step. For material scenes, σ_max is the largest singular value of the full ε,μ-weighted operator, estimated by power iteration, and the check uses the smallest εμ in the scene. A violation is an immediate reject, logged as such, never an infinite fitness.
- **T-STAB-4** Conservation unit test, run before any search: a paired stencil (H step uses the exact transpose of the E-step kernel) with random fixed coefficients runs 20,000 steps, and staggered-energy drift must stay at fp32 roundoff (order 1e-5). The same test with field-dependent coefficients must show drift. Repeat it with random per-cell weights from a random material map, which must also stay at roundoff. If any result goes the other way, the harness is wrong and the search does not run.
- **T-DISP-1** Dispersion check: single plane waves at 4, 8 and 16 coarse cells per wavelength, along the axis and at 45°. Report phase error per wavelength travelled, next to coarse Yee. Once energy is conserved by construction, this is the main way models fail.
- **T-SEED-1** Every model on the frontier is retrained with 3 seeds. Report the median and the spread, never the best seed.

## Cost and timing

- **T-COST-1** During the search, cost is the estimated FLOPs and bytes moved per cell update. The parameter count is reported but never scored.
- **T-COST-1M** For material scenes the network runs once at setup, so it is scored separately: per-step cost is the bytes read per cell per step including the stencil-weight storage (for example a 1-byte class index per cell plus the shared weight table), and the network's size counts only toward setup time, which is reported on its own.
- **T-COST-2** The estimate is checked against measurement: time the frontier plus about 5 random models from off the frontier. If the Spearman rank correlation between estimated and measured cost is below 0.8, the frontier is re-picked from measured times.
- **T-TIME-1** Timing uses grids far past L2 (the T2000's L2 is about 1 MB): 4096² fine against 1024² coarse. Every method is one fused kernel at the same precision (fp32).
- **T-TIME-2** For the laptop (`abaddon`, Quadro T2000): run on AC power, warm up for 30 s, log the SM clock and temperature from `nvidia-smi` for every run, and take the median of 5 runs. A run is invalid if its clock drops more than 10% from its warm-up clock, or if any other process is using the GPU.
- **T-TIME-3** Every result records the machine, GPU, driver, CUDA, PyTorch and Python versions, and the git commit.

## Claim and kill criteria

- **T-CLAIM-1** A searched model "wins" only if it is on the error-vs-measured-cost Pareto frontier above coarse Yee, the hand-picked Bar-Sinai-style baseline, and the best fixed linear paired stencil (radius 1 to 3, trained on the same data), after T-SEED-1. For material scenes, it must also beat the fixed stencil on T-MAT-4 interface error on the held-out geometries of T-MAT-3.
- **T-KILL-1** If no candidate beats fine Yee's wall-clock time at equal error, counting the coarse-grid advantage, the experiment stops before NEAT.

## Report

The findings report lists each gate with pass or fail, the raw numbers and the commit. Every failure gets a short note on what it taught us.
