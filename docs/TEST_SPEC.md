# Test and fairness spec

Owner: The Tester. No speed or accuracy claim from this project counts unless it passes every gate below. Each gate has an ID so results and reports can cite it.

## Data and reference

- **T-REF-1** Truth comes from a 2D TMz Yee solve at 512², which is 2× finer than the 256² "fine" baseline. Both the 256² Yee run and every coarse model are scored against it, so no model gets credit for matching Yee's own dispersion error.
- **T-REF-2** All methods use one shared, documented cell-averaging operator that respects the Yee staggering (E_z at nodes, H at half-cell offsets). Naive striding is not allowed.
- **T-DATA-1** The split is 150 train, 25 validation, 25 held-out test. Validation is used for selection during the search. Test is touched once, after the search is frozen, and the result is reported as-is.
- **T-DATA-2** The held-out test also includes cases from outside the training distribution: pulses near the coarse grid's resolution limit (about 4 coarse cells per wavelength) and at least one dielectric-block case.

## Accuracy and stability

- **T-ACC-1** Relative L2 field error against T-REF-1 at 100, 1,000 and 5,000 coarse steps. Comparisons are made at one fixed error target.
- **T-STAB-1** Each model runs 20,000 coarse steps with no NaN, Inf or blow-up. Any failure disqualifies the model.
- **T-STAB-2** Energy drift |U(t)-U(0)|/U(0) is computed with one discrete energy formula for all methods and must stay under 1% at 5,000 steps. Coarse Yee's drift on the same run is reported next to it as the yardstick.
- **T-SEED-1** Every model on the frontier is retrained with 3 seeds. Report the median and the spread, never the best seed.

## Cost and timing

- **T-COST-1** During the search, cost is the estimated FLOPs and bytes moved per cell update. The parameter count is reported but never scored.
- **T-COST-2** The estimate is checked against measurement: time the frontier plus about 5 random models from off the frontier. If the Spearman rank correlation between estimated and measured cost is below 0.8, the frontier is re-picked from measured times.
- **T-TIME-1** Timing uses grids far past L2 (the T2000's L2 is about 1 MB): 4096² fine against 1024² coarse. Every method is one fused kernel at the same precision (fp32).
- **T-TIME-2** For the laptop (`abaddon`, Quadro T2000): run on AC power, warm up for 30 s, log the SM clock and temperature from `nvidia-smi` for every run, and take the median of 5 runs. A run is invalid if its clock drops more than 10% from its warm-up clock.
- **T-TIME-3** Every result records the machine, GPU, driver, CUDA, PyTorch and Python versions, and the git commit.

## Claim and kill criteria

- **T-CLAIM-1** A searched model "wins" only if it is on the error-vs-measured-cost Pareto frontier above both coarse Yee and the hand-picked Bar-Sinai-style baseline, after T-SEED-1.
- **T-KILL-1** If no candidate beats fine Yee's wall-clock time at equal error, counting the coarse-grid advantage, the experiment stops before NEAT.

## Report

The findings report lists each gate with pass or fail, the raw numbers and the commit. Every failure gets a short note on what it taught us.
