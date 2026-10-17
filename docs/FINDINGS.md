# Findings

Owner: The Tester. Each finding cites gates from `docs/TEST_SPEC.md`. Findings are never edited after the fact; corrections get a new entry.

## F-001: Random search fails the energy gate (2026-10-17, harness `afc9bef`)

Source: The Dev's first run on `abaddon` (Quadro T2000), as reported in the group chat. Raw logs are on the laptop and not in git.

| Gate | Result |
|---|---|
| T-STAB-1 (no blow-up, 20k steps) | pass for all 48 random designs and the hand-picked net |
| T-STAB-2 (energy drift < 1% at 5k) | **fail for all 49.** The best stencil-coefficient model drifts 3%; the hand-picked Bar-Sinai-style net drifts 32.7%; direct-correction models drift 150 to 440%. Every model gains energy steadily. |
| T-ACC-1 at 1k steps | Stencil-coefficient models beat coarse Yee (best 0.134 vs 0.240 relative L2) |
| T-ACC-1 at 5k steps | Advantage mostly gone (0.62 vs 0.66) |
| T-KILL-1 | Points to no win so far: 256² Yee is 7 to 30× more accurate than any coarse model. Not final, because wall-clock timing hasn't been run yet (GPU shared with other jobs, so T-TIME-2 forbids timing now). |
| Evolution run | Not a valid comparison: every fitness was infinite, so parent selection was random. Stopped. |

**What we learned.** A learned stencil trained on short (8-step) rollouts does not discover energy conservation by itself, and its 1,000-step accuracy edge is eaten by drift by 5,000 steps. Short-horizon accuracy is a poor proxy for long-horizon fitness in this problem.

**Next (plan deviation, recorded before the rerun).** Make conservation structural by tying each learned E-update stencil to the negative adjoint of its H-update partner, then rerun random search vs evolution on the same budget. Gate changes are not allowed: T-STAB-2 stays at 1% and is still measured with the shared energy formula, not the model's own conserved quantity.
