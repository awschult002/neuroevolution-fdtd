#!/bin/sh
# after run_checks.sh: freeze the material bar (validation), then the unroll test
while pgrep -f run_checks.sh > /dev/null; do sleep 10; done
python3 s9_mat_baselines.py val > logs/s9.log 2>&1
python3 s10a_unroll_test.py > logs/s10a.log 2>&1
