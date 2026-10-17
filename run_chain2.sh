#!/bin/sh
# after run_checks.sh: 2048^2 material truth, freeze the bar (validation), unroll test
while pgrep -f run_checks.sh > /dev/null; do sleep 10; done
python3 s8d_truth2048.py > logs/s8d.log 2>&1
python3 s9_mat_baselines.py val > logs/s9.log 2>&1
python3 s10a_unroll_test.py > logs/s10a.log 2>&1
