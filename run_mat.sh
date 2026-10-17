#!/bin/sh
# D2 stages 9-11 back to back. Logs in logs/.
set -e
[ -f runs/mat_baselines.json ] || python3 s9_mat_baselines.py > logs/s9.log 2>&1
python3 s10_mat_search.py random 40 > logs/s10_random.log 2>&1
python3 s10_mat_search.py evolution 40 > logs/s10_evolution.log 2>&1
python3 s11_mat_final.py > logs/s11.log 2>&1
