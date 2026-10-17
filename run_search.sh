#!/bin/sh
# Stages 3-4 back to back. Logs in logs/.
set -e
[ -f runs/baselines.json ] || python3 s3_baselines.py > logs/s3.log 2>&1
python3 s4_search.py random 48 > logs/s4_random.log 2>&1
python3 s4_search.py evolution 48 > logs/s4_evolution.log 2>&1
