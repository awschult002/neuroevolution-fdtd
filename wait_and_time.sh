#!/bin/sh
# Material-scene timing (s12) must run on an idle GPU (T-TIME-2); wait for run_final.sh and any other GPU user.
while pgrep -f run_final.sh > /dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,utilization.gpu,power.draw --format=csv -l 1 > logs/nvidia-smi_mat_timing.log 2>&1 &
S=$!
python3 s12_mat_timing.py > logs/s12.log 2>&1
kill $S
echo TIMING_DONE
