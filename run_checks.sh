#!/bin/sh
# Gate checks before the material search resumes. Each step logs to logs/.
python3 s8c_reference_check.py > logs/s8c.log 2>&1
python3 s0b_material_conservation.py > logs/s0b.log 2>&1
python3 s0c_equivariance.py > logs/s0c.log 2>&1
python3 s3d_d4_stencil.py > logs/s3d.log 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,utilization.gpu,power.draw --format=csv -l 1 > logs/nvidia-smi_vacuum_timing.log 2>&1 &
S=$!
python3 s5b_vacuum_timing.py > logs/s5b.log 2>&1
kill $S
python3 s8b_masks_and_extra_test.py > logs/s8b.log 2>&1
