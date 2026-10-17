#!/bin/sh
# Full remaining pipeline, detached. Each stage checkpoints its own outputs.
set -e
CUDA_VISIBLE_DEVICES= python3 s3f_dispersion_fit.py > logs/s3f.log 2>&1
python3 s9_mat_baselines.py val > logs/s9.log 2>&1                       # bar frozen (validation)
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,utilization.gpu,power.draw --format=csv -l 1 > logs/nvidia-smi_mat_timing.log 2>&1 &
S=$!
python3 s12_mat_timing.py > logs/s12.log 2>&1 || true
kill $S
[ -f runs/unroll_choice.json ] || python3 s10a_unroll_test.py > logs/s10a.log 2>&1
python3 s10_mat_search.py random 40 > logs/s10_random.log 2>&1
python3 s10_mat_search.py evolution 40 > logs/s10_evolution.log 2>&1
python3 s2b_vacuum_truth2048.py > logs/s2b.log 2>&1
python3 s3e_vacuum_rescore.py > logs/s3e.log 2>&1
if python3 -c "
import json,sys
bar=json.load(open('runs/mat_bar.json'))['val_ierr1000']
best=min([json.loads(l)['ierr1000'] for m in ('random','evolution') for l in open(f'runs/mat_search_{m}.jsonl') if json.loads(l)['feasible']] or [9])
sys.exit(0 if best < bar else 1)"; then
  python3 s11_mat_final.py > logs/s11.log 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,utilization.gpu,power.draw --format=csv -l 1 > logs/nvidia-smi_mat_timing2.log 2>&1 &
  S=$!
  python3 s12_mat_timing.py > logs/s12b.log 2>&1 || true
  kill $S
fi
python3 s9_mat_baselines.py test > logs/s9_test.log 2>&1                 # held-out baselines, once
echo PIPELINE_DONE
