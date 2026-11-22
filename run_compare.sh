#!/usr/bin/env bash
# Trains every configs/compare/*.yaml one after the other, then fits per-task temperatures on the
# calibration split for each best checkpoint. Results: runs/cmp-<name>-<timestamp>/
# Compare them in compare_models.ipynb. Usage: bash run_compare.sh [config names...]
set -euo pipefail
cd "$(dirname "$0")"
names=("${@:-deberta_cls deberta_mean qwen_bev}")
for name in ${names[@]}; do
  run_name="cmp-${name//_/-}"
  echo "===== $run_name ($(date +%H:%M))"
  uv run python train.py "configs/compare/$name.yaml" --name "$run_name"
  run_dir=$(ls -td runs/"$run_name"-* | head -1)
  uv run python calibration.py "$run_dir/checkpoints/best" --max_questions 2000
done
echo "===== done ($(date +%H:%M))"
