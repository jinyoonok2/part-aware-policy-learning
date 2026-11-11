#!/usr/bin/env bash
#SBATCH --job-name=part-pose-features
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4500
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
source /u/xna8aw/workspace/part-aware-policy-learning/env.sh
module load cuda/12.8.1 gcc/12.4.0
cd "${PART_POSE_SOURCE_ROOT:-$PART_POLICY_ROOT}"
: "${PART_POSE_CACHE:?Set a new output directory for frozen Train features}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
exec "$PART_PYTHON" -u -m pose_learning.prepare_features \
  --index "${PART_POSE_INDEX:-$PART_POLICY_STORE/datasets/interaction_pose_train_test_v1_20261107/train_early_pose_targets.jsonl}" \
  --role train \
  --encoder "$PART_POLICY_STORE/models/dinov2-base/f9e44c814b77203eaa57a6bdbbd535f21ede1415" \
  --output "$PART_POSE_CACHE" --device cuda "$@"
