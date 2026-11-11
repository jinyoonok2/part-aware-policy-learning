#!/usr/bin/env bash
#SBATCH --job-name=part-pose-test-errors
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4500
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -euo pipefail
source /u/xna8aw/workspace/part-aware-policy-learning/env.sh
module load cuda/12.8.1 gcc/12.4.0
: "${PART_POSE_SOURCE_ROOT:?Set the frozen evaluation package root}"
: "${PART_POSE_TEST_INDEX:?Set the frozen full Test pose index}"
: "${PART_POSE_TEST_CACHE:?Set a fresh Test feature directory}"
: "${PART_POSE_CHECKPOINT:?Set the fixed final learned-pose checkpoint}"
: "${PART_POSE_OFFLINE_REPORT:?Set a fresh offline report directory}"
cd "$PART_POSE_SOURCE_ROOT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$PART_PYTHON" -u -m pose_learning.prepare_features \
  --index "$PART_POSE_TEST_INDEX" --role test \
  --encoder "$PART_POLICY_STORE/models/dinov2-base/f9e44c814b77203eaa57a6bdbbd535f21ede1415" \
  --output "$PART_POSE_TEST_CACHE" --device cuda --verify-h5
exec "$PART_PYTHON" -u -m pose_learning.evaluate \
  --cache "$PART_POSE_TEST_CACHE" --checkpoint "$PART_POSE_CHECKPOINT" \
  --output "$PART_POSE_OFFLINE_REPORT" --role test --device cuda
