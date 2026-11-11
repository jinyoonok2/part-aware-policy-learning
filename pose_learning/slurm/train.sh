#!/usr/bin/env bash
#SBATCH --job-name=part-pose-train20k
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4500
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
source /u/xna8aw/workspace/part-aware-policy-learning/env.sh
module load cuda/12.8.1 gcc/12.4.0
cd "${PART_POSE_SOURCE_ROOT:-$PART_POLICY_ROOT}"
: "${PART_POSE_CACHE:?Set a completed Train feature cache}"
: "${PART_POSE_TRAIN_RUN:?Set a new pose-training output directory}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if [[ -n "${PART_POSE_GPU_PREFLIGHT:-}" ]]; then
  "$PART_PYTHON" -u -m pose_learning.train \
    --cache "$PART_POSE_CACHE" --output "$PART_POSE_GPU_PREFLIGHT" \
    --device cuda --updates 4 --microbatch 1 --effective-batch 4 \
    --precision bf16 --seed 6198 --warmup 0 \
    --checkpoint-every 4 --diagnostic-every 4
fi
exec "$PART_PYTHON" -u -m pose_learning.train \
  --cache "$PART_POSE_CACHE" --output "$PART_POSE_TRAIN_RUN" \
  --updates 20000 --microbatch 1 --effective-batch 4 \
  --precision bf16 --seed 6198 --checkpoint-every 1000 --diagnostic-every 500 "$@"
