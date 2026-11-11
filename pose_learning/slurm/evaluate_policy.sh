#!/usr/bin/env bash
#SBATCH --job-name=part-pose-eval
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100_40gb|a100_80gb|h100_94gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -euo pipefail
source /u/xna8aw/workspace/part-aware-policy-learning/env.sh
module load cuda/12.8.1 gcc/12.4.0
cd "${PART_POSE_SOURCE_ROOT:-$PART_POLICY_ROOT}"
: "${PART_POSE_EPISODES:?Set a materialized EpisodeSpec manifest}"
: "${PART_POSE_EVAL_RUN:?Set an evaluation output directory}"
: "${PART_POSE_CHECKPOINT:?Set the frozen learned-pose checkpoint}"
run_exit=0
"$PART_PYTHON" -u -m pose_learning.evaluate_policy \
  --episodes "$PART_POSE_EPISODES" --output "$PART_POSE_EVAL_RUN" \
  --pose-checkpoint "$PART_POSE_CHECKPOINT" --index "${PART_POSE_CASE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" \
  --methods baseline predicted_pose --seed 42 --horizon 400 --video "$@" || run_exit=$?
if [[ -n "${PART_POSE_SOURCE_INVENTORY:-}" ]]; then
  "$PART_PYTHON" -u -m pose_learning.report_policy_evaluation \
    --manifest "$PART_POSE_SOURCE_INVENTORY" --output "$PART_POSE_EVAL_RUN"
fi
exit "$run_exit"
