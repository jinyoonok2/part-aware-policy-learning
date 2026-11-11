#!/usr/bin/env bash
#SBATCH --job-name=part-pose-oracle
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
: "${PART_POSE_ORACLE_RUN:?Set the frozen expert-reference oracle comparison}"
cd "$PART_POSE_ORACLE_RUN/source"
"$PART_PYTHON" -u -m pose_learning.prepare_oracle_comparison verify --run "$PART_POSE_ORACLE_RUN"
checkpoint="$PART_POLICY_STORE/training/pose_20k_20261107_01/train/checkpoints/step20000.pt"
run_exit=0
"$PART_PYTHON" -u -m pose_learning.evaluate_policy \
  --episodes "$PART_POSE_ORACLE_RUN/episodes/policy_episode_manifest.json" \
  --output "$PART_POSE_ORACLE_RUN/policy" --pose-checkpoint "$checkpoint" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" --methods baseline predicted_pose oracle_pose \
  --seed 42 --horizon 400 --prediction-interval 8 --video || run_exit=$?
"$PART_PYTHON" -u -m pose_learning.report_policy_evaluation \
  --manifest "$PART_POSE_ORACLE_RUN/episodes/policy_episode_manifest.json" \
  --output "$PART_POSE_ORACLE_RUN/policy" --methods baseline predicted_pose oracle_pose
exit "$run_exit"
