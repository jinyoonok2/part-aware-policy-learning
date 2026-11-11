#!/usr/bin/env bash
#SBATCH --job-name=early-pose-quality
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_EARLY_POSE_VIEW:?Set prepared early-pose quality-view directory}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u \
  "$PART_EARLY_POSE_VIEW/source/build_early_interaction_pose_view.py" build --run "$PART_EARLY_POSE_VIEW"
