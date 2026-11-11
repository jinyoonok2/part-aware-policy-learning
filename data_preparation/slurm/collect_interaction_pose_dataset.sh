#!/usr/bin/env bash
#SBATCH --job-name=part-pose-data
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
: "${PART_POSE_DATA_RUN:?Set the prepared interaction-pose dataset directory}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u \
  "$PART_POSE_DATA_RUN/source/collect_interaction_pose_dataset.py" --run "$PART_POSE_DATA_RUN"
