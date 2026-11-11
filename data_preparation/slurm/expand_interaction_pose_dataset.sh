#!/usr/bin/env bash
#SBATCH --job-name=part-pose-data-5000
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_POSE_EXPANSION_RUN:?Set the prepared expansion dataset directory}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u \
  "$PART_POSE_EXPANSION_RUN/source/expand_interaction_pose_dataset.py" collect --run "$PART_POSE_EXPANSION_RUN"
