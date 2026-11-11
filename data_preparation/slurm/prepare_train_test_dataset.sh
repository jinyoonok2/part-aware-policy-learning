#!/usr/bin/env bash
#SBATCH --job-name=part-train-test-data
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_TRAIN_TEST_DATASET:?Set the prepared official Train / Val-as-Test dataset directory}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="${1:-collect}"
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u \
  "$PART_TRAIN_TEST_DATASET/source/prepare_train_test_dataset.py" "$mode" --run "$PART_TRAIN_TEST_DATASET"
