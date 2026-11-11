#!/usr/bin/env bash
#SBATCH --job-name=part-existing-assets
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_EXISTING_TEST_RUN:?Set the prepared existing Test replay run}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MLSPACES_CACHE_DIR=/bigtemp/xna8aw/part-aware-policy-learning/cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/bigtemp/xna8aw/part-aware-policy-learning/cache/molmospaces/assets
cd /u/xna8aw/workspace/part-aware-policy-learning
exec /bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u \
  "$PART_EXISTING_TEST_RUN/source/prepare_existing_test_verification.py" stage --run "$PART_EXISTING_TEST_RUN"
