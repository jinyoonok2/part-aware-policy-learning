#!/usr/bin/env bash
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_BENCHMARK_EXPERT_RUN:?Set the prepared Test addition run}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MLSPACES_CACHE_DIR=/bigtemp/xna8aw/part-aware-policy-learning/cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/bigtemp/xna8aw/part-aware-policy-learning/cache/molmospaces/assets
cd /u/xna8aw/workspace/part-aware-policy-learning
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u \
  "$PART_BENCHMARK_EXPERT_RUN/source/data_preparation/prepare_test_set.py" stage \
  --run "$PART_BENCHMARK_EXPERT_RUN" --batch "${1:-1}"
