#!/usr/bin/env bash
#SBATCH --job-name=part-bench400-export
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
source /u/xna8aw/workspace/part-aware-policy-learning/env.sh
: "${PART_BENCHMARK_EXPERT_RUN:?Set the standalone benchmark cohort}"
cd "$PART_BENCHMARK_EXPERT_RUN/source"
exec "$PART_PYTHON" -u -m data_preparation.export_benchmark_pose_labels \
  --run "$PART_BENCHMARK_EXPERT_RUN" "$@"
