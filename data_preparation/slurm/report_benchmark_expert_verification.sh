#!/usr/bin/env bash
#SBATCH --job-name=part-benchmark-expert-report
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_BENCHMARK_EXPERT_RUN:?Set the prepared benchmark expert-verification run}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/usr/bin/python3 "$PART_BENCHMARK_EXPERT_RUN/source/report_benchmark_expert_verification.py" \
  --run "$PART_BENCHMARK_EXPERT_RUN"
