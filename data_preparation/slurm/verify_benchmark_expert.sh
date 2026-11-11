#!/usr/bin/env bash
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4500
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_BENCHMARK_EXPERT_RUN:?Set the prepared exact-start verification run}"
export PART_POLICY_ROOT=/u/xna8aw/workspace/part-aware-policy-learning
export PART_POLICY_STORE=/bigtemp/xna8aw/part-aware-policy-learning
export PYTHONPATH="$PART_BENCHMARK_EXPERT_RUN/source:$PART_POLICY_ROOT/molmospaces"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$PART_POLICY_STORE/cache/xdg" MPLCONFIGDIR="$PART_POLICY_STORE/cache/matplotlib"
export HF_HOME="$PART_POLICY_STORE/cache/huggingface" TORCH_HOME="$PART_POLICY_STORE/cache/torch"
export TORCH_EXTENSIONS_DIR="$PART_POLICY_STORE/cache/torch_extensions/curobo_417"
export MLSPACES_CACHE_DIR="$PART_POLICY_STORE/cache/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="$PART_POLICY_STORE/cache/molmospaces/assets"
export WANDB_MODE=offline PYTHONFAULTHANDLER=1
export PART_CUROBO_BUILD_RECEIPT="$PART_BENCHMARK_EXPERT_RUN/source/planner_build_receipt.json"
module load cuda/12.8.1 gcc/12.4.0
cd "$PART_POLICY_ROOT"
case_index="${PART_EXPERT_VERIFY_CASE_INDEX:-${PART_BENCHMARK_EXPERT_INDEX:-}}"
if [[ -z "$case_index" ]]; then
  case_index=$(( ${PART_BENCHMARK_EXPERT_CASE_OFFSET:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
fi
"$PART_POLICY_STORE/envs/mlspaces/bin/python" -u "$PART_BENCHMARK_EXPERT_RUN/source/verify_benchmark_expert.py" \
  worker --run "$PART_BENCHMARK_EXPERT_RUN" --index "$case_index"
