#!/usr/bin/env bash
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%2
#SBATCH --no-requeue
set -euo pipefail
: "${PART_GRASP_RUN:?Set a new output directory}"
PART_POLICY_ROOT="${PART_POLICY_ROOT:-/u/xna8aw/workspace/part-aware-policy-learning}"
PART_POLICY_STORE="${PART_POLICY_STORE:-/bigtemp/xna8aw/part-aware-policy-learning}"
module load cuda/12.8.1 gcc/12.4.0
export PYTHONPATH="$PART_POLICY_ROOT:$PART_POLICY_ROOT/MolmoBot/MolmoBot:$PART_POLICY_ROOT/molmospaces"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PART_POLICY_STORE/cache/huggingface" TORCH_HOME="$PART_POLICY_STORE/cache/torch"
export XDG_CACHE_HOME="$PART_POLICY_STORE/cache/xdg" MPLCONFIGDIR="$PART_POLICY_STORE/cache/matplotlib"
export MLSPACES_CACHE_DIR="$PART_POLICY_STORE/cache/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="$PART_POLICY_STORE/cache/molmospaces/assets" WANDB_MODE=offline
cd "$PART_POLICY_ROOT"
"$PART_POLICY_STORE/envs/mlspaces/bin/python" -u -m grasp.reproduce \
  --run "$PART_GRASP_RUN" --index "${SLURM_ARRAY_TASK_ID:-0}" "$@"
