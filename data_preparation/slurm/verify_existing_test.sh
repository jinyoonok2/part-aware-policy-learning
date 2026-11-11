#!/usr/bin/env bash
#SBATCH --job-name=part-existing-test
#SBATCH --account=xna8aw_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_EXISTING_TEST_RUN:?Set the prepared existing Test replay run}"
export PART_POLICY_ROOT=/u/xna8aw/workspace/part-aware-policy-learning
export PART_POLICY_STORE=/bigtemp/xna8aw/part-aware-policy-learning
export PYTHONPATH="$PART_EXISTING_TEST_RUN/source:$PART_POLICY_ROOT/MolmoBot/MolmoBot:$PART_POLICY_ROOT/molmospaces"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu
export XDG_CACHE_HOME="$PART_POLICY_STORE/cache/xdg" MPLCONFIGDIR="$PART_POLICY_STORE/cache/matplotlib"
export MLSPACES_CACHE_DIR="$PART_POLICY_STORE/cache/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="$PART_POLICY_STORE/cache/molmospaces/assets"
export WANDB_MODE=offline HF_HUB_OFFLINE=1
module load cuda/12.8.1 gcc/12.4.0
cd "$PART_POLICY_ROOT"
exec "$PART_POLICY_STORE/envs/mlspaces/bin/python" -u "$PART_EXISTING_TEST_RUN/source/existing_test_worker.py" \
  worker --run "$PART_EXISTING_TEST_RUN" --index "${PART_EXISTING_TEST_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
