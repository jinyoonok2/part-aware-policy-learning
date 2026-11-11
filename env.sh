#!/usr/bin/env bash
# Source this file from any directory to use the isolated project environment.
export PART_POLICY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PART_POLICY_STORE=/bigtemp/xna8aw/part-aware-policy-learning
export PART_PYTHON="$PART_POLICY_STORE/envs/mlspaces/bin/python"
export PYTHONPATH="$PART_POLICY_ROOT:$PART_POLICY_ROOT/MolmoBot/MolmoBot:$PART_POLICY_ROOT/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PART_POLICY_STORE/cache/huggingface" TORCH_HOME="$PART_POLICY_STORE/cache/torch"
export XDG_CACHE_HOME="$PART_POLICY_STORE/cache/xdg" MPLCONFIGDIR="$PART_POLICY_STORE/cache/matplotlib"
export MLSPACES_CACHE_DIR="$PART_POLICY_STORE/cache/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="$PART_POLICY_STORE/cache/molmospaces/assets"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu
