#!/usr/bin/env bash
#SBATCH --job-name=part-pose-test-assets
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
source /u/xna8aw/workspace/part-aware-policy-learning/env.sh
: "${PART_POSE_SOURCE_ROOT:?Set the frozen evaluation package root}"
: "${PART_EXISTING_TEST_RUN:?Set the prepared source-asset staging run}"
: "${PART_POSE_SOURCE_INVENTORY:?Set the complete frozen 197-case source inventory}"
: "${PART_POSE_EPISODE_DIR:?Set a fresh output for frozen EpisodeSpecs}"
cd "$PART_POSE_SOURCE_ROOT"
"$PART_PYTHON" -u "$PART_EXISTING_TEST_RUN/source/prepare_existing_test_verification.py" \
  stage --run "$PART_EXISTING_TEST_RUN"
exec "$PART_PYTHON" -u -m pose_learning.stage_policy_manifest \
  --input "$PART_EXISTING_TEST_RUN/policy_episode_manifest.json" \
  --expected-source "$PART_POSE_SOURCE_INVENTORY" --output "$PART_POSE_EPISODE_DIR"
