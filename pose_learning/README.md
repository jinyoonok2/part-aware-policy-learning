# Part-conditioned interaction-pose learning

See [input data origins](../doc/INPUT_DATA_ORIGINS.md) for the baseline's four
input groups, this predictor's four input groups, mask-derived point generation,
and expert grasp-pose target construction.

This module trains a new interaction-pose head. It does not fine-tune MolmoBot.
The frozen pretrained MolmoBot remains the navigation/manipulation policy for
online comparisons. Physical grasp-retention and opening are evaluated by the
separate online policy evaluator, rather than inferred from offline pose error.

The model receives three current RGB views, current robot joints, the push/pull
task, and a visible handle point in each current camera. Each point is the mean
of the finite visible recorded handle segmentation samples, with an explicit
visibility flag. At deployment, use the same current-frame point sensor or a
current-image perception module. A pixel from MolmoBot's initial conditioning
image cannot be reused in a later camera frame. Missing points remain invalid.
This first implementation uses point information; part-mask pooling is future work.

A local frozen DINOv2-base encoder provides global CLS and bilinearly sampled
part features for each view. A three-layer, width-256 transformer combines these
features with state, camera/point embeddings and the task. The head predicts TCP
translation, continuous 6D rotation, active arm and finger-displacement span.
All RGB images are stretched in full to 224x224, preserving normalized point
coordinates. Training and inference use exactly the same image transform.

The target is the selected future early-grasp TCP expressed in the robot-base
frame at the conditioning observation. It is not a complete handle-relative
pose: the source offline labels do not include the handle rotation. Online
execution maps the prediction to world coordinates using the current base
transform. A moving-handle execution anchor, when used by the controller, is a
separate simulator execution mechanism and must be disclosed.

Finger span is the sum of absolute measured left/right finger displacements in
metres. It is a supervision proxy, not the native gripper command (`+100` closes
the gripper). Source grasp labels come from phase/closure/flags; they have not
all been certified using physical bilateral force. Pose accuracy alone does not
establish successful or continuous grasping.

Run commands from the project root with the project Python environment:

```bash
PY=/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python
ENCODER=/bigtemp/xna8aw/part-aware-policy-learning/models/dinov2-base/f9e44c814b77203eaa57a6bdbbd535f21ede1415
$PY -m pose_learning.prepare_features --index data/interaction_pose_train_test_v1_20261107/train_early_pose_targets.jsonl --role train --encoder "$ENCODER" --output /bigtemp/xna8aw/part-aware-policy-learning/training/train_features --device cuda
$PY -m pose_learning.train --cache /bigtemp/xna8aw/part-aware-policy-learning/training/train_features --output /bigtemp/xna8aw/part-aware-policy-learning/training/pose_20k --device cuda
```

The requested defaults are 20,000 optimizer updates, microbatch 1, effective
batch 4, BF16, seed 6198, gradient clipping at 1, checkpoint every 1,000 updates,
and Train-only diagnostics every 500. The frozen-encoder pose head uses an
explicit learning rate of 0.0003 with warmup/cosine decay. This is a head-training
choice, not a claim to reproduce the separate LRL model's fine-tuning recipe.
Every 5k milestone and the latest checkpoint are retained. The final 20k
checkpoint is predetermined; Test never selects a checkpoint or a setting.
Resume using the same arguments plus `--resume RUN/checkpoints/stepXXXXX.pt`.

Feature extraction refuses overwrite and records source-index/encoder hashes,
observed frames, full/smoke counts, split, transforms and output hashes. The
trainer refuses Test caches and fits state normalization on Train only. Global
base coordinates are excluded from robot-state features. `--verify-h5` can
recheck raw HDF5 hashes in addition to the preceding dataset-preparation audit.

For a small actual-data integration check, add `--limit 4` to Train feature
extraction, then train with `--updates 4 --warmup 0 --checkpoint-every 4
--diagnostic-every 4 --allow-smoke-cache`. Use `--device cpu --precision float32`
for CPU optimizer checks. A smoke-limited cache cannot be used for a final run
without an explicit override and cannot represent final Test evaluation.

Offline Test evaluation is an explicit later action:

```bash
$PY -m pose_learning.prepare_features --index data/interaction_pose_train_test_v1_20261107/test_early_pose_targets.jsonl --role test --encoder "$ENCODER" --output /bigtemp/xna8aw/part-aware-policy-learning/training/test_features --device cuda
$PY -m pose_learning.evaluate --cache /bigtemp/xna8aw/part-aware-policy-learning/training/test_features --checkpoint /bigtemp/xna8aw/part-aware-policy-learning/training/pose_20k/checkpoints/step20000.pt --output /bigtemp/xna8aw/part-aware-policy-learning/training/test_pose_metrics
```

The offline report includes translation/orientation errors, arm accuracy and
finger-span error, per observation and averaged by source trajectory. Multiple
conditioning frames from one expert trajectory are not separate episodes.
The Test loader only accepts official Val sources under the project's Test
role. No Test feature extraction or evaluation runs automatically during Train.

```bash
$PY -m unittest discover -s pose_learning/tests -v
```

The tests cover split refusal, point visibility/padding, current-frame indexing,
world-coordinate exclusion, quaternion sign equivalence, proper rotations,
masking missing local features and finite, learnable pose-head supervision.

## Matched simulator evaluation

After freezing a learned checkpoint and final task manifest, run:

```bash
source env.sh
"$PART_PYTHON" -m pose_learning.evaluate_policy \
  --episodes /path/to/policy_episode_manifest.json \
  --pose-checkpoint /path/to/checkpoints/step20000.pt \
  --output /bigtemp/xna8aw/part-aware-policy-learning/runs/pose_comparison \
  --methods baseline predicted_pose --seed 42 --horizon 400 --video
```

`data_preparation.export_policy_episodes` converts source initial-state metadata
into explicit EpisodeSpecs after matching scene identities. The independent
Benchmark400 cohort also contains materialized EpisodeSpecs. All 400 starts
remain in its task-performance population; expert recording readiness and
pose-label coverage are tracked separately. Keep it separate from Val197.
No Test preparation or execution is launched by training.

The policy evaluator restores the same integration state, checks physics and
initial sensor/state inputs, and requires the first MolmoBot proposal to match.
Initial images may differ by at most 2/255 before using the same audited pixels.
A failed comparison is recorded as failed/incomparable. Both methods use the
same original start, model checkpoint, seed and 400-action budget. The learned
method selects its own arm and uses predicted TCP targets; it does not load
expert actions or expert target poses. It also uses current-frame simulator
handle points and contact feedback, which must be disclosed when comparing
against MolmoBot's original initial-image point prompt.

The execution bridge currently uses bounded arm corrections inside a local
contact gate; MolmoBot supplies approach/opening actions. The full handcrafted
base-recovery/guarded-opening controller remains separately reproducible in
`grasp/`. These are distinct execution experiments. The predicted finger-span
proxy is logged; native gripper open/close still uses geometric alignment and
contact checks. Ordinary opening and onset-based retained-grasp opening are
reported separately. CPU checks do not substitute for a GPU rollout test.

Slurm wrappers are in `pose_learning/slurm/`. The full training wrapper requests
BF16 and therefore needs a GPU that supports BF16. They support a frozen package
root with `PART_POSE_SOURCE_ROOT` and a frozen Train index with `PART_POSE_INDEX`.
Setting `PART_POSE_GPU_PREFLIGHT` makes the training wrapper run a separate
four-update BF16 integration check before beginning fresh full training.

The full Train feature extraction, BF16 GPU check and 20,000-update training are
complete. See [the run record](TRAINING_RUN.md). Test was not accessed; held-out
pose accuracy and physical grasp/opening improvement remain to be evaluated.

The subsequent existing-set evaluation is now submitted. It uses all 748 labeled
Test observations for offline errors and all 197 source starts for matched
rollouts. The report preserves pending cases and technical errors separately
from task failures. See [the evaluation record](EVALUATION_RUN.md).
