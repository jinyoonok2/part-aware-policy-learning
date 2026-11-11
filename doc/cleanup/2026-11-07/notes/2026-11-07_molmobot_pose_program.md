# Frozen MolmoBot with local interaction-pose guidance

Updated November 7, 2026. This is the current course-project direction. The
previous candidate-selector/critic study remains intact but further work on it
and mask-only adaptation is deferred.

## Goal and execution order

Improve handle grasping while preserving the pretrained MolmoBot policy for
approach and opening. Contact points identify a location; part geometry and a
TCP pose can additionally describe orientation and gripper configuration.
First test the value of an accurate supplied pose, then learn to predict that
pose from observations if the intervention is useful. No learned pose module
or pose-model training is started.

1. Expand official Train expert references from 503 toward at least 5,000.
2. Validate real MolmoBot inference and the local correction interface with a
   short smoke; only then run the matched 30-rollout development pilot.
3. Review oracle benefit, source/split audits and early-grasp label quality.
4. If these checks pass, implement and train a pose predictor; replace oracle
   inputs and compare against the same frozen baseline and guard control.
5. Tune on development/Val data and evaluate the final system on untouched
   starts. Do not use these ten Train cases as generalization evidence.

## Expanded expert data

Dataset: `/bigtemp/xna8aw/part-aware-policy-learning/datasets/interaction_pose_v2_5000_20261107`.
Configuration: `configs/interaction_pose_dataset_v2_5000.json`.
CPU collection job `6836690` completed with 5,004 references across 984 houses
and 55.27 GB compressed total. Final source/transform/split audits passed.

The expansion reuses the original 503 references from 98 houses without
modifying their source files. It continues the frozen official Train house
order and pinned source revision. Target: at least 5,000 eligible pose
references, stopping at whole-house boundaries. Rough compressed-transfer
estimate: 55 GB total; cap: 80 GB including the original subset. The final
achieved count and transfer total come from the dataset report, not this
estimate. No final benchmark rollout is performed by collection.

The separate Val pose index remains 192 references from 197 existing expert
trajectories. The official Val benchmark's 2,000 task starts are a different
resource. Train/Val source isolation, unchanged Val index, duplicate detection,
source HDF5 hashes and measured/conditioning pose transforms are audited.
Numeric house-ID separation is an extra check, not proof of geometric
disjointness across scene families.

Final `audit/dataset_integrity.json` and `audit/expansion_integrity.json` must
have `state: passed`, `complete_target_required: true`, source/Val-isolation
flags and manifest hashes matching the actual files. An early audit with
`complete_target_required: false` does not certify the 5,000-reference target.
Keep the original [extraction rules and limitations](2026-11-06_interaction_pose_dataset.md).
Metadata grasp flags and measured closure are offline evidence; they do not
establish that every extracted pose is a firm physical grasp.

A separate early-grasp quality view selects targets at
at most 10% source door opening, with no prior major opening, and records cases
without such an event. It preserves original raw files and indices. Its
coverage and exclusions must be reviewed before choosing training labels;
5,000 general references need not yield 5,000 early-grasp targets. The initial
503-reference audit retained 484 Train and 187 Val early targets. Of 169
original Train targets selected after 10% opening, 150 had an earlier eligible
event that could be recovered. This is an extraction-quality result, not a
physical grasp success rate.

Final view: `/bigtemp/xna8aw/part-aware-policy-learning/datasets/interaction_pose_v2_early_quality_20261107`.
CPU job `6836705` completed after successful collection `6836690`. It retained
4,866 Train labels / 19,464 conditioning examples across 973 houses, and 187
Val labels / 748 examples. The quality view does not replace contact/retention
validation in the simulator.

## Actual policy pilot

Runner: `scripts/molmobot_oracle_pilot.py`.
Checkpoint: `/bigtemp/xna8aw/LRL/hf_models/MolmoBot-RBY1Multitask`.
Corrected frozen run: `/bigtemp/xna8aw/part-aware-policy-learning/runs/molmobot_oracle_pose_20261107_03`.
Smoke job `6836701` completed successfully (exit 0, 5m29s, RTX5080). All three
methods performed actual inference with identical first proposals and initial
conditioning hashes. The two-step runs had no local interventions; they do not
measure task performance or validate an actual local-control handoff. Full case
array `6836702` (`0-9%2`) and report `6836703` completed all 30 rollouts and
matched-input audits. Under the pilot's historical criterion, strict retained
opening was baseline 2/10, guard 3/10, oracle 4/10; door-threshold success was
9/10, 10/10 and 8/10 respectively. This is a development feasibility result,
not learned-model or overall task improvement. The separately frozen
[final evaluation](2026-11-07_final_pose_evaluation.md) uses onset-based grasp
verification; do not compare its rates directly with these pilot values.

The ten fixed Train development cases begin from their original recorded
starts. Each method gets a fresh scene with the same initial integration
state, images, robot state, point prompt, policy checkpoint, seed and first
model proposal. All methods use native action/control modes and a 400-decision
budget. There is no expert approach replay.

| Method | What changes |
| --- | --- |
| `baseline` | Frozen native MolmoBot actions, without the local grasp wrapper |
| `guard` | The same MolmoBot plus bounded grasp verification/recovery |
| `oracle_pose` | The same guard plus expert-relative local pose correction |

MolmoBot remains responsible for approach and opening in all three methods.
Local intervention is triggered by current proximity and the model's closing
intent, restricted to the expert-labelled arm, and bounded in duration/retries.
Oracle targets and live handle state are privileged information, not learned
perception. Opposite-arm choices, timeouts and failed tasks stay in the report.

The native policy uses relative arm/base commands and scalar torso height.
The bridge converts local targets without resetting the scene or
re-registering the policy. It invalidates stale action chunks at handoff while
preserving initial image/point conditioning and observation accounting. Raw
NaN-padded point prompts are preserved for the model; diagnostics hash their
arrays instead of writing non-finite numbers to strict JSON.

Native early stopping on door-angle success is disabled equally for all three
methods so grasp retention can still be observed. Report native door-threshold
success and strict retained-grasp opening separately; also save per-arm traces,
interventions, action counts, audit hashes and videos. Do not compare these
strict rates with past near-handle or differently stopped runs.

Submission provenance: v1 job `6836694` failed on an unnecessary
`candidate_search` import dependency. The import now uses the existing state
helper directly. Pending v2 smoke `6836699` was cancelled after review found
non-finite point data in JSON diagnostics; the raw policy inputs were retained
and only diagnostics changed. The v3 smoke supersedes both attempts. Neither
attempt is a valid policy-performance result.

## Requested training protocol (not started)

On November 7 the user requested that future training follow the recent Live
Robotics Lab 20k setup. The reference is the completed corrected grounded run,
not the older reduced-input recipe:

- Recipe: `nav_door_house_grounded_20k.yaml` (historical path: `live-robotics-lab/LRL_project/configs/nav_door_house_grounded_20k.yaml`).
- Actual run: `/bigtemp/xna8aw/nav_door/train/final/house/grounded_20k_gb4_20261030`.
- Completion evidence: `stages/20000.json` records 20,000 completed updates and
  `model/step20000`; executed stage commands are retained in `commands/`.
- Budget: 20,000 optimizer updates, with stages at 5k, 10k, 15k and 20k.
- Microbatch 1, global batch 4: four accumulation steps on one GPU.
- Validation every 500 updates, up to 512 offline validation examples.
- Checkpoints every 1,000 updates, temporary checkpoints every 250 and retained
  checkpoints every 5,000; resume includes optimizer and update state.
- BF16, seed 6198 and gradient norm clipping at 1.0.
- Reference resources: one H100 94 GB, 10 CPUs, 200 GB host RAM.

The reference fine-tuned MolmoBot's LLM and action expert while freezing the
ViT and connector. Their learning rates were 1e-5 and 1e-4, with warmups of
2,000 and 200 updates respectively. The pose-head design has different
trainable parameters, so those parameter-group learning rates and freeze flags
are reference values, not an implemented pose-training configuration. Preserve
the pretrained policy for the initial pose-head experiment; record any later
decision to fine-tune it as a separate method.

The shared budget, batch, validation and saving protocol are the requested
starting reference. Pose loss, head learning rate, augmentation compatibility,
and the exact module remain to be implemented after oracle/data review.
20,000 updates counts optimizer steps, not trajectories or simulator actions.
No pose-model training job has been submitted; current collection, quality,
pilot and readiness jobs do not automatically launch training.

## Gate before learned-pose training

`scripts/report_pose_training_readiness.py` records the frozen minimum gate:
all ten cases and all three methods complete, no infrastructure errors,
explicit model-inference and matched-input audits pass, and oracle strict
retention has at least one net additional success over the guard. The paired
gain is oracle wins minus guard wins; equality or regression fails the gate.
This is a feasibility threshold for ten Train development cases, not a
statistical finding or held-out result. Baseline versus guard identifies the
wrapper effect independently of oracle versus guard.

The same report requires at least 5,000 general Train references, the completed
source/Val-isolation audit and an audited early-grasp view of those exact source
manifests. It checks separate Train/Val identities, <=10% opening with no prior
major opening, manifest hashes, conditioning counts and exclusion ledgers.
Usable early Train/Val counts are reported separately; they may be below 5,000.
Missing evidence or a view of the original 503-reference subset blocks the
gate. Review coverage and target quality before choosing the learned model's
exact training index. A passed automatic gate never starts training or
establishes that the labels are physically validated.

```bash
python scripts/report_pose_training_readiness.py \
  --pilot /bigtemp/xna8aw/part-aware-policy-learning/runs/molmobot_oracle_pose_20261107_03 \
  --dataset /bigtemp/xna8aw/part-aware-policy-learning/datasets/interaction_pose_v2_5000_20261107 \
  --quality-view /bigtemp/xna8aw/part-aware-policy-learning/datasets/interaction_pose_v2_early_quality_20261107 \
  --output NEW_REVIEW_DIRECTORY
```

The helper writes `criteria.json`, `readiness.json` and `index.html`. Add
`--require-ready` to return exit code 2 when prerequisites are missing. It does
not submit jobs or train models. Criterion schema v2 adds the explicit
early-grasp quality requirement. Existing `criteria.json` is immutable: use a
new output directory rather than overwrite a v1 report. The program directory
is `/bigtemp/xna8aw/part-aware-policy-learning/runs/molmobot_pose_program_20261107_01`;
its earlier `readiness/` preserves the original criterion. Final readiness job
`6836708` waits with `afterany:6836703:6836705` and writes `readiness_v2/` using
the frozen helper through `scripts/slurm/report_pose_program.sh`. It reports
missing/failed prerequisites even when a preceding experiment fails; it never
launches training. Read each report's recorded input paths and criterion
version before interpreting it.

Authoritative progress comes from each run's `study.json`, `submissions.json`,
`status.json`, per-case summaries and scheduler records. A submitted job or
successful smoke is not evidence that all 30 task rollouts have completed.
