# Part-aware interaction poses for stable door grasping

Project: **Part-Aware Pose / PAP**, RB-Y1 in MolmoSpaces. The 20k run here
trains an added pose predictor with MolmoBot frozen; it is separate from
Live Robotics Lab policy fine-tuning and Part2Action Panda experiments.

## Objective

Improve opening while maintaining a firm handle grasp over frozen MolmoBot.
Ordinary door-angle success can occur without stable grasping. Report ordinary
opening and the stricter joint grasp/opening criterion separately.

MolmoBot already receives a target-part point prompt. The project makes that
input operational through an explicit TCP pose and execution mechanism. The
final predictor consumes current RGB, robot joint state, visible handle points
and push/pull task context. Part-mask conditioning is a later extension.

## Retained controller

`grasp/` retains the combined 20 mm closing gate and bounded base recovery,
force-gated opening, grip-loss recovery, native action conversion and contact
measurement. Ten expert-pregrasp local tests improved from 7/10 to 10/10
sustained contact. This privileged diagnostic is distinct from full-start
opening and strict grasp retention. The exact fixtures and result definitions
are preserved in `grasp/fixtures/recovery_10/`.

## Prepared data

Official Train supplies 5,004 general pose references; 4,866 initial-grasp
references produce 19,464 observed-frame/target pairs. Official Val is the
project Test split: 197 source expert trajectories, 192 general pose references
and 187 early-grasp references. Preserve source roles and the exposure ledger.

`data_preparation/` owns source collection, expansion, label extraction and
Train/Test preparation. Current manifests retain raw-data and frozen-source
provenance links. Source grasp flags and successful opening do not prove the
stricter force-qualified physical retention criterion.

Evaluation keeps two separate cohorts: all 197 released Val demonstrations and
400 independently selected benchmark starts across 200 houses, with one push
and one pull task per house. Benchmark400 excludes prepared Train and existing
Test houses. Membership is fixed before expert or policy outcomes; failed
experts remain cases. Pose-label coverage and force-qualified grasp evidence
are reported separately. The earlier merged 400-total proposal is retired.

## Learned model

`pose_learning/` implements frozen local DINOv2 features from three RGB cameras,
visible part-point feature sampling, robot-state input and a trainable pose
head. Outputs are TCP position and rotation in the current observation's
robot-base frame, active arm and a measured finger-displacement proxy. The
supervision pose is the future initial-grasp TCP expressed in that observed
base frame; future images, expert phases and target poses are not inputs.

Training uses the early-grasp Train index only. The predetermined final run is
20,000 optimizer updates, microbatch 1/effective batch 4, BF16, seed 6198,
gradient clipping at 1, checkpoints every 1,000 updates and Train-only
diagnostics every 500 updates. Test does not choose a checkpoint or settings.
The PAP 20,000-update pose-predictor run completed successfully (training job
6838322; feature job 6838321). The fixed final checkpoint is saved; see
[the completed training record](../pose_learning/TRAINING_RUN.md). Offline Test
pose evaluation has completed; matched simulator grasp/opening results remain
a separate evaluation stage.

Offline evaluation measures pose errors and arm accuracy. Matched simulator
evaluation compares frozen MolmoBot with predicted-pose local correction under
the same start, seed and action budget. The latter must report ordinary door
opening, strict grasp-retention opening, failures and computation cost.
The learned finger-width proxy is not yet used as a native actuator command;
the execution bridge uses explicit open/close and contact verification.

## Next work

Complete the submitted matched baseline-versus-pose-guided simulator evaluation
using the fixed step-20,000 pose checkpoint; see
[the evaluation record](../pose_learning/EVALUATION_RUN.md). Keep the original
197-case Test cohort separate from the independent Benchmark400 expert
collection. Review oracle-control failures and metric/controller compatibility
before claiming improved grasping. No learned pose-to-action controller has
been trained by the completed pose-prediction run.

The [expert-pose diagnostic](../pose_learning/ORACLE_EVALUATION.md) compares
baseline, learned pose and recorded expert pose on the same 187 labeled starts
with the same local controller. It tests whether prediction error or execution
limits performance; source pose labels are not certified optimal grasps.

The simulator-selection and learned-critic route is deferred optional work.
Its implementation was removed and bulk experiment artifacts were moved to a
reversible archive during cleanup;
compact historical summaries and baseline evidence remain in `doc/cleanup/`.
