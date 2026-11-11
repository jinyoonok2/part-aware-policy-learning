# MolmoBot-p: Part-conditioned interaction poses for stable door grasping

Presentation date: 2026-11-08.

Status snapshot: 2026-11-08T18:07:27-05:00 (America/New_York).

Text draft; visualizations will be added in [the November 8 media folder](media/2026-11-08/README.md).

## MolmoBot-p: From part prompts to grasp poses

Goal: Open doors while maintaining a firm handle grasp.

MolmoBot already receives a target-point prompt; we test whether explicit pose guidance makes part information more useful.

Current RGB + robot state + handle points → gripper position, orientation, arm and finger span.

Use expert-derived poses for supervision and controller checks; predict poses from current observations at deployment.

Keep MolmoBot frozen and guide local grasp execution with the predicted pose. Part-mask inputs are a later extension.

<!-- Planned visualization: part information -> pose prediction -> grasp execution. -->

## MolmoBot-p: What the control experiments showed

Local contact after an expert approach: 7/10 → 10/10 with combined closing and base recovery; this measures local grasping.

Separate full-start Train pilot: ten starts, seed 42, 400-action budget.

Ordinary opening / retained-grasp opening: MolmoBot 9/10 / 2/10; grasp guard 10/10 / 3/10; oracle correction 8/10 / 4/10.

The full-start results use the historical retention criterion and show a tradeoff between opening and grasp retention.

Learned-model improvement remains untested.

<!-- Evidence: [Local contact](../../grasp/fixtures/recovery_10/results.json), [baseline](../cleanup/2026-11-07/notes/2026-11-07_baseline_reference.json), [full pilot](../CoRL_camera_ready_template_2024/part_aware_pose_milestone.tex). Planned visualization: compact results table and grasp examples. -->

## MolmoBot-p: Expert supervision and pose model

Train: 4,866 early-grasp trajectories across 973 houses → 19,464 observation–pose pairs.

Eval-A: 197 released Val demonstrations, 187 with pose labels. Eval-B: 400 separate benchmark starts; expert collection queued.

Targets are recorded expert gripper poses expressed in the conditioning observation's robot-base frame.

Source grasp flags and closure select labels; continuous stable grasp is not verified for every demonstration.

Frozen DINOv2 encodes three views; a three-layer transformer combines image features, robot state, handle points and push/pull task information.

<!-- Evidence: [Dataset](../../data_preparation/README.md), [model](../../pose_learning/README.md). Planned visualization: data counts and pose-model architecture. -->

## MolmoBot-p: Training and next evaluation

Completed: features for all 19,464 Train examples, the BF16 GPU check and all 20,000 optimizer updates.

Trained only the pose module: effective batch 4, BF16, seed 6198; MolmoBot remained frozen.

Saved the predetermined final checkpoint using Train-only diagnostics; Test was not accessed.

Eval-A pose errors: 8.8 cm mean position, 11.7° mean orientation, 98.0% arm accuracy. Matched rollouts are preparing.

Compare ordinary opening and retained-grasp opening on both sets. Training curves and comparison videos will follow.

<!-- Evidence: [Protocol](../../configs/train_test_protocol_v1.json), [training run](../../pose_learning/TRAINING_RUN.md), [evaluation run](../../pose_learning/EVALUATION_RUN.md). Planned visualization: training curves and matched rollouts. -->
