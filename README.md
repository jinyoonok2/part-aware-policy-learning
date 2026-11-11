# Part-Aware Interaction Poses for Stable Door Grasping

**Project scope: Part-Aware Pose (PAP).** An added pose predictor is trained
with MolmoBot frozen. This repository covers RB-Y1 door manipulation in
MolmoSpaces; Live Robotics Lab policy fine-tuning and Part2Action's Panda
experiments are separate projects.

The project learns an explicit gripper pose from current RGB, robot joints,
push/pull task type, and current handle points in three views. A local controller
uses that pose around frozen MolmoBot. The simulator supplies the current handle
points from segmentation; MolmoBot retains its original initial-image point
prompt. See [input origins](doc/INPUT_DATA_ORIGINS.md) for the exact inputs and
supervision labels. Full part-mask conditioning is a later extension.

## Code

| Folder | Purpose | Entry point |
| --- | --- | --- |
| [grasp/](grasp/README.md) | Retained combined closing/base-recovery controller and grasp-aware opening | `python -m grasp.reproduce` |
| [data_preparation/](data_preparation/README.md) | Official Train subset, early-grasp labels and separate Val197/Benchmark400 Test cohorts | `python -m data_preparation.prepare_test_set` |
| [pose_learning/](pose_learning/README.md) | Frozen visual features, pose-head training, offline pose evaluation and matched simulator evaluation | `python -m pose_learning.train` |

Historical results and original baseline evidence are summarized in
[the cleanup record](doc/cleanup/2026-11-07/README.md). Superseded combined-Test
finalization code has been retired; current evaluations keep Val197 and
Benchmark400 separate.

## What has been demonstrated

The retained combined controller improved sustained local grasp contact from
7/10 to 10/10 on ten Train starts after an expert approach. The reproduction
settings and reviewed evidence are under `grasp/fixtures/recovery_10/`.
That result uses expert pose and simulator geometry; it does not establish
10/10 full-start opening or learned-model improvement.

The requested 20,000-update pose-head training is complete. Train feature
extraction job `6838321` and training job `6838322` both finished successfully;
the final checkpoint is saved. The recipe used effective batch 4, BF16,
seed 6198 and Train-only diagnostics. Held-out pose evaluation is complete;
grasp/opening improvement is assessed separately through matched rollouts. See
[the training run record](pose_learning/TRAINING_RUN.md) for outputs and status.
The original Test evaluation covers 187 labeled trajectories for pose errors
and 197 matched starts for baseline/pose-guided rollouts. A separate
[expert-pose diagnostic](pose_learning/ORACLE_EVALUATION.md) compares three
methods on the 187 labeled starts. See [the evaluation record](pose_learning/EVALUATION_RUN.md)
and [Benchmark400 collection record](data_preparation/BENCHMARK400_RUN.md) for
run identities; each run's reports and scheduler state provide live progress.

## Data

Use `data/interaction_pose_train_test_v1_20261107`:

- Train: 5,004 general pose references; 4,866 early-grasp references provide
  19,464 observed-frame/target pairs.
- Test: all 197 released official Val expert trajectories remain; 192 have
  general pose references and 187 have early-grasp references.
- Separate Test cohort: `data/benchmark_eval400_v1_20261108` freezes 400 starts
  from the 2,000-episode benchmark, across 200 houses with 200 push/200 pull tasks.
  Native experts provide additional recordings and pose labels. This set
  contains zero cases from the original 197 and does not change them.

All 400 starts remain in the task-performance population, including expert
failures. Recording completeness and usable pose-label coverage are reported
separately. Old combined-target plans are historical records. Collection is tracked
under `runs/benchmark_eval400_collection_20261108_01`. Raw source folders beneath `data/`
are dependencies of the current manifests, including symlink targets, rather
than disposable duplicate datasets. Test never selects training settings or
checkpoints. See [the protocol](configs/train_test_protocol_v1.json).

## Environment and commands

```bash
source env.sh
"$PART_PYTHON" -m grasp.reproduce --help
"$PART_PYTHON" -m data_preparation.prepare_test_set --help
"$PART_PYTHON" -m pose_learning.train --help
"$PART_PYTHON" -m pose_learning.evaluate --help
"$PART_PYTHON" -m pose_learning.evaluate_policy --help
```

Use the corresponding package's `slurm/` directory for GPU execution. No command
above submits jobs. [Setup](doc/SETUP.md) describes the isolated environment.

The [research plan](doc/RESEARCH_PLAN.md), [milestone requirements](doc/PROJECT_MILESTONE_PROMPT.md)
and [CoRL milestone source](doc/CoRL_camera_ready_template_2024/part_aware_pose_milestone.tex)
remain available. Upstream `MolmoBot/` and `molmospaces/` remain pinned submodules.

[Presentation history](doc/presentations/README.md) groups dated slide documents
and their media for the November 2 and November 8 updates.

## Repository contents

Commit source, configuration, tests, paper sources and presentation assets.
Datasets, checkpoints, simulator caches, run outputs and local cleanup backups
are ignored. The November 11 cleanup removed duplicate fixture copies and
kept historical raw traces locally under `outputs/cleanup_20261111/`;
[the file manifest](doc/cleanup/2026-11-11/files.json) records hashes and locations.
The canonical reproduction fixtures and compact historical summaries remain
in the repository.
