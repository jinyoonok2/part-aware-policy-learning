# Cleanup and implementation record: 2026-11-07

This is a dated historical record. Current workflows are documented in the
[project README](../../../README.md). The old merged 400-total finalizer was
retired on November 11; original Val197 and Benchmark400 remain separate.

November 11 repository cleanup kept `baseline_pilot_10/*/summary.json` and
`initial_reference.json` here. Its duplicate `episode.json` and
`decoded_config.json` files are retained canonically in
[`grasp/fixtures/recovery_10/`](../../../grasp/fixtures/recovery_10/).
The 20 raw contact traces and copies of every removed file remain in the
ignored local `outputs/cleanup_20261111/` archive. The
[file manifest](../2026-11-11/files.json) records their hashes and exact paths.
Frozen experiment snapshots and source datasets were unchanged by that cleanup.

## Active structure

- `grasp/`: combined 20 mm closing/base recovery; guarded opening; contact and
  retention metrics; native action bridge; frozen ten-case reproduction fixtures.
- `data_preparation/`: source collection/expansion, early labels, Train/Test
  manifests, metadata planning toward 400 total Test cases, expert verification,
  episode export and final qualified-set assembly.
- `pose_learning/`: frozen point-conditioned visual features, pose training,
  offline pose error evaluation and matched MolmoBot simulator evaluation.

The old flat scripts directory (130 inventoried files) and ten obsolete root
configs were removed after extracting the retained functionality. File identities
are in `removed_code_manifest.json` and `removed_config_manifest.json`.

## Results retained

`runs/grasp_7to10` is a curated copy of the original reviewed traces, reference
and combined controller results. It is not a new simulation run. Its provenance
and original worker statuses are preserved. Three historical workers finished
all simulations but failed an overly strict reference-matching assertion;
the independent numerical review and saved contact traces support the 7/10 to
10/10 local contact count. This does not establish 10/10 full-start opening.

`historical_results.json` retains compact experiment summaries. The
`baseline_pilot_10/` directory preserves the original frozen MolmoBot baseline
summaries, contact traces and episode/configuration evidence needed to compare
future model results. `training_history.json` records that readiness passed but
no 20,000-update pose-training job had been submitted.

## Data and reversible archive

Current Train/Test raw sources, labels and provenance remain in place. The
current alias depends on the v2 dataset, its v1 raw-house targets and the v2
quality view. Those folders, the project environment, frozen visual encoder,
shared pretrained MolmoBot and upstream submodules were preserved.

A new `data/test_400_plan` records 197 retained existing Test trajectories and a
400-candidate pool for additions toward 400 total. If all existing cases qualify,
203 additions are needed. Planning did not run an expert or qualify new cases.

Twenty-four old run folders and two unused dataset views were moved to
`/bigtemp/xna8aw/part-aware-policy-learning/archive/cleanup_20261107`.
`archived_artifacts.json` is the exact original-to-archive mapping. This action
is reversible and frees no disk space. Automatic approval review rejected
permanent deletion because it could remove useful evidence. Nothing in that
archive has been permanently deleted; that step remains pending explicit review.
The active candidate-house reservation manifest was copied into `configs/`
before archiving so future Train preparation keeps its exclusions.

The presentation files and CoRL report source remain available. Dated experiment notes in `notes/` describe historical paths; use this record for their archive
locations. The new documentation is the active command reference.

## Implementation validation

CPU checks passed for the retained controller, native action adapter, dataset
preparation, pose geometry/losses, split contracts and evaluation inputs. A
real-data smoke extracted frozen DINOv2 features for four Train observations,
completed four optimizer updates, reloaded the checkpoint for a Train-only
pose-error report, and saved its predictions without
accessing Test. Compact receipts are in `pose_training_smoke/`. This is an
integration test, not evidence of improved grasping.

The requested full 20,000-update job and GPU policy rollouts have not launched.
The rebuilt milestone PDF still has four main pages plus the GenAI appendix.
