# PAP pose predictor — 20,000 updates with MolmoBot frozen

Owner: `part-aware-policy-learning`, Part-Aware Pose (PAP). This is the added
pose-module run, not either Live Robotics Lab 20k policy checkpoint and not a
Part2Action run. The shared update count and seed do not make them duplicate
experiments. See the
[experiment ownership index](../../research_projects/interactive-robotics/part2action/docs/EXPERIMENT_OWNERSHIP.md).

Submitted November 7, 2026 (America/New_York), at the user's request. Further
validation expert generation and replay were deferred; existing data remain.

## Completed workflow

| Stage | Slurm job | Status | Elapsed |
| --- | --- | --- | --- |
| Verify Train HDF5 hashes and extract frozen DINOv2 features | 6838321 | Completed, exit 0 | 1h 51m 59s |
| Four-update BF16 integration check, then fresh 20,000-update training | 6838322 | Completed, exit 0 | 42m 19s |

Verified completion: the full 19,464-observation Train cache is complete, the
four-update GPU check passed, and `train/status.json` records all 20,000 updates.
The final `step20000.pt` checkpoint exists (32,601,547 bytes). Test was not
accessed. Held-out pose accuracy and simulator grasp/opening performance remain
to be evaluated. Validation expert expansion remains deferred.

## Data and recipe

- 4,866 early-grasp expert trajectories from 973 Train houses.
- Four conditioning observations per trajectory: 19,464 supervised examples.
- All 15,795 referenced HDF5/video files were present at submission; the feature
  job additionally recomputes every selected HDF5 hash.
- Frozen DINOv2-base; train the three-layer width-256 pose transformer.
- Inputs: three current RGB views, robot joints, current visible handle points
  and push/pull task. Outputs: TCP position/orientation, arm and finger span.
- 20,000 optimizer updates; microbatch 1, effective batch 4, BF16, seed 6198.
- AdamW, learning rate 0.0003, 500-update warmup, cosine decay, gradient clip 1.
- Checkpoint every 1,000 updates; retain 5k milestones and latest checkpoint.
- Train-only diagnostics every 500 updates; predetermined final step-20,000
  checkpoint. The GPU integration-check weights are not reused for full training.
- Seventeen CPU tests passed. Physical grasp/opening improvement remains to be
  evaluated after training; source pose labels are not force-certified grasps.

## Outputs

The local `training/` symlink points to project storage. This run is
`training/pose_20k_20261107_01/`:

- `plan.json`, `preflight.json`, `submissions.json`: recipe, checks and jobs.
- `source/`, `source_manifest.json`: frozen code/index and source hashes.
- `features/metadata.json`: complete/incomplete Train cache and source identity.
- `gpu_preflight/status.json`: GPU integration-check outcome.
- `train/status.json`, `train/train.jsonl`: full-run progress and Train losses.
- `train/checkpoints/step20000.pt`: completed final checkpoint.
- `logs/features-6838321.log`, `logs/train20k-6838322.log`: scheduler output.

```bash
squeue -j 6838321,6838322
```
