# Evaluation of the first trained pose head

Requested and submitted November 8, 2026. Run:
`runs/pose_test197_20261108_01`.

The checkpoint is the predetermined step-20,000 checkpoint from
`training/pose_20k_20261107_01/train/checkpoints/step20000.pt`. Evaluation does not
select a checkpoint or change the trained model. The model, input indices and
evaluation code are frozen and hashed in the run.

## November 10 recovery

The original asset job ended with 196/197 starts staged. Extraction of 14
Objaverse packages failed for one source case in house 4098. No policy rollout
started, and no missing case was dropped. The offline pose evaluation completed
successfully and is not being repeated.

| Retry job | Stage |
| --- | --- |
| 6858961 | Retry source asset staging and freeze all 197 starts |
| 6858962 | First full matched baseline/pose-guided episode, after staging succeeds |
| 6858963 | Remaining cases 1–196, after the first paired episode completes |

The retry uses the same frozen policy sources, checkpoint, seed, horizon and
population. `retry_submissions_20261110_01.json` records the commands. Benchmark400
asset staging is already complete; no concurrent asset writer is queued. Its
expert retry waits for the Eval-A asset writer to end. The two GPU streams each
remain limited to one worker.

Asset staging subsequently completed for all 197 starts, and the paired GPU
evaluation is running. Its first 37 matched cases completed without reported
comparison/runtime problems. The partial snapshot is baseline 28/37 ordinary
openings and 10/37 retained-grasp openings, versus pose-guided 27/37 and 6/37.
These counts describe a provisional prefix; read `policy/report.json` for live
coverage and results. No model/settings change is made using this snapshot.

## Original submitted workflow

| Job | Stage | Scope |
| --- | --- | --- |
| 6843326 | Test feature extraction and offline pose evaluation | 187 labeled trajectories, 748 observation–pose pairs |
| 6843327 | Resume source assets and freeze executable starts | All 197 released source trajectories, unchanged order |
| 6843328 | First complete matched episode | Source case 0; starts after asset staging |
| 6843329 | Remaining matched episodes | Cases 1–196, one worker at a time; starts only if the first episode completes without a runtime/comparison error |

The first episode uses the same full protocol as the others and counts toward
the 197. A physical failure is a measured outcome and does not block the other
cases. Runtime or matching errors fail the integration gate. At most two GPUs
are requested simultaneously: one for offline evaluation and one for rollouts.
Original expert replay remains unqueued. The subsequent independent
[Benchmark400 collection](../data_preparation/BENCHMARK400_RUN.md) is separate.
The original first rollout `6843328` also waited for asset job `6843421`.
The original paired jobs were cancelled after the staging failure; use the retry
jobs above for current progress.

Offline pose evaluation completed on all 748 observations from 187 trajectories:
mean position error 8.79 cm, mean orientation error 11.75 degrees, arm accuracy
97.99% and mean finger-span error 1.25 cm. These are held-out pose metrics;
matched simulator performance remains pending.

## Measurements

Offline evaluation measures position/orientation error, arm accuracy and finger
span error. It reports both observation-level and trajectory-averaged metrics;
four observations from a trajectory are not four independent episodes.

Policy evaluation runs frozen MolmoBot and the trained pose-guided method from
each of the same 197 starts: 394 rollouts, seed 42 and a 400-action budget. The
initial simulator state, images, conditioning and first policy proposal must
match. The learned method uses current-frame handle points and contact feedback;
those are additional inputs beyond the original baseline's initial point prompt.
No expert target pose or recorded expert action is passed to the pose-guided policy.

Ordinary door opening and onset-based force-qualified retained-grasp opening
are measured separately using the fixed `grasp.metrics.FINAL_POLICY`. Videos,
predictions, executed actions, contact traces and per-method summaries are saved.

All 197 source starts remain in the population, including ten without an early
pose label. Successful retained-grasp expert feasibility is not certified for
all of them. This evaluation must not be described as the separately proposed
400-case expert-qualified benchmark. Prior exposure flags in the source
inventory are preserved.

## Outputs and progress

Under `runs/pose_test197_20261108_01/`:

- `plan.json`, `submissions.json`, `source_manifest.json`: requested population,
  jobs, checkpoint identity and frozen sources.
- `test_features/metadata.json`: Test feature extraction progress/completeness.
- `offline_pose/report.json`: held-out pose metrics after completion.
- `episodes/policy_episode_manifest.json`: all 197 frozen starts after staging.
- `policy/report.json`, `policy/report.md`: current aggregate paired results.
- `policy/CASE_ID/{baseline,predicted_pose}/`: videos, traces and summaries.
- `logs/`: offline, staging and rollout scheduler logs.

The report retains pending cases and technical failures as unresolved. Partial
success rates are labeled provisional; a full-population rate is emitted only
when all 197 pairs are measured. The trained weights stay fixed throughout.

```bash
squeue -j 6858961,6858962,6858963
```

The proposed data expansion is described in
[the benchmark-expert extension plan](../data_preparation/EVALUATION_EXPANSION.md).

The subsequent [expert-reference oracle comparison](ORACLE_EVALUATION.md)
reruns all three methods on the 187 starts with usable references. It has its
own frozen run and keeps this evaluation unchanged. Proposed predictor changes
and the conditioning-stage error breakdown are recorded in
[the pose refinement plan](POSE_REFINEMENT_PLAN.md).
