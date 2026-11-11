# Independent Benchmark400 v1 collection

Dataset: `data/benchmark_eval400_v1_20261108`.
Run alias: `runs/benchmark_eval400_collection_20261108_01`.
The original `interaction_pose_train_test_v1_20261107` dataset stays intact.

This cohort contains 400 frozen episode specifications from the downloaded
2,000-episode door-opening benchmark: 200 houses, 200 push and 200 pull tasks,
selection seed 6198. Prepared Train houses and existing Test houses are excluded.
No original 197-case demonstration is included. House reservations are registered
for future Train collection. No model outcome selects membership.

## November 10 planner-termination recovery

The first 20 native attempts reached the recording pilot. Seven finalized:
three demonstrated retained-grasp opening and four did not. Thirteen stopped
with the native planner's explicit `Max planning reattempts reached.` error
during the opening phase. Their videos and partial state/control traces were
saved, but the collector did not finalize the full HDF5/label bundle.

Runtime revision 7 treats only that specific planner exhaustion as a normal
expert termination and saves the measured prefix through its last completed
action. Other exceptions remain technical errors. Planner settings, action
budget, success metrics and selected starts are unchanged. Twelve focused CPU
checks passed. This change does not improve the expert's physical performance;
it makes failed attempts usable as complete recordings.

The seven completed recordings are preserved with unchanged summary/artifact
hashes. Only the thirteen incomplete pilot cases are retried. Their previous
attempts and revision-6 source/study are preserved under
`audit/recovery_20261110_02/`. The 197-start policy comparison is unchanged.

| Current job | Stage |
| --- | --- |
| 6862240 | Retry pilot case 1 with terminal recording |
| 6862241 | Retry the other twelve incomplete pilot cases, one at a time |
| 6862242 | Recheck the same 20-case recording/label pilot |
| 6862243 | Collect cases 20–399 only after the recording gate passes |
| 6862244 | Export coverage and the complete eligible pose index |

`retry_submissions_20261110_02.json` records the current commands and unchanged
population. A completed failed expert attempt remains in the 400; it does not
certify successful grasping, provide a pose label automatically, or establish
that the task is impossible. The new GPU terminal-recording path still needs
its first runtime validation.

## November 10 configuration-export recovery

All four asset batches completed. The first native expert attempt verified its
initial alignment, then failed while writing `resolved_config.json`: the native
configuration contains infinity for unbounded settings, which strict JSON
rejects. It executed no physical actions. This is a recording/setup error,
not an expert task failure. The remaining attempts did not start.

Runtime revision 6 represents configuration infinities as explicit `Infinity`
and `-Infinity` strings in the saved snapshot. Live controller settings are
unchanged; measurements and metrics still reject nonfinite numbers. Eleven
focused CPU checks passed, including the fix and the existing observation/label
contract. The previous source, study, error records and pilot review are
preserved under `audit/recovery_20261110_01/`; frozen source hashes were updated
and rechecked. All 400 EpisodeSpecs and the selection plan retain their hashes.

| Retry job | Stage |
| --- | --- |
| 6858964 | First native expert, after Eval-A asset writer 6858961 ends |
| 6858965 | Cases 1–19, after first expert completes without a runtime error |
| 6858966 | Review the 20-case recording/label contract |
| 6858967 | Cases 20–399, after pilot review passes |
| 6858968 | Export inventory and complete pose index |

Commands are recorded in `retry_submissions_20261110_01.json`. The original
checkpoint, collection settings and fixed episode population are unchanged.
The subsequent native pilot exercised this configuration-export repair; the
planner-termination recovery above supersedes these job IDs for current work.

## Original submitted jobs

| Job | Stage | Dependency |
| --- | --- | --- |
| 6843418 | Stage first 100 starts | Existing Eval-A asset writer ends |
| 6843419 | Stage starts 100–199 | Previous staging succeeds |
| 6843420 | Stage starts 200–299 | Previous staging succeeds |
| 6843421 | Stage starts 300–399 | Previous staging succeeds |
| 6843422 | Native expert on case 0 | All new asset staging succeeds |
| 6843423 | Native experts on cases 1–19 | First case completes without a runtime error |
| 6843424 | Validate the 20-case recording/label pilot | Pilot attempts finish |
| 6843425 | Native experts on cases 20–399 | Pilot review passes |
| 6843426 | Export inventory and complete pose index | Remaining attempts finish |

One expert GPU is active at a time. The existing Eval-A comparison uses at most
one policy GPU, so the two streams use at most two GPUs. The original Eval-A
first-rollout job `6843328` also waited for `6843421`. Those submissions stopped;
use the retry jobs above for current progress.

The pilot cases count toward the same 400. Physical failures remain cases.
Technical errors block the compatibility gate and are recorded as unknown.
The review checks aligned videos, state, camera parameters, handle points and
the existing pose-loader contract. At least one usable source-phase pose target
is required; there is no minimum retained-grasp success-rate threshold and no
selection using learned-policy outcomes.

## Data and readiness

The implemented `benchmark_expert_witness_v2` format saves three RGB videos,
commands and applied controls, robot qpos/qvel, simulator state, timestamps,
camera calibration, current visible handle points, both TCP/base poses, expert
phases, grasp flags and contact/force evidence. The pose adapter uses the same
early source-phase/closure rule and conditioning-base target convention as the
trained head. Force-qualified grasp poses are recorded separately.

The 400 starts are prepared; expert trajectories are not completed yet.
Read readiness and label coverage from the output files:

- `study.json`, `selection_plan.json`: fixed membership, settings and source hashes.
- `policy_episode_manifest.json`: immutable identity manifest for all 400 starts.
- `cohort_comparison.json`: separate Eval-A/Eval-B counts and source identities.
- `submissions.json`, `collection_submission.json`: jobs and collection intent.
- `status.json`, `report.md`, `index.html`: expert progress, failures and unknowns.
- `pilot_review.json`: the native recording/label gate after 20 cases.
- `collection_status.json`, `collection_inventory.json`: final completeness and coverage.
- `test_early_pose_targets.jsonl`: common-format index, emitted after all 400
  expert records complete and eligible labels validate.
- `cases/episode_NNNN/`: EpisodeSpec, expert HDF5, videos, commands, forces,
  pose-label status, optional target and measured grasp references.

The primary task-performance population stays fixed at 400. Pose-error metrics
use legitimate labels and report their own denominator. Do not describe 400
selected starts as 400 successful or force-certified demonstrations.

Later, evaluate the same baseline and step-20,000 pose checkpoint on both cohorts
using seed 42, a 400-action budget and the same opening/retention metrics. Report
paired gains within each set. Differences between sets also reflect their task
distributions. Model evaluation on the new 400 has not been submitted yet.

```bash
squeue -j 6862240,6862241,6862242,6862243,6862244
```
