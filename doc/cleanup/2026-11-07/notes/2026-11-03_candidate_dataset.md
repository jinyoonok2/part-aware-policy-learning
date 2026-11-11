# Complete candidate-data collection v1

The user authorized preparing all data for the original 120-episode course-project
cohort. The proposed incremental balancing redesign was withdrawn; the frozen
`configs/candidate_search_split_v1.json` is unchanged.

## Collection scope

40 houses, 120 starts: train 72 / dev 24 / eval 24. Collect both K=1 baseline and
K=4 simulator selection on every start (240 rollouts). Keep failures as data.
The existing checkpoint is frozen; no expert trajectory actions are replayed,
no critic is trained, and no original demonstration train/val files are changed.

Use the existing score weights: success 10, opening gain 5, final grasp .25,
clipped world-frame TCP approach gain 1. Predict 16 native actions; evaluate and
execute up to eight at a time; 400 actual-action cap. For dataset completeness,
baseline candidates also receive branch score labels at every decision, while
the baseline still selects its sole candidate. The cost of those labels is
recorded as evaluation overhead and is not an uninstrumented baseline runtime.

Every selected start has separate reference physics, canonical initial policy
inputs and initial point annotations. Both methods use these shared inputs to
remove the tiny fresh-render differences seen in the first pilot. Every search
checks that live integration state is untouched. Both methods check first-chunk
branch/live equivalence. The collection validator checks exact initial candidate-0
agreement, input/action finiteness, label coverage, score terms, step counts,
video frames, paired starts, and split eligibility before indexing a case.

## Storage and jobs

Dataset root:
`/bigtemp/xna8aw/part-aware-policy-learning/datasets/candidate_search_v1_20261103_01`.

- `split.json`: immutable copy of the existing selection.
- `study.json`, `source/`, `source-sha256.json`: collection parameters and frozen code.
- `assets/`: staged benchmark plus staging manifest; `STAGED` marks successful preparation.
- `{train,dev,eval}/episode_NNNN/`: paired rollouts, references and case verification.
- Per decision: RGB/proprioception NPZ, conditioning image, all candidate action
  chunks, candidate branch traces/scores/terms, seeds, task prompt and selection.
- `train_index.jsonl`: verified training-eligible candidate labels only.
- `dev_index.jsonl`: verified development labels, excluded from training.
- `eval_index.jsonl`: held-out addresses only; scores and success aggregates are
  omitted from the root report. Raw outputs are retained in eval directories.
- `index.html`, `aggregate.json`, `cases.csv`: current collection inventory.

Scheduler chain (authoritative IDs in `submissions.json`):

1. Asset staging **6795746**: only missing packages are fetched, existing LRL
   resources are read-only sources for project-owned copies.
2. One-case GPU integration/label-validation pilot **6795747**, `afterok` staging.
3. Full array **6795748**, `0-39%2`, `afterok` pilot, at most two GPUs concurrently.
   One model load per house, three start pairs per worker. Completed pilot data
   is verified/reused. No rerunning task failures to obtain successes.
4. CPU inventory report **6795764**, `afterany` collection.

Asset staging completed with exit 0: all 40 houses and exact 120 selected specs
were verified. The collection pilot is running on A40. Its baseline completed
successfully after 149 actions; both methods passed first-chunk branch/live
equivalence. Initial physics, RGB/proprioception, conditioning image and candidate
zero were checked to be bit-identical across methods, fixing the earlier pilot's
fresh-render discrepancy. Dev labels were verified as training-ineligible.
The full array remains gated until the pilot's complete paired data validation.

## Collection update and infrastructure recovery

The pilot passed and the original array finished: 116/120 paired starts are
verified (232 completed rollouts). Train has 70/72 starts and 9,295 verified
candidate labels; dev has 22/24 and 3,269 labels; eval has 24/24 and remains
excluded from training and top-level outcome reports. No critic is trained.

The remaining four starts (`train/episode_0189`, `train/episode_0190`,
`dev/episode_0837`, `dev/episode_0839`) failed before their first actual action
while starting the ffmpeg subprocess: `OSError: [Errno 12] Cannot allocate memory`.
Both affected house workers ran on nekomata02. Their Slurm measured peak RSS was
about 9–11 GiB, below the 64 GiB request, so this is not evidence of a policy
failure or a proven per-job RSS exhaustion. The precise node memory/commit
condition at the failure time is unavailable.

Only these zero-action failures were archived to `recovery/memory_01/attempts/`.
Original initial-state/canonical-input references are preserved. All 116 completed
case validations and summary hashes were captured and checked unchanged. Recovery
array **6798837**, `0-3%2`, uses a fresh process per episode, 96 GiB host memory,
and excludes nekomata02/RTX GPUs. It imports the original frozen collection code
without changes to physics, reward, sampling or episode split. Report **6798838**
runs after recovery. `submissions.json` and `aggregate.json` remain authoritative.

Incomplete/error directories are not overwritten on retries. Review and archive
explicitly before rerunning infrastructure errors; completed outcomes remain.
Invalid dependencies cancel the dependent rollout jobs. Collection errors remain
separate from policy failures. `dataset_ready` becomes true only when all 120
case validations exist with no collection errors.

## Using the data later

These are project-specific partitions of upstream validation *start specs*.
Critic training uses only newly generated branch labels from project train
houses, not official held-out expert demonstration actions. Use the train index
explicitly, never a recursive scan across all splits. Eval outputs must not be
used for critic fitting, reward tuning or model selection. Generating and storing
eval data now does not replace the learned critic's future independent closed-loop
evaluation from those same starts after training and configuration selection.

The existing one-case pilot succeeded with baseline 154 / simulator 144 actions.
This does not establish a success-rate improvement or prove sufficient critic
training coverage. Learning curves on dev will determine whether more train
houses/episodes are necessary; additions must keep held-out houses separate.
