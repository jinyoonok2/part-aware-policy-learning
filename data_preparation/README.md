# Dataset preparation

This package prepares source-backed expert data. It does not train a model or
submit simulation jobs. `grasp/` contains the retained controller and shared
measurement helpers; `pose_learning/` contains learned-model work.

## Retained data

The working dataset is `data/interaction_pose_train_test_v1_20261107`:

- Official Train: 5,004 general pose references, 4,866 early-grasp references,
  and 19,464 observation/target pairs.
- Official Val used as project Test: 197 source expert trajectories;
  192 general and 187 early-grasp pose references.
- Separate Benchmark400 v1: 400 fixed starts across 200 houses; native-expert
  recording completeness and pose-label coverage are tracked in
  [the run record](BENCHMARK400_RUN.md).
- A source-success flag or pose-label eligibility does not certify continuous
  force-qualified grasping. Existing Test cases remain present and carry that
  qualification status.

Raw data are shared through paths and symlinks. The current Train/Test manifests
depend on `interaction_pose_v2_5000_20261107`, its parent
`interaction_pose_v1_20261106`, and the v2 early-quality view. The released Test
raw files remain under the shared official dataset directory. These are source
dependencies, not independent duplicate datasets.

## Train preparation

Run commands from the project root with the project Python environment:

```bash
PY=/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python
STORE=/bigtemp/xna8aw/part-aware-policy-learning
```

The existing prepared data can be used immediately. For a fresh subset, the
collector consumes the pinned package catalogs already retained with v1:

```bash
"$PY" -m data_preparation.collect_interaction_pose_dataset --prepare \
  --run "$STORE/datasets/train_fresh" \
  --config data_preparation/configs/train_subset.json \
  --metadata-dir "$STORE/datasets/interaction_pose_v1_20261106/metadata"
"$PY" "$STORE/datasets/train_fresh/source/collect_interaction_pose_dataset.py" \
  --run "$STORE/datasets/train_fresh"
"$PY" -m data_preparation.audit_interaction_pose_dataset \
  --run "$STORE/datasets/train_fresh"
```

Collection downloads only the selected house packages. It freezes the source
revision, metadata, house order and Test-house reservations before downloading.
The target is a minimum count, so complete houses can produce a small overshoot.
`expand_interaction_pose_dataset` extends an existing subset under a fixed
budget. `build_early_interaction_pose_view` selects early grasp events and
rebuilds observation-relative pose targets. `prepare_train_test_dataset` creates
the Train/Test aliases, preserves original source roles, and audits state/video
alignment. Use each module's `--help` for its prepare/build/collect commands.

## Active separate benchmark cohort

The current design keeps the original 197 demonstrations and an independent
400-start cohort from the 2,000-episode benchmark. Existing Test and Train houses
are excluded when planning the new cohort. Do not merge the original 197 into
the 400 or select cases using policy success.
[The design](EVALUATION_EXPANSION.md) describes the full recording contract and
pilot gate followed by bulk collection.

## Plan and stage a separate benchmark cohort

The planner defaults to an independent fixed cohort. The optional `--standalone`
flag is accepted for existing command lines; the old merged-cohort mode has been
removed. The original Val197 manifest supplies house exclusions and stays intact.

```bash
"$PY" -m data_preparation.prepare_test_set plan \
  --run "$STORE/datasets/benchmark_eval400_fresh" \
  --benchmark /bigtemp/xna8aw/LRL/cache/molmo-spaces-resources/mujoco/benchmarks/molmospaces-bench-v2/20260327/procthor-10k/rby1_benchmarks/door_opening_benchmark/benchmark.json \
  --total 400 --candidate-houses 200
```

Planning selects one push and one pull start per house with seed 6198, excluding
prepared Train and existing Test houses. It freezes EpisodeSpecs and executable
source dependencies before expert outcomes are observed. Use a fresh output path;
the retained Benchmark400 collection already has a fixed manifest.

Stage assets separately, then run the expert worker with an allocated GPU:

```bash
"$PY" "$STORE/datasets/benchmark_eval400_fresh/source/data_preparation/prepare_test_set.py" \
  stage --run "$STORE/datasets/benchmark_eval400_fresh" --batch 1
"$PY" "$STORE/datasets/benchmark_eval400_fresh/source/verify_benchmark_expert.py" \
  worker --run "$STORE/datasets/benchmark_eval400_fresh" --index 0
"$PY" "$STORE/datasets/benchmark_eval400_fresh/source/report_benchmark_expert_verification.py" \
  --run "$STORE/datasets/benchmark_eval400_fresh"
```

`slurm/` contains matching CPU/GPU entry points. Planning and reporting do not
submit jobs. Staging one batch does not stage or execute later batches. The
recording gate reviews the first 20 cases before bulk execution. Failed experts
remain in the task-performance population; successful expert witnesses and
usable pose labels are separate coverage measures.

Before expanding Train again, register new Test houses and manifest hashes in
`configs/final_evaluation_reservations.json`. Retain historical reservations;
numeric house-ID separation alone does not prove geometry uniqueness.

Planner build receipts remain in `fixtures/planner_build/`. Fresh plans freeze
those receipts and their executable dependencies, so queued workers can use
immutable source copies rather than the live workspace.

## Export starts for the learned-policy comparison

The 197-case source inventory contains HDF5 references, not executable benchmark
starts. Export explicit initial-state EpisodeSpecs before online evaluation:

```bash
"$PY" -m data_preparation.export_policy_episodes \
  --output "$STORE/datasets/test_policy_episodes"
```

The exporter reads the frozen initial robot, base, door, cameras and object
configuration. It matches the door and all object identities to a scene XML;
it does not guess a scene family from the house number. Missing or ambiguous
scenes produce a pending ledger and block the full runnable manifest. The
optional `--stage-scenes` flag permits CPU scene downloads. An explicit
`--allow-partial --limit N` export is a diagnostic subset only.

`policy_episode_manifest.json` can then be passed to
`pose_learning.evaluate_policy --episodes`. Its EpisodeSpecs contain no expert
action sequence or target grasp pose. Source/exposure information remains in
the external manifest. Exporting starts does not certify strict grasp success
or live initial-state/controller equivalence.

## Export new expert pose labels

`export_benchmark_pose_labels` exports the recorded Benchmark400 observations
through the same early-grasp selection and conditioning-base target convention
used by the trained head. New recordings use `benchmark_expert_witness_v2`;
source-phase/closure targets and force-qualified grasp evidence stay separate.
The run's `collection_inventory.json` and `collection_status.json` report
completeness and label coverage. Missing labels do not remove task cases.

The historical finalizer that combined qualified existing cases with successful
additions has been retired. The current protocol evaluates Val197 and
Benchmark400 independently and preserves their fixed memberships.

## Checks

```bash
"$PY" -m unittest discover -s data_preparation -p 'test_*.py'
```

The expansion fault-injection suite requires an explicitly supplied disposable
`/tmp` preparation. It never mutates the retained source datasets.

## Historical runs

November 7's original-expert replay and combined 400-total plans are retained
in their frozen run directories. They are historical records, not the current
Benchmark400 procedure. The source replay/staging helpers remain available
for reproducibility and scene preparation. See
[the cleanup record](../doc/cleanup/2026-11-07/README.md) for old paths.

The 197 released demonstrations are the complete official Val inventory at
revision `159a5aecaf2ba73855940ad325b477b27d85da73`. They are neither a random
sample nor benchmark episodes numbered 1 through 197. Original source split,
trajectory identities and exposure ledgers remain attached to each case.
