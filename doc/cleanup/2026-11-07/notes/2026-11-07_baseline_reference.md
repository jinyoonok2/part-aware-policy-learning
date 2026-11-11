# Original MolmoBot baseline records

Verified November 7, 2026. Original pretrained MolmoBot results are already
retained. These development results establish an implementation reference;
final learned-pose performance still requires a matched held-out comparison.

## Current pose-pilot baseline

Ten original official Train development starts, seed 42 and a 400-action
budget. The baseline executes frozen native MolmoBot with no local grasp
interventions. Initial inputs and first policy proposals match the other
methods. Native stopping on door-angle success is disabled for every method
so grasp retention can be observed.

| Metric | Original MolmoBot | Grasp guard | Guard plus oracle pose |
| --- | ---: | ---: | ---: |
| Door-angle threshold reached | 9/10 | 10/10 | 8/10 |
| Retained-grasp opening | 2/10 | 3/10 | 4/10 |

The oracle column uses privileged expert poses, not a trained pose predictor.
Improved retained-grasp opening does not establish improved door opening
overall. These ten already-inspected Train starts cannot provide final
generalization evidence.

Evidence: `complete pilot report` (historical path: `runs/molmobot_oracle_pose_20261107_03/report.md`),
`frozen protocol` (historical path: `runs/molmobot_oracle_pose_20261107_03/study.json`),
[baseline CSV](2026-11-07_baseline_pose_pilot.csv) and
[provenance registry](2026-11-07_baseline_reference.json).
Each of the ten `cases/XX/baseline/` directories retains `summary.json`,
`head_camera.mp4`, executed-action events, both-arm contact traces and simulator
joint states. The CSV records episode/configuration and summary hashes; the
registry binds the checkpoint identity, source revisions and metric protocol.
Original outputs are preserved.

## Earlier selector-development baseline

The `earlier 24-start Dev comparison` (historical path: `runs/closed_loop_dev_20261103_01/aggregate.json`)
retains native task success of 16/24 (66.7%) for original MolmoBot. Its starts,
sampling rules and success stopping differ from the current pose pilot.
Use each result with its own protocol; do not pool these rates or compare
them directly with a learned model evaluated on other starts.

## Final performance comparison

Before final evaluation, freeze an untouched episode/seed manifest outside
training and checkpoint selection. Evaluate original pretrained MolmoBot,
guard-only control and the learned pose method on those same starts. Match
initial scene/robot state, original policy inputs, physics/controllers, action
budget and stopping rules. Freeze both success definitions before evaluating.

Save per-episode outcomes, actions, raw contact/state traces, videos and
checkpoint/code identities. Report both success metrics and paired gains and
losses on the common episode list. Current development baselines remain
available for debugging. The [final evaluation set and protocol](2026-11-07_final_pose_evaluation.md)
are now fixed: 200 starts / 100 houses / three seeds, with original-model
baseline array `6837253` queued behind development smoke `6837252`. It uses
the new onset-based metric and retains its own matched reference artifacts;
do not pool it with the development CSV above. The learned pose method has
not been trained or evaluated yet. No model training was submitted while
creating this inventory.
