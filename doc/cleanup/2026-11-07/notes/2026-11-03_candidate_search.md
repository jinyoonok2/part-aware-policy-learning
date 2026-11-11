# Simulator action candidate search development pilot

The course-project direction is now to compare action selection using simulator
branches with a learned evaluator later. No learned critic is implemented or
trained by this pilot. Part masks are not policy inputs.

## Fixed first implementation

Reuse the frozen MolmoBot RBY1 multitask checkpoint and the existing clean
house-0 door start from `baseline_robustness_20261102_01`, episode index 0.
This previously inspected start is development data, not final held-out eval.
No new dataset or model download is required.

- Baseline: sample one native 16-action chunk and execute its first eight actions.
- Simulator: sample four chunks, simulate each one's first eight actions from
  identical current state, choose the highest score, execute its first eight
  actions, then observe and repeat.
- Stop at native success, terminal/truncation, or 400 actual actions.
- Frozen native point prompt, conditioning image, gripper decoding and controllers.
- Per-decision seed `42 + 1000 * decision`; flow generator seed adds candidate
  index. All candidates share visual augmentation seeds and observations.
  Candidate 0 matches the baseline sampling rule at the same decision index;
  after selections differ, their states and observations naturally diverge.
- This controlled K=1 baseline is not an exact reproduction of the older smoke
  run's random stream. No performance improvement is assumed.

The score is

`10 * terminal_success + 5 * opening_gain + 0.25 * final_grasp + clipped_distance_gain`.

Distance gain is initial minus final minimum world-frame TCP distance to the
selected handle's body origin, clipped to [-1, 1] metres. Opening gain is final
minus initial normalized native door opening. Final grasp uses native two-finger
handle contacts. These heuristic weights are an initial development choice,
not learned/calibrated probabilities. No motion-safety or orientation objective
is claimed. Near-handle distance is only a proxy and short lookahead may miss
long-horizon benefits. A candidate stops early on native success. All methods
share the same task success definition and actual-action budget; their compute
budgets differ and generation/evaluation time is recorded separately.

## Isolation and verification

Each candidate gets an independent `copy.copy(MjData)` including derived data,
contacts, warm starts and integration state, a separate RBY1 robot/controller
instance, and a separate door task view. Mutable controller targets/stationary
flags are copied. The MuJoCo model is shared read-only. Branches invoke the same
update-control / control-step / physics-step loop as native `task.step`, without
camera rendering, sensor polling, live caches or live episode counters.

The implementation explicitly rejects non-door tasks, multiple environments,
action noise, dynamic arm clipping and unsupported controller types. The first
selected chunk in each method is checked against actual `task.step`: qpos, qvel,
control inputs, simulation time, opening, grasp, success and TCP distance must
match. Every search verifies the live integration state is unchanged by candidate
evaluation. Nonfinite actions, physics or scores are execution errors, not task
failures. Source/checkpoint identities are checked before loading.

## Outputs and next experiment

Run directory: `/bigtemp/xna8aw/part-aware-policy-learning/runs/candidate_search_20261103_01`.
Initial GPU submission: **6795224**, one GPU, four CPUs, 64 GB host memory,
one-hour limit. `submissions.json` and per-method status files are authoritative.

Initial runtime validation passed for both methods: the first eight actual
actions match their independent branch predictions, and candidate evaluation
leaves the live integration state unchanged. Baseline completed successfully
after 154 actions (155 video frames, maximum opening 0.6771). The simulator
selector is running; its first scores were 0.1130, 0.1245, 0.1231, 0.1345 and it
selected candidate 3 (zero-based). This is implementation evidence on one start,
not evidence of improved success rate.

`integration_validation.json` records identical starting physics, proprioception,
point annotations and GPU. Fresh scene renders have tiny rounding differences
(at most one uint8 level), so matching flow seeds do not produce bit-identical
candidate-0 actions across the two methods (maximum action difference 0.00112).
Within each candidate set, the observation and augmentation are shared. Preserve
this pilot as measured; use shared saved initial RGB if a later experiment
requires bit-identical initial policy inputs across methods.

The live `index.html`, `aggregate.json` and `cases.csv` distinguish pending,
execution errors and complete policy outcomes. Per-method videos and actual
action events are separate from per-decision branch records. Decision records
save raw RGB/proprioception observations, native conditioning image, candidate
action matrices and their group layout, seeds, goal, scores, component terms,
branch traces, selection and timing. These are development branch labels for
possible future evaluator training; privileged labels must not be confused with
deployable critic inputs. Actual actions alone are recorded in the rollout video.

Do not silently overwrite interrupted runs or rerun policy failures to obtain a
success. Use a new run ID for changed code/configuration. After this pilot passes,
choose train/dev/final-eval episode groups before expanding data collection.
Both future simulator and critic methods may use the same split. Ranking
agreement can be compared on common candidate sets; task success requires
independent closed-loop execution from common evaluation starts.

The proposed full course-project cohort is now selected in
`configs/candidate_search_split_v1.json` (historical path: `configs/candidate_search_split_v1.json`):
40 disjoint houses, three entries each, split into train 72 / dev 24 / eval 24.
All previously inspected houses/entries are in dev. This is a project-specific
split of upstream val start specs; no official expert demonstration actions are
used for critic training. Branch labels would be generated anew on project-train
houses. Selection does not stage assets or submit batch jobs. See
`configs/README.md` (historical path: `configs/README.md`) for the staged execution plan.
