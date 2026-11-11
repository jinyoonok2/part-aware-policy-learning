# Source-backed expert trajectory validation pilot

This is a development diagnostic using three available official Val expert
demonstrations. It is separate from the stopped 200-task benchmark candidate
set. No task in that candidate set is certified by this pilot. No MolmoBot
inference, pose-model training or automatic final evaluation resume occurs.

The earlier work completed source-availability checks but did not launch a
physical expert-validation job. This pilot addresses that missing step.

## Source cases and staging

Run: `/bigtemp/xna8aw/part-aware-policy-learning/runs/expert_validation_pilot_20261107_01`.
The source files already exist under the prepared official data; no replacement
model trajectory is used as ground truth. Selection is deterministic, using
source order and fixed task/arm categories, excluding the prior candidate-study
houses. These three houses become development cases and are excluded from a
future final test set.

| Case | House | Source trajectory | Source observations | Task / active arm |
| --- | ---: | --- | ---: | --- |
| 00 | 204 | `traj_0` | 151 | Push / left |
| 01 | 7 | `traj_0` | 183 | Pull / right |
| 02 | 397 | `traj_0` | 132 | Push / right |

Staging matches the exact door and all frozen object identities against
ProcTHOR-Objaverse, ProcTHOR-10k and Holodeck-Objaverse Val scene candidates.
Exactly one match is required. Numeric house equality alone is insufficient.
HDF5, configuration, episode and matched scene XML identities are retained.
Original Train/Val indices, raw files and the old 200-task manifest are unchanged.

## Declared physical diagnostics

Replay `actions/commanded_action` starting at frame 1. Frame 0 is observation
padding; `actions/joint_pos[0]` can contain stale targets and is not an initial
robot state. Logged command frame k is aligned with source observation k.
The live robot and door are advanced through controllers and MuJoCo, never
teleported along the source state sequence.

Two explicitly distinct modes use a fixed audit seed of 42:

- `evaluation_noise_disabled`: replay recorded commands with noise disabled,
  matching the intended noise condition for later model evaluation.
- `source_noise_distribution`: restore the recorded noise parameters after the
  JSON sampler's initialization, then sample a new noise realization. The
  original RNG/applied-noise sequence is unavailable; this is not exact replay.

Source policy timing is recorded as 100 ms. Current pinned control/physics
defaults are 20/4 ms; these were not persisted in the original frozen config
and are recorded as compatibility assumptions. Source XML, gravity compensation,
gains and absolute command modes are restored. Native model relative-command
and torso-height equivalence is a separate check before final certification.

The original source recordings end at their first ordinary task success, so
they do not contain the final metric's ten-observation success-hold segment.
Every case/mode therefore has the same preregistered ten-step repetition of
its final recorded command after the original prefix. Original-prefix and
extended measurements are scored separately. The extension is newly executed
diagnostic evidence, not original recorded ground truth. No gripper command,
door state or target is silently replaced to force success. All three prefixes
plus continuation fit within a 400-action budget.

## Measurements and interpretation

Measure both arms with frozen `FINAL_POLICY` (`firm_grasp_opening_v3_onset`):
force-qualified onset before major opening, continued contact and bounded
handle-relative slip, opening progress and the final contact hold. Save source
pose-tracking errors, initial alignment, both-arm traces, actual post-noise
controller targets, each control substep's actuator controls, qpos and video.
Native environment stepping refreshes mounted-camera transforms and object
caches, in addition to advancing the same MuJoCo physics.

A passing physical execution supplies a newly measured witness under its
declared replay conditions. A failure, unavailable hold coverage, tracking
drift or compatibility mismatch does not establish task impossibility. Every
pilot result remains `final_eligible: false` until control/physics equivalence,
reference provenance and final data splitting have been reviewed. Original
source grasp flags are not bilateral-force ground truth.

The first case tests integration. Remaining cases can follow only after that
worker completes without an infrastructure error; that dependency does not
require task success and does not select cases by outcome. Final policy
evaluation stays stopped until the independent expert-eligibility gate passes.

## Submission and checks

Asset staging `6837519` completed with one exact scene-family match for each
case. Integration worker `6837522` is submitted after staging; remaining workers
`6837524` (array 1–2, one at a time) depend on integration completion. Report
`6837525` runs after the remainder terminates. These dependencies never require
a positive task outcome. Live case states and results are saved in the run's
`status.json` and `report.md`.

The replay runtime was frozen at revision 2 before GPU execution. The revision
uses native environment stepping to refresh mounted cameras and cached object
states. Source snapshots and their hashes are preserved. Nine focused tests
passed, covering source selection, noise modes, action-frame alignment,
continuation coverage and physical control stepping. Tests do not establish
task feasibility; the measured replay does that only under its declared
conditions.

Runtime review found that `initial_alignment.base_pose_error` compares two
source records rather than the live robot base. It is not an eligibility gate
and must not be interpreted as actual base alignment. Live base poses are
retained in both-arm traces, and active-TCP/handle alignment gates are applied.
Any future certification must validate live base alignment as well.
