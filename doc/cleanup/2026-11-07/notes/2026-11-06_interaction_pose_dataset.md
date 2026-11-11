# Official expert subset for interaction-pose learning

The user authorized an initial official Train subset, retaining existing Val for
validation, with expansion to thousands later. This direction predicts an
interaction pose from point/part geometry and makes the approach reach it. Data
preparation is separate from the previous evaluator experiment. No model is
trained by this collection job.

## Source and selection

Dataset: `allenai/molmobot-data`, `DoorOpeningDataGenConfig`, pinned revision
`159a5aecaf2ba73855940ad325b477b27d85da73`. Only official Train packages enter
the training subset. The target is at least 500 eligible successful trajectories,
stopping after a whole house. First audit at least 20 eligible trajectories.

House order sorts SHA256 of `pose-data-v1:20261106:HOUSE`; all available parts of
each selected house are included. Empty archives are excluded from the source
inventory. The full order of 15,205 nonempty Train houses is preserved for later
expansion without replacing the initial subset. Initial limits are 180 houses
and 12 GB compressed transfer. Selection does not use a learned model's outcomes;
invalid demonstrations and unsupported events remain in the exclusion ledger.

Download individual house archives by exact HTTP byte range from the published
index. Check the returned range and byte count, save compressed-payload SHA256,
and extract regular files/directories with path checks. Whole multi-GB shards
are unnecessary. These checks do not verify a complete upstream shard checksum.

Data directory:
`/bigtemp/xna8aw/part-aware-policy-learning/datasets/interaction_pose_v1_20261106`.
`study.json`, metadata, source hashes and `house_order.json` freeze the selection.

## Offline pose references

Require the released valid-trajectory mask, recorded success, matching
observation/action lengths, 100 ms timestep and required pose/phase fields. For
each contiguous `GRASP_HANDLE` segment before first success, identify the arm
with the clearest measured gripper closure. Retain events whose final three
recorded grasp flags are true and whose closure proxy exceeds 5 mm. Select the
last eligible grasp event before task success as the current pose reference;
keep earlier eligible events to expose re-grasp/recovery behavior.

Save measured TCP pose, raw gripper joint positions, robot base pose, arm, phase
indices, handle origin and source commanded pose. Pose vectors are `xyz, wxyz`.
Measured TCP is robot-base relative; transform it with
`T_world_tcp = T_world_base @ T_base_tcp`. At four frames in the corresponding
approach, express that fixed world target in the current observation's base
frame. Keep source timing explicit; commanded poses are not measured TCP states.
This RBY1 `xyz, wxyz` layout differs from the pose-vector layout in Part2Action;
shared geometry ideas require explicit format/controller conversion.

Video frame counts match observation counts for every retained pose example.
For the first 20, additionally decode three cameras at approach/target/success
and save head-camera contact sheets. HDF5/video content is preserved. No opaque
pickled `frozen_config` is executed.

These are offline expert references, not replay-validated grasp labels. Contact
flags can precede the named grasp phase, and finger contact alone does not prove
a robust grasp. Gripper joint sums are a closure proxy, not calibrated physical
aperture. Handle-relative translations are available, but a full part frame
requires handle orientation from matched assets/replay. Scene/robot versions
still need native replay validation. Pixel part masks are not present in the
inspected files; reconstruction/segmentation is a subsequent step.

## Validation, pilot and outputs

The existing official Val directory stays at
`/p/liverobotics/molmobot_molmospace/dataset/molmobot_official/val/DoorOpeningDataGenConfig/part0`.
Its index contains 197 valid trajectories. The frozen index hash is checked at
every report update. Val examples never enter either training index.

The same frozen numerical event rule was applied read-only to all 197 Val
trajectories. It produced 192 eligible pose references and 768 conditioning
examples. Five trajectories lack the required stable-grasp/clear-closure event;
they remain in the validation inventory with reasons. This is pose-label
coverage, not a task-success evaluation or permission to drop those cases from
a future full-task benchmark. `val_trajectories.jsonl`, `val_pose_targets.jsonl`
and `validation_status.json` stay separate from Train. `validation_reference/`
links to the unchanged official source directory.

The pilot completed with 23 eligible trajectories from five houses (six house
packages), 92 conditioning examples and 20 detailed three-camera audits. It has
13 push / 10 pull tasks and 8 left / 15 right active-arm events, with no pilot
exclusions. TCP-to-handle-origin distances span 0.052–0.099 m. One demonstration
includes re-grasping. The pilot checks establish offline extraction feasibility,
not physical control performance. Initial compressed transfer: 268,458,901 bytes.

CPU collection job **6833535** completed successfully in 9m27s with two CPUs and
8 GB memory. The initial subset contains **503 eligible pose references** from
**98 training houses**, with 2,012 conditioning examples. Whole-house selection
accounts for the slight excess over 500. There are 512 inspected source
trajectories: 510 satisfy the source valid/success checks; seven of those lack
the required pose event. Two other sources have an invalid mask or no recorded
success. All remain in the raw data and inventory. The retained pose examples
contain 228 push / 275 pull tasks and 176 left / 327 right active-arm events.

Compressed transfer was **5,417,937,971 bytes (5.42 GB)** across 110 packages.
The raw files occupy approximately 5.4 GiB. The final integrity audit passed:
224 unique HDF5 file hashes and 2,780 conditioning transforms checked, no
duplicate source trajectories, no Train/Val source overlap and the validation
index unchanged. Pose-transform checks used 1e-6 tolerance. This remains an
offline data result; physical replay is unvalidated and no part masks or model
weights were generated. `submissions.json`, `status.json`,
`audit/dataset_integrity.json` and the scheduler record are authoritative.

- `raw/partN/train/house_N/`: selected original demonstrations/videos.
- `packages/`: range/size/SHA256 provenance.
- `houses/`: resumable inventories, pose references and exclusions.
- `train_trajectories.jsonl`: all inspected Train records, including exclusions.
- `pose_targets.jsonl`: eligible Train references and conditioning frames.
- `val_pose_targets.jsonl`: separate validation pose references.
- `audit/previews/`: first-20 approach/grasp/target/success contact sheets.
- `status.json`, `index.html`: progress, task/arm balance and label limitations.
- `audit/dataset_integrity.json`: final source-hash, split and pose-transform checks.

The project-local `data/` link exposes the same directory through the workspace;
large demonstration files remain on `/bigtemp` and outside version control.

Preparation and continuation:

```bash
python scripts/collect_interaction_pose_dataset.py --run DATASET \
  --prepare --config configs/interaction_pose_dataset_v1.json
python DATASET/source/collect_interaction_pose_dataset.py --run DATASET --pilot-only
python scripts/prepare_interaction_pose_validation.py --run DATASET
PART_POSE_DATA_RUN=DATASET sbatch --export=ALL scripts/slurm/collect_interaction_pose_dataset.sh
```

Later expansion keeps existing houses and uses the next entries in the frozen
order, with an explicitly enlarged target/budget. Before model training, resolve
replay compatibility, event-label reliability and part-frame/mask construction.

## Queued native replay and supplied-pose audit

Run: `/bigtemp/xna8aw/part-aware-policy-learning/runs/interaction_pose_replay_20261106_01`.
CPU staging job `6833576`, ten-case GPU array `6833577` (at most two concurrent),
and dependent report job `6833578` were submitted on November 6. Job IDs and the
frozen protocol are saved in `submissions.json` and `study.json`; the current
report is `index.html` / `report.md`. A queued submission is not a completed result.

Ten Train cases from distinct houses are fixed before replay outcomes, cycling
through push/left, pull/right, push/right and pull/left. Restricted config decoding
never imports source pickle classes. Staging requires a unique scene matching
the exact recorded door and frozen object names, then installs only the selected
scene dependencies into this project's resource cache. Existing source caches
and HDF5 files remain read-only.

Native replay uses recorded absolute commands, 100/20/4 ms policy/control/physics
timesteps, and disabled replay action noise. Since expert collection enabled
noise, exact action replay is not promised. Initial handle/TCP alignment and
pregrasp tracking errors gate the supplied-pose test. That test starts an
independent branch at the replay pregrasp state, holds base/torso, solves arm IK
for the expert world-frame TCP and physically executes bounded joint commands,
then applies recorded gripper closure. It measures pose error and grasp flags;
it is a privileged approach/contact diagnostic, not a complete learned policy
or door-opening success comparison. Asset errors and alignment mismatches are
reported separately from replay outcomes. No training job is launched by this
audit; review the evidence before fitting the pose predictor.

The replay audit subsequently completed all ten cases. Nine expert-action
replays reached door-opening success. All ten supplied-pose attempts solved IK
and reached the expert TCP with final position error below 4.7 mm; five ended
with the benchmark's bilateral handle-contact flag. These are ten predetermined
Train pilot cases, not a learned-policy evaluation. No model has been trained.

A contact follow-up is queued in
`/bigtemp/xna8aw/part-aware-policy-learning/runs/interaction_pose_contacts_20261106_01`
(GPU array `6833811`, report `6833812`). It keeps all ten original cases and
clones the same pregrasp physics/controllers for each variant. Compare a short
expert continuation with the fixed expert pose using immediate closure, closure
after 0.5 seconds, and the original six-second delay. All fixed-pose variants run
for 12 seconds, so the onset of closure is the controlled difference. Report
early and final contact separately; prolonged static holding is not a complete
door-opening policy. Raw finger/body contact pairs, normal forces, aperture,
TCP errors, base state and handle motion are logged, with exact and descendant
contact checks. Synthetic MuJoCo checks passed bilateral-contact detection,
single-finger rejection and detection of child-body contacts that a strict body
name check would miss. Each worker verifies expert-continuation agreement with
the original replay and records whether the delayed variant reproduces its
original endpoint. Timing is a hypothesis until these matched results finish.

The contact follow-up completed all ten cases. Final bilateral contact counts
were 5/10 with the six-second delay, 6/10 with a 0.5-second delay, and 7/10 with
immediate closure. The short expert continuation also produced contact in 7/10.
The original delayed outcomes reproduced for every case. No grasp detections
were rescued by descendant-body matching, so body-name strictness did not
explain these failures. These counts measure contact, not door-opening success.

Cases 04, 08 and 09 lack handle contact even in the short native expert replay.
Their door states have drifted from the recording by the selected event: source
versus native opening fractions at the pregrasp start are approximately
27% versus 95%, 30% versus 87%, and 18% versus 12%. The robot TCP still tracks the
recording, so a fixed world-frame pose can miss the current handle. This is
evidence of a scene-state mismatch at the event, not proof the source expert
demonstration was unsuccessful. Keep all cases in the diagnostic denominator.

The user authorized a handle-relative control pilot in
`/bigtemp/xna8aw/part-aware-policy-learning/runs/interaction_pose_servo_20261106_01`.
GPU array `6833944` and dependent report `6833945` compare the same ten starts
using the fixed-world immediate-close baseline, a handle-relative arm servo,
and the same servo with bounded base XY motion (at most 0.6 m commanded displacement;
base yaw and torso held). The routine uses a damped Jacobian servo, closes near
the target, verifies persistent bilateral contact, and permits two deterministic
5 mm tool-axis corrections with reopening. All branches have 120 control
decisions at 100 ms each. Motion limits, relative-frame equivariance, singular
Jacobian behavior, retry exhaustion and persistent-contact verification pass
four focused tests in `scripts/test_interaction_pose_servo.py`.

For label conversion only, a disposable FK data copy is placed at the recorded
door hinge angle. Its reconstructed visual handle position must agree with the
recorded position within 5 mm before deriving the full handle-relative TCP
transform. Live branches receive bounded control commands and are not placed at
recorded robot or door states. Expert TCP labels, recorded hinge angles and live
simulator handle/contact information make this a privileged control audit.
Results do not establish learned perception or full-task performance. Source
files remain unchanged, no model is trained, and the fixed-world baseline must
reproduce the previous contact result in each completed case.

The handle-relative pilot completed: fixed-world immediate closing,
handle-relative arm control, and handle-relative arm/base control each ended
with contact in 7/10 cases. The relative controllers recovered case 09 but lost
case 01, which remained about 16--17 mm from its goal and never passed the
12 mm closure gate. Cases 04 and 08 remained in approach; mobile control reached
the 0.6 m command limit. All seven contact cases in each relative variant reached
the sustained verification state. No learned-model improvement is established.

The user authorized testing both proposed fixes, separately and together. Run
`/bigtemp/xna8aw/part-aware-policy-learning/runs/interaction_pose_recovery_20261106_01`
uses GPU array `6834175` and report job `6834176`. Four variants retain the same
ten Train starts, the same 120 decisions/12-second budget, and matched initial
physics: reference 12 mm/mobile controller; a 20 mm closure gate only; stalled
base recovery only; and both changes. The 5-degree orientation gate, bilateral
contact verification and grasp retry limits remain the same. The wider gate is
not a relaxed definition of success.

Base recovery activates while approaching with more than 4 cm TCP error after
a 15-step progress stall, or upon reaching the normal 0.6 m base travel limit.
Each plan follows the horizontal remaining TCP error for at most 25 cm, with at
most two plans. Normal DLS base travel remains capped at 0.6 m until recovery
activates; then total commanded displacement may reach 0.85 m. Base speed is
bounded, yaw and torso remain fixed, and recovery ends when closing begins.
These are local simulated repositioning moves, not collision-free navigation.

Eight focused tests pass: the original frame/control contracts plus the 20 mm
gate and orientation guard, bounded base plans, stall detection, and avoiding
recovery during contact or normal progress. The reference must reproduce the
previous contact outcome, final controller phase and TCP error within 1e-8 m in
each case. Frozen code, parameters, source data hashes, per-step traces and
recovery events are saved with the experiment. Results remain privileged contact
diagnostics; pose-model training and full door-opening evaluation are separate.

The four-way simulations finished for all ten cases and all 40 branches. Seven
workers passed the original audit; cases 02, 03 and 05 exited with an assertion
after saving all four complete traces because the reference final TCP-error
agreement exceeded the overly strict 1e-8 m criterion. Their reference grasp
flags and controller phases matched the earlier run at every step. Maximum TCP
position deviation was about 0.023 mm, orientation deviation 0.0213 degrees and
base XY deviation 0.050 mm. These are numerical reproduction differences, not
missing simulations or failed grasps.

`scripts/review_interaction_pose_recovery.py` performed a separate, explicitly
post-hoc review without altering original statuses, logs, source or traces.
Every reference step must match phase, attempt and grasp flags exactly; full
reference trajectories must agree within 0.1 mm in TCP/handle/base positions,
0.05 degrees in TCP orientation (1% of the controller's orientation gate), and
0.0001 rad in base yaw. The review also verifies source hashes, identical
reference parameters/episode configs, handle-frame reconstruction, and all 120
decisions for every branch. Its focused checks reject millimetre deviations,
changed contact flags and changed controller phases.

All ten cases passed this review. Final bilateral contact and sustained
verification counts are reference **7/10**, closing change only **8/10**, base
recovery only **9/10**, and both **10/10**. Those same counts maintained active-arm
contact throughout the final ten decisions. The separate authoritative reviewed
results and provenance are in `review/results.json`, `review/audit_policy.json`
and `review/index.html` under the recovery run. The original report still shows
the seven cases that passed its original nanometre-scale check. These ten Train
cases were used to develop the controller; 10/10 is a pilot contact result using
expert poses and simulator state, not held-out generalization, full-task
door-opening performance or a trained model result.

## Joint grasp and opening pilot

The user authorized testing whether the improved grasp connects to completed
door opening. Run
`/bigtemp/xna8aw/part-aware-policy-learning/runs/grasp_and_open_20261106_01`
uses GPU array `6834308` (two concurrent workers) and report job `6834309`.
Each of the same ten Train episodes has three methods: full recorded expert
action replay, improved control from the original closed-door state with no
expert action prefix, and a diagnostic improved controller after replaying the
expert approach. The expert reference is not inference from MolmoBot. All use
at most 500 decisions/50 seconds; the diagnostic prefix counts against this
budget. Grasping may use up to 200 decisions, then hands off immediately upon
verified contact rather than waiting for the previous 12-second audit to end.

Opening holds the gripper closed and uses a damped Jacobian servo toward a
20 mrad hinge-arc lookahead. The actual contact-relative TCP at handoff is
preserved, and forward opening motion pauses while contact needs reacquisition.
Robot arm/base command increments are bounded; base XY travel during opening
is capped at 1 m from handoff, with yaw and torso held. Only physical position
commands advance live simulation. Door angles are never assigned to create
opening success. Expert grasp poses, current handle/hinge state and contacts
remain privileged inputs. No learned pose model is trained or evaluated.

Primary joint success requires verified active-arm contact (13 consecutive
samples) and the episode's original opening threshold together for ten further
consecutive samples, including the first qualifying simultaneous sample.
Door-only, grasp-only, instantaneous joint, and sustained joint outcomes are
reported separately. The near-handle diagnostic's primary metrics use only
post-prefix behavior; whole-episode metrics are preserved separately. Whether
the door was already open at the diagnostic grasp start or at opening handoff,
opening gain afterward and new threshold crossings are recorded explicitly.
This avoids attributing an already-open door to the opening controller.

Twelve focused tests pass: eight servo contracts and four opening/metric checks
covering hinge geometry, grasp-transform preservation and joint-success timing.
The opening tests reject door opening without grasp and grasp without opening.
Detailed per-decision contacts/poses/stages and compressed qpos/control/time
states are saved for subsequent replay and visual inspection. This is a
development controller test on ten Train starts, not a final benchmark result.

All 30 method/episode simulations completed. Original-start expert replay
verified grasp in 8/10, reached the door threshold in 9/10 and achieved sustained
joint success in 7/10. Improved control from the original start verified grasp
in 3/10 and sustained joint success in 1/10. After the expert approach, improved
control verified grasp in all ten, opened all eight doors that were below the
threshold at handoff, and achieved sustained joint success in seven of those
eight. One pull case opened the door while losing handle contact. Two diagnostic
doors were already open before grasping; they must not be credited as newly
opened by this stage. The main limitation remains the original-start approach;
the previous 10/10 near-handle grasp result does not generalize to this start.

The opening loop's early-stop test initially checked ten raw simultaneous
contact/open samples, while the reporting metric required those ten samples
after grasp verification. In already-open cases 04 and 08 this stopped after
only one opening step, before the reporting condition could finish. The stop
condition now calls the same joint metric as reporting and excludes expert
prefix history. A regression test checks that an already-open handoff cannot
stop prematurely; five opening tests now pass. Original results are preserved.

`grasp_and_open_20261106_02` reruns only cases 04 and 08 to complete verification.
The other eight cases are copied with an explicit hash/provenance ledger after
checking that their original stopping conditions already met the unchanged
metric, exhausted the budget, failed before opening, or exhausted grasp retries.
This correction does not change the joint success criterion or tune the
controller. `study.json`, `reused_case_provenance.json` and `submissions.json`
record the correction and retained results. The original report remains at
`grasp_and_open_20261106_01/index.html`; the verification-complete report will be
at `grasp_and_open_20261106_02/index.html`.

## Retained-grasp objective

The user clarified that pushing a door open without a firm handle grasp should
not count for this project. The official door-angle criterion and even a late
simultaneous grasp/open check are insufficient for that objective. The two
verification reruns had already completed when this clarification arrived;
there were no remaining queued diagnostic jobs to cancel.

`retained_grasp_metrics.py` defines a separate, explicit sampled retention
criterion: 13 consecutive closed-gripper samples, nonempty aperture above 2 mm,
at least 1 N normal contact force on each finger against the selected handle,
established before 10% opening. Require at least 50 percentage points of further
opening, at least 95% retained force/contact samples with no gap longer than one
100 ms sample, handle-relative translation slip at most 20 mm and rotation slip
at most 10 degrees, and ten consecutive force/contact samples above the episode
opening threshold. Small incidental motion before grasp is allowed. A mostly
opened door followed by a late grasp cannot pass. These numerical thresholds
are simulation proxies chosen for the new objective, not a hardware robustness
certificate; 10 Hz logs cannot establish uninterrupted contact between samples.

Four tests pass, covering successful retained opening and rejection of
push/late-grasp, absent contact force, empty/open grippers, contact loss and
excessive slip. A post-hoc audit applies the same policy to all completed traces,
excluding the expert prefix for the near-handle diagnostic, without changing
original benchmark outcomes. It found retained-grasp opening in 2/10 expert
replays, 1/10 improved original-start runs and 6/10 near-handle diagnostic runs.
These are expert replay/controller results, not a MolmoBot model comparison.

The policy, reasons per case, input hashes and frozen audit/test sources are in
`grasp_and_open_20261106_02/retained_grasp_review/`. The lower counts reflect a
stricter new objective and must not be mixed with earlier door-only or
simultaneous-contact success rates. Future model comparisons should report this
grasp-retention objective alongside the original benchmark metric.

## Guarded physical control for retained-grasp opening

The user authorized applying both control fixes: establish grip before opening,
and pause/recover rather than continue after losing it. These are implemented in
`firm_grasp_controller.py`, with a matched before/after trial in
`/bigtemp/xna8aw/part-aware-policy-learning/runs/retained_grasp_control_20261106_01`.
GPU array `6836435` and dependent report `6836436` use the same ten original
closed-door Train starts, no expert approach prefix and 500 decisions per
controller. The reference is the previous controller, not a MolmoBot model.
Expert interaction poses and simulator handle/contact state remain privileged.

The guarded controller approaches a pre-grasp waypoint 6 cm behind the target
along tool Z, then inserts in bounded 5 mm target increments. Closing retains
the 20 mm position/5-degree orientation gate. Deliberate hinge-arc commands are
unlocked only after 13 consecutive closed-gripper, nonempty, bilateral-force
samples. Opening more than 10% before this verification is a failure, rather
than a late-grasp success. Initial normal approach/base motion can still cause
incidental door movement; the controller does not freeze physical door state.

Opening preserves the measured contact-relative TCP at verification. It uses
a 15 mrad hinge lookahead, reduced to 5 mrad for low-force contact. Force loss or
more than 5 mm/5 degrees of relative slip pauses opening and enters a holding
recovery phase. Resuming requires 13 force-verified samples and return within
3 mm/3 degrees; persistent loss triggers reopening, a shorter pre-grasp approach
and bounded retries. The original slip reference is retained. At each 20 ms
control tick, further arm/base opening targets are cancelled in favor of holding
the measured joint state if gripping force is absent. This is a command change,
not a state assignment; inertia/contact dynamics may still move the door.

Ten focused tests pass: six state/geometry/guard checks and four retained-grasp
metric checks. The trial freezes the same success policy for both controllers,
preserves the previous controller code, and saves policy-step plus 20 ms force
observations and qpos/control/time states. It reports retained-grasp opening as
the primary outcome and door-angle success as secondary. No trained policy,
learned pose predictor or final held-out evaluation runs in this trial.
