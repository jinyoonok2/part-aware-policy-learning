# First native door-opening pilot

October 28, 2026. The independent environment completed a pretrained MolmoBot
rollout and handle-segmentation diagnostics. No part-aware modules were trained.

## Result

The single episode succeeded after **191 action calls**, stopping at door-open
fraction **0.6786** against the configured **0.67** threshold. The simulator
reported handle grasp at completion. This verifies bring-up on one start state;
it is not a success-rate or robustness estimate.

| Measurement | Result |
| --- | --- |
| Slurm job | `6746721`, COMPLETED, exit 0, 7 min 28 sec including setup/hash/model loading |
| Hardware | One NVIDIA A40, four CPUs, 200 GB requested host memory |
| Model load | 45.8 seconds, after hashing the checkpoint |
| Peak allocated / reserved GPU memory | 10.17 / 10.90 GiB |
| Saved actions / video frames / simulator states | 191 / 192 / 192 |
| Video rate | 10 fps |
| Mean `get_action` time | 131.4 ms, including buffered calls; not end-to-end control latency |

Run directory and local video/overlay review:

`/bigtemp/xna8aw/part-aware-policy-learning/runs/native_door_smoke_20261028_01/index.html`

The run contains source snapshots, scheduler IDs, `manifest.json`, checkpoint
hash, resolved config, `verification.json`, MP4, action JSON/JSONL, simulator qpos,
and raw RGB/depth/segmentation with mask overlays. The 19,992,232,602-byte
checkpoint SHA-256 matches its recorded Hugging Face download hash:
`009fa4dc60493da7b4aad9ffa58481fdd87298c2c11d95f6d472b67b56789a3d`.

## Protocol and interpretation

One fixed episode: index 0 of the local `molmospaces-bench-v2/20260327`
ProcTHOR RB-Y1 door-opening benchmark, house 0, “Pull the door open.”
Policy/action RNG seed 42; stop at success, termination/truncation or 400 actions.
Policy/controller/simulator periods are 100/20/4 ms. The policy predicts 16
20-dimensional actions and refreshes after executing eight. Its input state has
22 dimensions. The native checkpoint configuration retains point prompts,
the initial conditioning image, head-camera fisheye warping and default gripper
thresholding. Masks are diagnostics and never policy inputs.

The installed Objaverse version is `20260131`, while the benchmark code warns
that its original assets used `20251016_from_20250610`. Treat this as a local
bring-up result, not a reproduction of the authors' reported benchmark score.
Native point prompts already provide privileged target localization; future
controls must retain equivalent target information.

Job `6746707` was cancelled while pending to broaden the GPU constraint.
Attempt `6746709` loaded/rendered the scene but failed on configuration logging
before inference. Its outputs/source are preserved in `pilot_attempt1/` and
`source_attempt1/`. The serialization fix preceded the successful retry; no
policy setting was tuned using an episode failure.

## Handle masks

The task-selected handle body maps to seven visual/collision geometry IDs.
Rendered segmentation channels are object ID, object type and body ID. Masks
select the handle's geometry IDs with geometry-type filtering. Raw overlays were
visually inspected in head and right-wrist cameras.

| Step | Head pixels | Left-wrist pixels | Right-wrist pixels |
| --- | --- | --- | --- |
| 0 | 36 | 0 | 0 |
| 20 | 42 | 0 | 10 |
| 100 | 150 | 0 | 32,380 |
| 191 | 121 | 0 | 53,148 |

These are native-camera masks. The subsequent
[October 29 saved-data validation](2026-10-29_mask_preprocessing.md) checked
controlled fisheye/resize transformations and patch/pooling geometry. Historical
warp samples and training integration remain unverified. Small initial
handles and zero-visibility cameras require explicit handling before pooling
part features.

## Existing demonstration inspection

No dataset download was needed. Existing official validation data contains a
successful `house_125/trajectories_batch_1_of_1.h5:traj_0` episode, inspected
separately from the policy benchmark episode. It has 154 observations, 154 frames
in each of three camera videos and 154 action records, with empty initial action
padding. Recorded success occurs at index 153. Point annotations and recorded
phases exist; pixel-level handle masks do not appear in this trajectory.

Commanded actions are sparse: the inspected non-padding record contains base,
head, left-arm and left-gripper commands. Derived joint-relative actions and
robot observations have different move-group shapes. Loader-compatible action,
gripper, torso and timing conventions still need validation.

The demonstration's target-door ID was not found in the checked installed
train/val scene-125 metadata. Replay remains unverified; this is not a paired
expert/policy comparison, nor a training-ready data claim. Keep these official
validation demonstrations held out from training.

## Next experiment

Resolve replay/asset identity and action alignment for one demonstration, or
collect a new expert trajectory with the installed scene version. Shared
per-image RGB/mask preprocessing and empty-mask behavior now pass on saved frames;
validate their training integration. Only afterward select
a small training subset and implement the frozen-backbone adapter alongside a
matched global-feature control. A bulk dataset download is unnecessary for
these immediate checks.
