# Native baseline visual robustness study

November 2, 2026. Fixed development pilot to identify a measurable failure before
implementing part-aware adaptation. No adaptation training is submitted.

## Fixed design

Twelve benchmark starts: first two entries per house in houses **0, 1, 2, 3, 5,
9**, selected before evaluating their policy outcomes. Original benchmark indices:
`0, 1, 3, 4, 237, 238, 393, 394, 836, 837, 1747, 1748`.

Every start has three conditions, **36 rollouts total**:

| Condition | Change |
| --- | --- |
| Clean | Original scene appearance |
| Lighting | Diffuse ×0.55, ambient ×0.75, specular ×0.65; RGB cast [0.85, 0.95, 1.10], including renderer headlight |
| Background recolor | Override materials only on non-target structural geometry; blend 70% seeded color / 30% original geom RGB, preserve alpha; shared materials/textures and target/robot geometries remain untouched |

The initial texture-only calibration produced mean absolute RGB change
`2.22e-9`, below the predefined `0.003` visibility check, and was rejected before
policy inference. It was replaced with explicit background recoloring before
the first background policy rollout. Clean and lighting outcomes were retained;
the background calibration was based on render visibility, not policy success.

The native multitask checkpoint, action representation, point prompts,
conditioning-image policy, gripper threshold and controller are retained.
Evaluation/action-generator seed 42; per-action-call NumPy seeds
`42 + 100000 + step` keep the paired fisheye stream matched even after motions
diverge. Policy/controller/simulator periods: 100/20/4 ms. Predict 16 actions,
execute eight before refreshing, stop at success/termination/truncation or 400
action calls. This controlled RNG coupling differs from the original smoke run;
do not require its 191-action trajectory to reproduce exactly.

## Pairing checks

Each visual condition uses the same source episode and a freshly constructed
task/model. Restore the clean start's full `mjSTATE_INTEGRATION`, then verify:

- Exact initial integration-state equality.
- A fingerprint of geometry, physics/controller arrays and camera parameters.
- Exact initial head-camera segmentation and matching metric depth.
- Identical initial point annotations and formatted native task/point prompt.
- Nonzero, recorded head-camera RGB change for shifted conditions.
- Unchanged target/robot material and texture data.

Lighting may change target illumination; the recolor condition preserves the
target/robot style and all shared material/texture values. Background geometry
gets its own material override to avoid changing the intended target. Save initial RGB, intervention parameters and all
pairing checks beside each video. Full simulation trajectories may diverge in
response to changed observations; only their starting states are constrained.

The visual-change code passed checks on an actual small MuJoCo model, including
exclusion of a background texture shared with the target. The live GPU pilot
must complete all three conditions with pairing/render/video checks before the
full array starts. A task failure is a valid outcome; an execution error blocks
the pilot gate.

## Execution and outputs

Live review page:

`/bigtemp/xna8aw/part-aware-policy-learning/runs/baseline_robustness_20261102_01/index.html`

The run contains `study.json`, `manifest.json`, `submissions.json`, immutable
worker snapshots and hashes, a newly verified checkpoint SHA-256, and per-case
outputs under `pairs/`. The model is loaded once per house worker and reset
between conditions. Completed pilot cases are reused by the full array.

Scheduler records in `submissions.json` are authoritative. Initial submissions
were replaced after a renderer startup check:

- Pilot **6793028** failed during renderer initialization, before policy
  inference. Gated array **6793041** was automatically cancelled and report
  **6793042** preserved the error. This was not counted as a policy failure.
- Probe **6793070** established the actual CUDA/EGL device mapping.
- Corrected pilot **6793076** completed clean and lighting policy rollouts.
  Its texture-only visibility check failed before background policy inference.
- Background continuation pilot **6793125** reuses valid clean/lighting outcomes
  and evaluates the visible recolor condition on the same GPU model, with the
  same initial-state, geometry, depth/segmentation and prompt checks.
  It completed successfully; all three pilot conditions passed execution and
  pairing checks.
- Full array **6793133**, six house workers with at most two GPUs concurrently,
  is gated by `afterok:6793125`; final report **6793134** uses `afterany:6793133`.
  The gate passed and workers 0 and 1 started on November 2.

The pilot can use the available RTX 5080: the installed torch 2.7.1+cu128 wheel
includes `sm_120`. The resource request allows H100, A100, A40, RTX 4000 Ada,
A4500 and RTX 5080 GPUs, four CPUs and 64 GB host memory. The previous pilot
measured 10.17 GiB GPU allocation and under 30 GiB host peak. Each matched triple
normally uses the same worker/GPU; the resumed pilot requires the same GPU model
and exact pairing checks. GPU identity is saved per case. A preceding pending
request was replaced to improve scheduling. Startup job 6793020 was stopped to
inspect EGL selection before scene rendering; its source/startup records are
preserved. The initial physical-index assumption was incorrect under Slurm GPU
isolation. The corrected worker queries `EGL_CUDA_DEVICE_NV` and chooses the
unique EGL device matching the sole CUDA-visible device. Slurm's global GPU
index must not be assumed equal to the allocation-local EGL index.

## Measurements and interpretation

Report native task success, grasp-frame counts, first registered grasp,
maximum door opening, video, action records, simulator qpos and memory/timing.
A 5 cm TCP-to-handle distance is only an approach proxy, not a validated phase
label. Separate task failures, execution errors and pending/missing cases.
Compare clean→failure and failure→success flips only among completed pairs;
do not remove unsuccessful clean cases from the overall totals.

An auxiliary diagnostic in the original pilot mixed robot-relative TCP sensor
values with a world-frame handle location. Those distances were excluded with
original JSON/JSONL backups and `tcp_metric_audit.json`; action, success,
contact and opening records are unchanged. The bulk worker now reads world-frame
TCP sites directly. The fix passed a frame-offset regression check. Do not use
the excluded pilot distances to label approach failures.

Initial development evidence on start 0: clean succeeded after 163 actions
(maximum opening 0.6789, 76 grasp frames), while lighting failed at 400 actions
(maximum opening 0.0665, 9 grasp frames). Their integration-state, physics,
segmentation/depth and point-prompt checks match. Background recoloring succeeded
after 146 actions (maximum opening 0.6803, 61 grasp frames), with the same pairing
checks passed. Lighting changes the initial
head RGB by mean absolute fraction 0.1648; background recoloring changes it by
0.1131. These are one-start preliminary observations, not aggregate robustness
claims.

The completed development study contains all 36 rollouts and no execution errors:
clean 6/12 successes, lighting 7/12, background recoloring 8/12. Lighting lost one
clean success and gained two; recoloring lost none and gained two. The aggregate
therefore does not support interpreting the first lighting failure as a general
lighting weakness. These small, clustered development results are descriptive.

This is a small development study: one policy seed, six clustered houses, one
severity for each visual shift. House 0 includes the previously inspected smoke
episode. Installed Objaverse assets differ from the benchmark's original
revision. Results are local paired diagnostics, not an independent generalization
test or reproduction of a published success rate. Reserve separate evaluation
episodes before selecting or training an adaptation method.

No training dataset was downloaded. Missing simulation-scene packages were
fetched only as needed for the six selected houses; existing assets were reused
through project-owned copies. LRL code, environments, checkpoints and jobs were
left unchanged.
