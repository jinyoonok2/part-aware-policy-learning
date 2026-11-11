# Matched closed-loop Dev comparison

The user authorized running the trained critic against the original policy and
simulator selector. Use all **24 Dev starts**, three methods each (**72 rollouts**).
No Eval starts or outcomes are used. Critic seed 44, checkpoint epoch 20, was
selected by the existing offline Dev house-balanced regret criterion. The other
two members and their ensemble are not used for this comparison.

## Controlled protocol

Each case uses a fresh GPU worker/process. Its three methods share one GPU,
native MolmoBot checkpoint, copied full integration-state reference, initial
RGB/qpos/point inputs, goal, controller, and sensor/candidate seed rules. The
native task success criterion is the endpoint. Stop on success, terminal,
truncation, or 400 actual actions. Predict 16 native actions and execute up to
eight per decision. Baseline generates one candidate; simulator and critic
generate four. When trajectories diverge, their later candidate banks naturally
differ because their observations differ. The first sampled candidate must be
bit-identical across all three methods; the initial full simulator/critic banks
must also match exactly.

Baseline executes its candidate directly. Simulator selection retains the fixed
score weights and evaluates four independent eight-action branches. Critic
selection uses live current RGB, qpos, initial conditioning image/point and the
candidate sequence; it predicts four scores with the frozen single critic.
The live scorer does not read simulation results, dataset labels, or cached
current-view training features. Initial conditioning tokens are cached from
the available initial image, as in feature preparation.

## Timing

Before each timed rollout, warm up four native candidate calls and the method's
own scorer. Restore native policy state and verify live physics is unchanged.
Exclude loading and warmup from decision timings, but record warmup separately.
CUDA synchronization surrounds candidate generation and scorer computation.
Report total decision time = candidate generation + scorer time, and each part.
Scorer time includes the critic's live visual encoding/normalization/model pass.
Simulator time includes all four branch physics evaluations and scoring.

Baseline and critic generate **zero branch labels during timed execution**.
No RGB/conditioning/branch-label dataset is compressed per decision. All methods
share the same video, action/decision logs and qpos recording. Episode wall time
includes environment stepping and logging; decision time excludes those costs.
Within a method, first average decision times per episode, then average the
matched episode means so long failures do not dominate the primary latency.
Retain GPU identity so heterogeneous-device results remain inspectable.

The earlier data-collection wall times include label instrumentation and are
not reused as these fresh benchmark timings. Sampling seed rule is unchanged:
42 + 1000×decision plus candidate index for flow sampling; per-action sensor
seed 42 + 100000 + actual step. No reward, policy or critic weights are tuned.

## Checks and outputs

CPU dispatch tests prove baseline and critic do not call the simulator branch
selector; report tests keep incomplete/unvalidated triples out of head-to-head
denominators. A GPU pilot first audits live critic preprocessing/features/scores
against a saved Dev example, before native model loading and timed decisions.
Only that implementation audit reads cached Dev features. Full paired starts,
candidate/goal hashes, finite actions/physics, actual video frame counts, method
branch counts, and validation completion are checked. Task failures remain valid
outcomes; infrastructure errors are recorded separately and are not silently
overwritten. Completed pilot data is reused, not rerun for a better result.

Run: `/bigtemp/xna8aw/part-aware-policy-learning/runs/closed_loop_dev_20261103_01`.

- Pilot **6803915**, case slot 0 with live-critic audit.
- Main array **6803921**, `0-23%2`, `afterok` pilot, max two GPUs concurrently.
- Final CPU report **6803922**, `afterany` main array.

Pending jobs were adjusted with scontrol to **64 GiB host memory**, four CPUs and
compatible H100/A100/A40/RTX 4000 Ada/A4500 GPUs; the previously problematic node
nekomata02 stays excluded. Native collection measured under 40 GiB host peak and
the extra critic/encoder are small. The pilot's limit is 30 minutes, main tasks
one hour. This resource override is recorded separately from frozen scripts.

`study.json`, `source-sha256.json` and model artifact hashes freeze the comparison.
`index.html`, `aggregate.json`, `cases.csv` show statuses, successful native tasks,
paired gains/losses and timing for validated triples. Individual method videos,
events, decision scores/timing and start hashes are retained. `comparison_ready`
requires all 24 validated triples. This is development evaluation; final held-out
Eval remains a separate subsequent experiment after settings are fixed.

The pilot started on cheetah02 after the resource adjustment. Its live-critic
GPU audit passed: maximum feature difference 0.015625, maximum score difference
0.001472 versus the saved-feature pipeline on the same Dev example. Cached data
is used only by that pre-timing implementation audit, never by rollout selection.
All jobs have now completed successfully. All 72 rollouts and all 24 matched
triples passed validation, with no execution errors or pairing errors.

## Completed Dev results (November 4)

| Method | Successes | Mean decision time | Mean scorer time |
| --- | --- | --- | --- |
| Baseline | 16/24 (66.7%) | 0.674 s | 0.000028 s |
| Simulator | 16/24 (66.7%) | 8.256 s | 5.606 s |
| Single critic | 15/24 (62.5%) | 2.735 s | 0.0367 s |

Simulator selection gained three successes and lost three relative to baseline.
The critic gained one and lost two relative to baseline. This Dev experiment
does not demonstrate improved success from either selector. The critic scorer
is faster than simulator branching, but its total decision time also includes
four native candidate generations. These are episode-averaged measurements
across heterogeneous GPUs, with the same device within each matched triple.
The checkpoint was selected using Dev; final held-out Eval remains untouched.

`scripts/make_comparison_videos.py RUN` renders all 24 triples under
`comparison_videos/`, with a gallery, an `episode_XXXX/` folder per episode,
`comparison.mp4`, links named `baseline.mp4`, `simulator.mp4` and `critic.mp4`,
preview images and a manifest with source/output hashes. The earlier combined
MP4 and flat-layout metadata are preserved separately in `video_layout_archive/`.
Panels synchronize by
action step at 10 steps/s, hold and label finished runs, and display native
opening metrics and measured mean decision time. Playback excludes computation
delays. Input event sequences, frame counts, success outcomes and validation
flags are checked, followed by output frame count, dimensions and FPS checks.
All episodes, including failures, are included.

The finish-frame overlay uses **zero-based source frames**: frame 0 is the
initial view, and frame N follows action N. First success is read from the
native `events.jsonl` success flag; it must equal the rollout's stop frame for
successful runs. Failures have no success frame and show their stop frame
separately. All 72 rollouts passed this mapping check. Comparison videos now
show persistent success/stop labels, a banner beginning at the exact finish
frame, and three timeline rows on one shared frame axis. A final 20-frame hold
at 10 fps keeps the last success visible for two seconds. This hold is
presentation only and does not add robot actions or alter rollout statistics.

Each episode page and the main gallery include first-success/stop-frame tables
and buttons to seek directly to the corresponding comparison-video frame.
`finish.jpg`, milestone `preview.jpg`, episode `summary.json` and the gallery
manifest record the result. Original individual method videos remain linked
to the native recordings. Previous comparison videos and metadata are retained
in `video_annotation_archive/before_finish_timeline_v1/`.
Use `make_comparison_videos.py RUN --refresh-annotations` to preserve an existing
gallery and render updated annotations; an existing archive is protected from
replacement.
