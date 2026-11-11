# Final pose-policy evaluation protocol

## Superseding status: stopped before expert certification

The user requires a witnessed expert solution satisfying retained-grasp opening
before policy evaluation. The 200 tasks below are now **unverified candidates**.
Baseline array `6837253`, progress report `6837254` and artifact audit `6837260`
were cancelled. Forty-seven completed rollouts and all partial outputs remain
preserved; they do not certify task feasibility and must not guide task selection.
No automatic evaluation resume is enabled.

The [expert eligibility protocol](2026-11-07_expert_eligibility_validation.md)
supersedes the earlier assumption that source ordinary-opening success was
sufficient. Exact corresponding expert H5s were not found in the inspected
released Val catalog or local benchmark/test resources: 0/200 exact matches,
0 witnessed retained-grasp certifications. The source pointers name a private
February benchmark collection. This does not establish that no copy exists
elsewhere, or that the tasks are impossible.

See `source availability evidence` (historical path: `runs/final_pose_eval_v1_20261107_01/expert_validation/source_availability.json`)
and `per-task eligibility status` (historical path: `runs/final_pose_eval_v1_20261107_01/expert_validation/eligibility.json`).
The initial metadata, seeds and proposed metric below remain historical
candidate protocol. A certified expert-backed final manifest must precede any
new policy-scoring run. Workspace workers enforce this prerequisite and reject
reuse of old pre-certification rollouts as certified final results.

Prepared November 7, 2026. The primary outcome is opening a door while
maintaining a force-qualified handle grasp. Ordinary door opening is a
secondary outcome. The former baseline collection was stopped; a submitted job is
not a completed result and no learned pose model is trained by this protocol.

## Final task set and comparison

The frozen set contains 200 official benchmark starts from 100 houses, two
starts per house, with 100 push and 100 pull tasks. Each start is evaluated
with action seeds 42, 43 and 44: 600 planned baseline rollouts. Selection uses
fixed source metadata and a deterministic seed, without policy outcomes.
The selection manifest and contamination audit, rather than this target,
record the achieved composition and precise exclusion history. The source is
the pinned 2,000-start official Val benchmark, SHA256
`a8fb5bb75a746acf2ed352fe5d6185f818c4629da8f8f711f19b169ab9aa96b9`.
This balanced project subset is not the full official benchmark score.

Run: `final_pose_eval_v1_20261107_01`.
`Frozen episode manifest` (historical path: `runs/final_pose_eval_v1_20261107_01/final_episode_manifest.json`),
`selection/exclusion audit` (historical path: `runs/final_pose_eval_v1_20261107_01/audit/selection_audit.json`),
`live progress` (historical path: `runs/final_pose_eval_v1_20261107_01/index.html`).
The manifest SHA256 is
`8d9d8de704a9c57e492c0632a126714c811cd59e99469a738ae03ad94fd1cd7b`.
The house strata are 60 mixed-task, 20 push-only and 20 pull-only houses.
Ten houses necessarily use two distinct starts for the same target door;
190 scene/target-door pairs are represented across the 200 starts.

Exclude prior project training/development starts and houses conservatively,
including the candidate-selector study and pose expert/validation sources.
Audit scene-family aliases as well as numeric house identifiers; source
isolation does not by itself establish disjoint underlying geometry. Do not
use this final set for gradient training, checkpoint selection, control-rule
tuning or choosing examples after inspecting baseline failures.

The audit excludes 1,120 numeric house IDs drawn from source Train/Val
inventories and all known project selection manifests. The reserved final
100-house list is registered in `configs/final_evaluation_reservations.json`.
Both future expert-dataset preparers verify the final manifest identity,
exclude the reserved IDs across scene families and freeze their reservation
snapshot. Reused Train/Val indices and raw parent directories must also be
disjoint. Current frozen datasets remain unchanged. Cross-ID geometry
duplication and the released pretrained model's full upstream exposure are
not certified by this project-use audit.

Freeze the original source benchmark hash, exact episode JSON, selected IDs,
selection order, seeds, source/code/model identities and evaluation-policy
dictionary before model execution. An already-inspected development episode
outside the final list provides the integration smoke. The original MolmoBot
baseline is collected now. Future guard-only and learned-pose methods use the
same final episode/seed list and saved initial conditions; oracle expert poses
are not an input to the learned method.

## Primary outcome: firm grasp and retained opening, version 3 onset

The executable definition is `FINAL_POLICY` in
`retained_grasp_metrics.py` (historical path: `scripts/retained_grasp_metrics.py`), with
metric version `firm_grasp_opening_v3_onset`. Both arms are measured independently;
the episode succeeds if either arm satisfies the complete criterion. Contact
from two incomplete arm attempts cannot be combined into one success, and a
handover does not count as a single retained grasp under this definition.

1. Record the initial observation, then sample after every action at 10 Hz.
2. Anchor the grasp at its first force-qualified contact observation while
   opening is at most 10%, with no earlier observed opening above 10%. Then
   verify 13 consecutive observations having contact on both fingers with the
   selected handle, at least 1 N normal force per finger,
   gripper aperture at least 2 mm and a closed-gripper control value at least
   -0.005 in the pinned native controller's internal units. Thirteen samples
   span 1.2 seconds at 10 Hz. The door may open during these confirming
   observations; a stationary wait before opening is not required.
3. Opening first, reclosing and subsequently grasping cannot recover onset
   eligibility. If contact verification is interrupted, another candidate
   onset is allowed only while no prior observation has exceeded 10% opening.
   An attempt that loses its grasp after substantial opening cannot reset
   this history by reclosing.
4. From the first force-qualified grasp onset through success, retain qualifying contact
   on at least 95% of sampled observations, with at most one consecutive
   missing-contact sample. Limit TCP translation relative to the selected
   handle to 2 cm and orientation change to 10 degrees from the anchor.
5. Gain at least 0.50 of the normalized door-opening range after the onset.
   Reach the native task's exact `door_openness_threshold` while retaining
   contact for ten consecutive observations. Ten samples span 0.9 seconds.
   Save that native threshold for each case; do not choose it after outcomes.
   Success requires both the 13-sample force verification and ten-sample
   above-threshold contact hold to be complete. Those windows may overlap.
   Slip and opening gain are measured from onset, including all observations
   during the initial verification window.

The force/contact and slip checks are simulation measurements at 10 Hz. They
provide a sampled operational definition of firm retained grasp, not proof of
uninterrupted contact between observations or real-hardware robustness.

Historical pilot records retain their original `POLICY` and metric semantics.
The final onset definition includes the initial observation, preserves the
no-prior-major-opening condition and allows legitimate fast grasp-and-open
motion while contact is being verified. Do not directly compare its rates with the earlier
2/10 baseline, 3/10 guard and 4/10 oracle rates computed under the old protocol.

## Runtime and eligibility

Use frozen pretrained MolmoBot with native point prompts, initial conditioning
image, head-camera warp, cameras and robot state. Preserve native relative
arm/base commands and scalar torso-height control. Keep the source/model
versions, physics, controller modes, action frequency, initial images/state
and seed rules equal for every comparison method.

Allow at most 400 executed actions (40 simulated seconds at 10 Hz). Disable
native early stopping on ordinary door-angle success equally across methods.
Stop on primary success, explicit native terminal/truncation, or the action
budget. Local correction actions count against the same 400-action budget.
Compute the primary criterion from both-arm traces. The frozen baseline model
receives its original observation inputs, with no added evaluator force,
exact-handle-pose or future-state features. Any future auxiliary controller
feedback must be documented explicitly as part of that method's input scope.

Before policy inference, validate initial finite state and closed-door
eligibility. Starts already above 10% opening are marked statically invalid,
with their initial observations and reason retained. Save one common static
eligibility mask for all methods. Report all 600 planned rollouts, eligible
rollouts and static invalid counts explicitly. Do not replace cases based on
success or failure. Infrastructure and asset errors are recorded separately;
they are not silently dropped or converted into policy successes, and they
block a complete final comparison until resolved for the common task list.

## Recorded outcomes and artifacts

Primary: retained-grasp opening success over the common eligible rollout list.
Secondary: door-angle threshold ever reached, elapsed/executed actions,
time to verified grasp and primary success, contact retention, slip and
failure reason. Keep per-arm measurements and intervention counts, including
failures. Report each sampling seed separately and the pooled result; repeated
seeds and starts sharing a house are correlated observations.

For matched methods, report paired gains and losses on the same episode/seed
list. Estimate uncertainty by resampling houses as clusters, rather than
treating all 600 rollouts as independent tasks. Keep baseline results for
comparison after the learned model/checkpoint and development settings have
been frozen; development studies remain the source for tuning decisions.

Each rollout saves the exact episode, complete initial simulator integration
state, RGB/robot-state/point-prompt inputs with NaN padding preserved, physics
and conditioning hashes, first action, executed actions, both-arm force/contact
and handle-relative pose traces, robot state sequence, video and summary.
Future methods verify the initial-condition hashes before running. Metrics,
thresholds, denominator rules and the selected final set stay fixed.

## Submitted baseline execution

| Stage | Job | Dependency / scope |
| --- | --- | --- |
| Private asset staging | `6837251` | 200 final starts plus the separate development smoke; completed |
| Development smoke | `6837252` | Completed: all three actor seeds and fresh-scene reference restoration, exit 0 |
| Original MolmoBot baseline | `6837253` | After successful smoke; array `0-199%2`, three seeds per worker, maximum two concurrent GPUs |
| Progress report | `6837254` | After every baseline worker finishes, including failed tasks |
| Saved-state/static-mask audit | `6837260` | After the progress report; verify actual NPZ checksums and freeze common static eligibility without publishing outcome rates |

Each GPU worker requests one previously compatible GPU, 4 CPUs and 96 GB host
RAM. The asset stage and the full development smoke passed. The smoke completed
three full rollouts and three exact first-action/conditioning restoration checks
in 12m58s. It uses benchmark episode 0 in house 0, outside the final list. Infrastructure errors
do not trigger replacement task selection. Preserve frozen episodes and logs
on a retry; no new task set may be sampled after seeing outcomes.

The acquisition report exposes progress only. A separately frozen helper in
`publication/source/` and CPU audit `6837260`
verifies saved reference-array integrity and binds future learned-method
summaries to their frozen candidate checkpoint. To publish final rates, freeze
the candidate selected using validation with a receipt containing
`checkpoint_path`, `checkpoint_sha256`, `selection_basis: validation_only` and
this `final_episode_manifest_sha256`, then use the reviewed publication helper.
Raw results are retained for that later matched comparison. No training or
checkpoint selection uses these final baseline outcomes.

The original acquisition worker source remains unchanged. Its lightweight
progress report is separate from the stronger post-acquisition artifact audit.
The latter validates the actual NPZ files, metadata/summary/checkpoint bindings
and pre-policy eligibility receipts, then writes the deterministic common
`static_eligibility_manifest.json`. Learned results must identify the same
candidate checkpoint as the publication receipt. Final rates are published
only after the complete artifact and common-eligibility audit passes.

Focused checks passed: 52 across selection (7), input/seed persistence (5),
grasp metric (15), future-data reservations (9) and publication/integrity
accounting (16). These software checks are separate from actual simulator
rollout results.
