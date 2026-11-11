# Expert eligibility before final pose-policy evaluation

The user requires final tasks with an independently demonstrated, retained-grasp
opening solution. The existing 200-start manifest is an unverified candidate
set until that requirement is checked. Its episode identities, reservations and
collected baseline artifacts must remain intact; do not silently relabel it as
an expert-certified final set.

This document defines the prerequisite validation. It does not claim that source
expert files have been obtained or that any of the 200 tasks has passed it.
No model evaluation or training is launched by this document.

## What each evidence layer establishes

| Evidence | What can be concluded | What remains unproved |
| --- | --- | --- |
| Benchmark JSON and initial robot/object state | The exact task start can be reconstructed | An expert can satisfy the retained-grasp objective |
| Exact corresponding expert HDF5, actions and robot-state history | A source trajectory and source identity are available | The source success criterion includes firm retained grasp |
| Recorded opening success plus grasp-phase/closure flags | An offline pose reference can be extracted | Bilateral force, limited slip and retention throughout opening |
| Matched physics/control replay with actual contact traces passing the frozen final metric | A witnessed expert execution satisfies the project objective from that exact start | General policy reliability or every possible random execution succeeding |

Expert reference poses are measured poses from that witnessed trajectory.
Achieved poses from original MolmoBot rollouts are model outputs, not expert
ground truth. Pose-error evaluation and closed-loop task evaluation remain
separate uses of the same certified task/reference pair.

## Source matching comes first

For each candidate, obtain its exact original trajectory, not merely a file in
the same numeric house. Bind the public/retrieved source revision, archive or
HDF5 SHA256, trajectory key, observation count and source dates to the benchmark
`source` reference. Check the task/door/selected-handle identity, scene family
and asset versions, initial robot base/joints, door state and object poses.

The historical benchmark's source pointers name the February 10 benchmark
collection and batches such as `trajectories_batch_7_of_25.h5`. A different
released validation corpus or a successful demonstration from the same house
is not automatically its matching source. Missing or incompatible sources must
be reported explicitly. They must not be replaced with a model-generated label.

Audit observation/action alignment and units before replay: determine whether
the logged action is an absolute joint target, a relative action or an applied
actuator command, and which observation follows it. Preserve original robot,
controller, policy/control/physics time steps and action-noise semantics. If
collection noise was applied after the logged command, the logged command alone
may not reproduce the executed action; require the applied controls or a
reproducible noise/RNG record, or retain an unresolved compatibility status.

## Constructive expert execution witness

1. Reconstruct the exact closed-door start independently of baseline outcomes.
   Verify finite states and the same static eligibility rule used for policy
   evaluation. Do not use the learned/original model's success to select tasks.
2. Match the source physics, assets, controller modes and executable action
   interpretation. Record initial alignment and subsequent tracking errors.
   Small target-pose error alone does not certify a grasp.
3. Execute the expert reference through the simulator. Do not teleport the live
   robot or door along a state trajectory and call that an executable replay.
   State restoration can support alignment diagnostics, with that role explicit.
4. Record both arms from initial observation zero onward: exact selected-handle
   contacts, per-finger normal force, gripper command/aperture, TCP and handle
   poses, opening fraction, executed actions, states and video.
5. Apply the same frozen `FINAL_POLICY` (`firm_grasp_opening_v3_onset`) and native
   door threshold intended for the models. Qualifying force contact must begin
   before major opening and persist through opening with the specified force,
   verification, retention, missing-sample, slip and final-hold rules. The door
   may move during the 13-sample contact verification; no stationary pause is
   required. A door-angle success or a later regrasp is not a substitute.
6. Save the witness actions and full initial-condition/trace identities. A
   measured successful pose can then be paired with prior observation/state
   frames, with coordinate transforms checked. This validates a reference under
   the simulated metric, not uninterrupted real-world contact.

One reproducible matched expert witness establishes feasibility of that fixed
task/start under the specified simulator. Repeating expert executions measures
robustness and is a separate budget/protocol choice; it does not follow from
choosing three action-sampling seeds for a stochastic learned policy.

If exact action replay is impossible but an independent privileged expert can
solve the same exact start, that can provide a new feasibility witness. Preserve
its provenance as a newly generated expert solution rather than the original
source trajectory. Do not use the baseline or learned model to manufacture
the reference labels.

## Fail-closed status and promotion gate

Keep source identity, offline evidence, execution compatibility and witnessed
feasibility as separate fields. Suggested per-task records:

```json
{
  "case_id": "episode_XXXX",
  "episode_sha256": "...",
  "source_status": "exact_match | missing | unmatched",
  "reference_origin": "original_source | new_privileged_expert_witness | unavailable",
  "source_identity_verified": false,
  "reference_identity_verified": false,
  "source_artifacts_sha256": {},
  "offline_evidence": "pose_reference_only | sufficient_recorded_measurements | incomplete",
  "compatibility": "matched | mismatch | unknown",
  "expert_execution": "not_run | completed | infrastructure_error",
  "metric_version": "firm_grasp_opening_v3_onset",
  "expert_metric_passed": null,
  "feasibility_status": "verified | not_demonstrated | unknown",
  "reference_pose_status": "verified | source_only | missing",
  "baseline_outcomes_used_for_eligibility": false,
  "final_eligible": false
}
```

The workspace runner now calls `verify_expert_eligibility` before native
simulator/model imports or inference, including its smoke entrypoint. The
required certificate is `expert_validation/eligibility.json`, containing
`state: complete`, the frozen `candidate_manifest_sha256`, the frozen
`metric_policy_sha256`, and a `cases` list covering exactly the frozen case IDs
and episode SHA256 values. Every case must have all verified decisions shown
above, `compatibility: matched`, both identity flags true, and a real
`witness_artifact_path` plus matching `witness_artifact_sha256`. Original-source
references require `source_status: exact_match`; an explicitly identified new
privileged expert witness can retain historical-source absence while supplying
its own verified identity. Missing, incomplete or changed evidence blocks the
worker. A successful gate receipt is bound into subsequent rollout summaries.
Existing frozen acquisition workers remain unchanged and must not be resumed
as if they already contained this new prerequisite.

Promote a task only when its source/reference identity is established, execution
compatibility is matched, and a recorded expert witness passes the frozen final
metric from the exact evaluation start. The reference pose must be tied to that
successful grasp rather than a late or unrelated event. If a newly generated
privileged expert witness is used, certify and hash its own trajectory against
the original task start, and keep historical-source absence explicit. Missing
required reference data, unknown
compatibility, infrastructure errors and incomplete force evidence leave
`final_eligible: false` and a precise unresolved reason.

A matched expert execution that fails the metric is **not demonstrated** by
that reference. It is not proof that the task is impossible. A replay with
tracking, asset, controller or noise mismatch is **unknown**, not an intrinsic
task failure. Neither category may be treated as successful feasibility merely
because the source benchmark says `success`.

Earlier project evidence illustrates this distinction: ten source-success
Train cases produced nine door-opening replays, while their historical strict
retention re-score passed only two. Source collection enabled action noise and
the replay disabled it; those old counts neither certify nor disprove the
current 200 candidates under the new onset metric. See the
[earlier replay and controller record](2026-11-06_interaction_pose_dataset.md).

## Set-level decision and existing artifacts

Do not resume final policy scoring until the promoted final manifest and its
expert/reference audit are complete. Report candidate count, exact matches,
verified witnesses and every missing/unmatched/unknown reason explicitly. Do
not invent 200 verified tasks if only a smaller number can be supported.

If the original 200 can all be certified, retain their manifest and add the
independent witness audit. If source availability or witness coverage requires a
different cohort, create a separately versioned, expert-certified manifest.
Select from that certified pool with a preregistered balancing/order rule,
without consulting baseline successes or failures. Record any changed task
population and its conditional expert-certified scope. Do not overwrite the
original 200 or pool its partial baseline results with a new task set.

When exact benchmark source trajectories are unavailable, an alternative is a
new held-out expert-based cohort whose episode starts are constructed from
those demonstrations and whose matching controls/physics are retained. Prefer
unused official validation sources when available; a newly reserved official
Train subset is held out from this project's fitting, but may have been seen
by the released pretrained policy. Preserve that limitation. Existing Val
references used for checkpoint selection cannot also become final test data.

All previously reserved candidate houses stay excluded from future training
until an explicit reservation decision is recorded. Previously collected
baseline artifacts remain useful execution diagnostics and provisional
records; they are not an expert-certification filter or a certified final
performance claim. Evaluate original and learned methods from the same final
certified starts after promotion, with the same metric and input protocol.
