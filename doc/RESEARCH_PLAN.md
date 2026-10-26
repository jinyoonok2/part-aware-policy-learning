# Part-Aware Adaptation of MolmoBot for Robust Articulated Manipulation

**Author:** Jinyoon Kim

**Status:** Working research plan; experiments have not started

**Updated:** October 26, 2026

## Research question

Can trainable part-aware modules improve a pretrained MolmoBot policy's
robustness in articulated manipulation, and when is each adaptation design useful?

We will study articulated manipulation, initially one task family such as
opening a cabinet, door, or drawer. A handle is a small region that directly
supports the interaction, while the surrounding image contains background,
appearance, and scene information that may change across environments.

The project will investigate three additions to a compatible pretrained
MolmoBot policy: a part-aware adapter, a residual action module, and
phase-specific experts. All three address the same question using the same
MolmoSpaces task and evaluation conditions. The goal is to identify when part
information helps, has little effect, or becomes harmful when localization is
inaccurate. Improved performance is a hypothesis, not an assumed outcome.

## Motivation and relationship to the original proposal

The original [proposal](IR_course_Project_Proposal.pdf) asks whether part-aware
features improve manipulation robustness and generalization using ManiSkill.
This plan retains that core question while studying adaptation of an existing
MolmoBot policy under controlled visual shifts and imperfect part localization.
The earlier small global-only versus part-aware policy experiment remains an
optional diagnostic, rather than the main model being developed.

The planned environment is now MolmoSpaces with MuJoCo. The author has prior
experience with MolmoSpaces/MolmoBot on the UVA CS Linux server. Reusing that
experience may reduce integration work and support experiments across varied
scenes and articulated objects. The previous ManiSkill setup encountered an
asset-hosting outage; this did not establish a defect in the dataset itself.

## Roles of MolmoSpaces and MolmoBot

| Component | Planned role |
| --- | --- |
| MolmoSpaces with MuJoCo | Environment, assets, task configuration, rendering, and demonstration generation |
| MolmoBot-Data | Optional source of compatible expert trajectories |
| Compatible pretrained MolmoBot policy | Shared starting checkpoint and unmodified reference for all adaptations |
| Our implementation | Part-feature extraction, adapters, residual corrections, phase experts, matched training, and evaluation |

Using these resources does not require reproducing full MolmoBot pretraining.
Initially freeze the pretrained backbone and train the added modules, with
limited action-head updates considered only if necessary and applied consistently
to the corresponding controls. Inspect the selected architecture before deciding
where to inject part information; these are proposed custom additions, not
verified plug-in interfaces in upstream MolmoBot.

No assumption is made that the released data already includes pixel-level
handle masks. Handle visibility metadata and part segmentation are different
forms of information; mask availability must be verified.

## Initial scope

- One robot embodiment, one controller, and one articulated task family.
- One intended target per episode, with equivalent task/target information
  supplied to all compared methods.
- RGB observations and robot proprioception as common inputs.
- A compatible pretrained MolmoBot checkpoint shared across the three designs.
- Part features from accessible MolmoBot visual features or a separate frozen
  encoder such as DINOv2; choose after inspecting the checkpoint and use the
  same part extractor across designs where feasible.
- Simulator-derived target-part masks for the first experiment, followed by
  controlled mask corruption.
- Linux execution on the UVA CS server; Slurm launchers will be developed there.

Language grounding, multi-task generalist training, real-robot deployment, and
full-backbone fine-tuning are outside the initial scope. The three adaptation
designs are planned as staged experiments, not one combined architecture.

## Proposed method

For each observation, retain the pretrained policy's normal inputs and construct
a target-part representation from spatial image features and a handle mask.
Use demonstration supervision compatible with the selected MolmoBot architecture;
do not assume every checkpoint uses a simple one-step action-regression loss.

### Design A: Part-aware adapter

Train a small module to fuse the part representation into an accessible policy
representation used for action generation. Begin with the backbone frozen.
Verify that the chosen connection actually influences the action predictor;
merely appending features to a disconnected output would not adapt the policy.
Use a neutral or near-identity initialization where the architecture permits.

### Design B: Residual action module

Keep the original policy as an action proposal and train a module to predict
bounded corrections using part features, robot state, and the base action.
Use the same action timing, normalization, chunk length, and controller as the
base policy. Define corrections in a valid action representation: rotations,
quaternions, discrete gripper commands, and termination decisions must not all
be treated as unconstrained additive vectors. Initialize corrections near zero
and measure both task success and control stability during rollouts.

### Design C: Phase-specific experts

Train small modules specialized for approach, grasping, and opening, together
with a router that chooses or blends their contributions. Keep the pretrained
policy shared. Define phases for the chosen task and obtain training labels
from validated demonstration annotations or simulator events.

At deployment, route from available observations and permitted policy features;
do not depend on simulator-only phase labels. Oracle phase routing may be
reported separately as a diagnostic upper bound. Compare against a single
module with a similar parameter budget, and examine errors or oscillation near
phase boundaries. Include a phase-agnostic part-aware design to test whether
phase specialization adds value beyond part information alone.

### Common comparison rules

Use the same starting checkpoint, demonstrations, controller, preprocessing,
perturbations, and evaluation episodes. Compare trainable parameter counts and
training compute, and provide capacity-matched controls where feasible.
These controls distinguish the benefit of part information from extra model
capacity, additional training, or routing. Begin with separate designs; combining
the adapter, residual module, and experts is a later experiment only if individual
results justify it.

Small handles can disappear when masks are resized to a patch grid. Verify the
alignment between image transforms, mask transforms, and patch coordinates.
Define a consistent representation of part location if needed; pooled appearance
alone may discard spatial information. Explicitly specify the fallback for an
invisible target, such as a zero part feature and a visibility indicator.

Simulator masks constitute privileged localization at inference time. Initial
results therefore test the utility of part information under known localization,
not an end-to-end system that discovers handles from RGB alone. Exclude unrelated
privileged object poses, success labels, and handle coordinates from policy inputs.

## Hypotheses

These are hypotheses to test, not expected findings presented as conclusions.

1. Part-aware features reduce performance degradation under nuisance changes,
   particularly background and appearance shifts.
2. The benefit varies by perturbation; viewpoint changes and occlusion may
   reduce the quality of the visible part representation.
3. Increasing mask error eventually removes or reverses the advantage over the
   global baseline.
4. Improvements on unseen geometry may differ from improvements on unseen
   appearance and must be measured separately.
5. Phase-specific modules may help most at particular interaction stages, but
   routing errors may offset the benefit of specialization.
6. Adapter and residual designs may have different robustness, latency, and
   control-stability tradeoffs; no ordering is assumed in advance.

## Data preparation

First inspect the existing server setup and select a supported articulated task.
Then choose either a compatible MolmoBot demonstration subset or a small dataset
generated with the MolmoSpaces pipeline. Validate one successful trajectory
before scaling collection.

Retain camera observations, proprioception, action/controller definitions,
success labels, target identity, scene/object identifiers, and available replay
metadata. Verify observation-action timing against the chosen dataset revision;
do not assume observations and actions align at identical array indices.

If pixel-level masks are absent, determine whether they can be rendered during
replay or recorded during new data generation. Verify that a mask isolates the
handle rather than an entire door or drawer link. A whole-link representation
can be studied, but it must be labeled accurately.

Keep complete episodes together in every split. For geometry generalization,
hold out object/asset identities, including repeated copies of the same asset
across scenes. A new background or scene ID alone does not establish unseen
object geometry. Reserve validation conditions separately from final test
conditions and keep all corruption variants of an episode in its assigned split.

## Experimental design

The architecture designs and evaluation directions are separate axes:

| Evaluation direction | Applied to |
| --- | --- |
| Visual robustness | Unmodified policy, matched controls, and all three part-aware designs |
| Imperfect localization | All designs consuming a mask-derived part representation |
| Phase-dependent usefulness | Phase-stratified analysis for all designs; explicit routing experiments for Design C |

The main comparison includes:

1. The unmodified pretrained MolmoBot policy.
2. A matched adaptation control without task-relevant part information for
   each design, using the same demonstrations and comparable trainable capacity.
3. MolmoBot with the part-aware adapter.
4. MolmoBot with the part-aware residual action module.
5. MolmoBot with phase-specific part-aware experts and observation-based routing.

For Design C, include a routing control with global features but no part input.
An irrelevant-region feature control can further test whether target relevance
matters. Treat all simulator-provided target information as an explicit experimental
condition, not hidden supervision available to only one baseline.

Begin with clean-condition training and evaluation. Apply controlled test-time
shifts using matched episode seeds and perturbation settings. Initially change
one factor at a time. Keep the adaptation data and budgets matched so that
additional training alone does not explain gains over the pretrained policy.

| Evaluation condition | Purpose |
| --- | --- |
| Clean, seen conditions | Establish whether both methods learn the task |
| Background, texture, and lighting changes | Measure sensitivity to nuisance appearance |
| Camera viewpoint changes | Test dependence on the original observation geometry |
| Part occlusion | Test behavior with reduced visibility |
| Object pose and initial-state changes | Examine robustness to configuration shifts |
| Held-out object/handle geometry | Test transfer beyond the training asset identities |
| Corrupted masks | Measure dependence on accurate part localization |

For mask corruption, consider translation, erosion/dilation, partial dropout,
and a missing-mask condition. Define severity relative to image size or target
extent, and distinguish artificial mask error from actual visual occlusion.

Start with visual changes that preserve the underlying task and its success
criterion. Geometry and physical-state changes may alter reachability or
difficulty; validate those conditions and report them separately.

Useful ablations, subject to time and feasibility, include whole-object versus
handle features, irrelevant-region features, and a capacity-matched global
baseline. These help test whether gains come from task-relevant localization
rather than extra features or target-selection information.

## Metrics and interpretation

The primary metric is rollout task success. Report clean success and success
under each shift, including the percentage-point drop from clean conditions.
Also measure success as a function of mask-corruption severity.
Report trainable parameter counts, training resources, inference latency, and
failure/control-stability diagnostics for each design. For phase experts, report
phase prediction quality when labels are available, transitions, and end-to-end
success; accurate phase classification alone does not establish better manipulation.

Use multiple training seeds and matched evaluation episodes. Report sample
counts, variability, and uncertainty; account for episodes sharing an object or
scene when summarizing generalization. Offline action prediction loss is a
diagnostic, not a substitute for closed-loop success.

Inspect failures around approach, grasping, and opening where feasible. A result
showing that part features help only under certain shifts is still informative.
Avoid claiming that improved performance demonstrates a particular internal
attention mechanism without supporting evidence.

## Milestones

1. Inspect and reproduce the existing UVA CS environment; record source revisions.
2. Run one articulated scene and inspect RGB, target identity, and handle masks.
3. Validate one successful demonstration and observation-action alignment.
4. Validate a task/robot-compatible pretrained MolmoBot checkpoint and measure
   its clean rollout performance and GPU memory use.
5. Prepare the shared pilot dataset and matched adaptation controls.
6. Implement and evaluate Design A, the part-aware adapter.
7. Implement and evaluate Design B, the residual action module.
8. Define and validate phase labels; implement Design C and its routing controls.
9. Compare all three under visual shifts, mask corruption, and phase analysis.
10. Expand to held-out geometries if asset diversity and resources permit.
11. Analyze failures, document limitations, and write the report.

All three designs are part of the planned study, implemented in that order to
reuse the same data and evaluation harness. Establish a working adapter and
controls before expanding scope. If compute, checkpoint compatibility, or phase
labeling blocks a design, document the limitation rather than treating that
design as completed. Dataset size, number of seeds, evaluation counts, and
Slurm resource requests will be selected after the pilot.

## Open questions before implementation

- Which existing task/robot combination is already working on the server?
- Can we render handle-only masks for that task and the selected assets?
- Can existing demonstrations be replayed with the necessary observations?
- Which object identities and geometries are available for a meaningful split?
- Which MolmoBot architecture/checkpoint supports the selected task and robot?
- Where can part information be fused while retaining compatible action generation?
- Does freezing the backbone leave enough trainable pathways for useful adaptation?
- What action representation permits valid, bounded residual corrections?
- Can approach, grasping, and opening be labeled reliably and inferred from
  observations during deployment?
- What checkpoint memory and training costs fit the available GPUs, even with
  most parameters frozen?
- What compute, rendering, and storage configuration is available under Slurm?

No new framework installation or experiments have been performed for this plan.
See [setup and server handoff notes](SETUP.md) for the current repository state.

## References

- [MolmoSpaces repository](https://github.com/allenai/molmospaces)
- [MolmoSpaces data-generation documentation](https://github.com/allenai/molmospaces/blob/main/scripts/datagen/README.md)
- [MolmoSpaces trajectory format](https://github.com/allenai/molmospaces/blob/main/docs/data_format.md)
- [MolmoBot repository and data workflow](https://github.com/allenai/MolmoBot)
- [MolmoBot project](https://allenai.github.io/MolmoBot/)
- [DINOv2](https://github.com/facebookresearch/dinov2)

These links provide upstream context. Compatibility and implementation details
must be checked against the revisions used on the server.
