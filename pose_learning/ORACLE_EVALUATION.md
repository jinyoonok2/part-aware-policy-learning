# Expert-reference oracle comparison

Run: `runs/pose_oracle_test197_20261110_01`, submitted November 10, 2026.

Compare three methods from each identical start: original MolmoBot,
MolmoBot with the fixed step-20,000 pose predictor, and MolmoBot with the recorded
expert initial-grasp TCP pose. Every method is rerun within this comparison so
initial state/images/conditioning, first proposal and physics are audited
together. The existing 197-start comparison continues unchanged.

The original population contains 197 starts. Exactly 187 have a reference
under the same early-grasp rule used for training. Those 187 form the common
three-method population: 561 rollouts. The ten missing references are listed
explicitly in the new manifest and are excluded from all three methods here.
There is no baseline fallback or selection using model success.

## What the oracle means

The oracle supplies the measured expert TCP world pose and expert arm label.
It is the exact target used for pose supervision, expressed in the robot's
current base frame and updated on the same eight-action schedule. Its world
target is fixed; no new handle tracking, expert action replay or navigation
shortcut is added. The local controller, trigger, contact feedback and budgets
are identical to the predicted-pose method. Native finger closure still uses
pose error/contact checks; the recorded finger-span proxy is not executed.

References were selected from source phase, closure and grasp flags. They are
not force-certified optimal grasps and are not a guaranteed performance upper
bound. If the oracle does not improve execution, inspect both the target
definition and controller. A gap between oracle and predicted poses would
support focusing on prediction accuracy.

All methods use seed 42, a 400-action horizon, identical simulator assets and
the fixed checkpoint. Model/vision code and local-control dependencies match
the ongoing comparison. Evaluation/report code is extended in a separate
frozen snapshot. Twenty-five CPU checks passed, covering frame conversion,
missing-reference coverage, method isolation, matched reporting and native
action conversion. The first GPU episode gates the remaining cases.

## Scheduling and progress

| Job | Stage | Dependency |
| --- | --- | --- |
| 6862998 | First full three-method episode | Expert pilot array 6862241 ends |
| 6862999 | Remaining 186 episodes, one GPU at a time | First oracle episode succeeds technically and original evaluation array 6858963 ends |

The pending expert bulk array 6862243 also waits for the first oracle episode
to end, in addition to its original recording review. This places the first
oracle episode between expert pilot and expert bulk. Later, oracle evaluation
uses the GPU slot released by the original evaluation. The project remains
limited to two simultaneous GPUs. Recorded expert outcomes do not determine
whether the oracle experiment proceeds.

Read `submissions.json` and `expert_capacity_barrier.json` for exact commands
and scheduling changes; `plan.json` and `source_manifest.json` bind settings,
references and source hashes. `policy/report.json` reports progress and rates
on complete matched triplets only. Missing/pending/technical failures are not
counted as physical failures. Each method saves video, actions and force traces.

```bash
squeue -j 6862998,6862999
```
