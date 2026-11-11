# Pose refinement: evidence and next experiments

The current checkpoint stays fixed for the original and oracle comparisons.
This document proposes later experiments; it does not start another training
run. The 197-start results are diagnostic once used to guide changes. Train
data should supply training/selection diagnostics, with the new 400-start
cohort reserved for evaluating a subsequently frozen method.

## Current evidence

The held-out pose report covers 187 source trajectories and 748 observations.
Its overall 8.79 cm mean position error hides a strong dependence on where the
robot is in the approach. Grouping the existing predictions by conditioning
sample gives:

| Conditioning sample | Mean position error | Within 2 cm of expert reference | Mean rotation error |
| --- | --- | --- | --- |
| First | 15.44 cm | 0.5% | 27.48 degrees |
| Second | 10.01 cm | 2.1% | 8.17 degrees |
| Third | 6.38 cm | 5.3% | 5.78 degrees |
| Last pre-grasp | 3.34 cm | 18.7% | 5.57 degrees |

Each row contains 187 observations. These are errors against heuristic
expert-derived pose references, not grasp-success percentages. The native
correction closes only within 2 cm and 5 degrees of its supplied target, so
prediction precision near contact is a plausible limitation, not an established
cause of the closed-loop failures. The oracle comparison helps distinguish
prediction error from target-definition/control problems.

Current vision features use complete RGB images resized to 224 by 224, a
16 by 16 DINOv2 patch grid, one global token and one sampled local token per
camera. The predictor receives joint values but no explicit measured current
TCP transforms. These choices are practical targets for improvement.

## Recommended first experiment

Keep MolmoBot responsible for the approach. Add a local refinement head using
explicit current TCP poses and high-resolution handle-centered crops. Predict
a translation/rotation correction relative to the current gripper, rather
than asking one head to infer the full target from raw joint angles and small
handle features throughout the approach. Apply it only near the handle and
keep the native action limits/contact feedback.

Train with additional pre-grasp observations from Train trajectories. Recompute
each residual in its actual observation frame. Later collect consistent RGB,
state and target observations under small robot-pose perturbations; changing
joint values alone would make images and state inconsistent. Inspect translation
threshold accuracy and orientation separately on Train diagnostics.

A useful ablation order is explicit TCP/residual inputs, then handle crops,
then denser near-grasp data. This separates geometric representation, image
detail and coverage. A later uncertainty/consistency gate could let MolmoBot
continue when a local target is unreliable; agreement alone does not certify
accuracy. These are hypotheses, not measured improvements.

## Related mechanisms

[RVT-2, Section III-B](https://arxiv.org/html/2406.08545v1) uses a coarse stage
followed by zoomed views for more precise gripper-pose prediction, and conditions
rotation on local features. It supports investigating finer local visual
information here; its reported improvements are on other systems and do not
establish results for our door-opening controller.

[PerAct](https://peract.github.io/) uses RGB-D geometry and a voxel action
representation. Explicit 3D geometry is a later option if matching depth or
calibrated geometry can be supplied consistently in Train and deployment.
The current RGB cache does not already provide that input to the pose head.
