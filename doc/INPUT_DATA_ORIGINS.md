# Input origins: MolmoBot baseline and our pose predictor

This describes the current DoorPlusOpen implementation. Each model has **four
input groups**, as listed below. These are conceptual groups: MolmoBot's actual
inference API receives an image list, a state vector, and a task string that
includes the point prompt.

## 1. Original MolmoBot baseline inputs

| # | Input group | Where it comes from | How the baseline uses it |
| --- | --- | --- | --- |
| 1 | Current RGB images from three cameras | Simulator RGB rendering from `head_camera`, `wrist_camera_l`, and `wrist_camera_r`; recorded as camera videos in the source demonstrations. | Supplies the current visual observation whenever an action chunk is requested. |
| 2 | Initial head-camera RGB image | The first head-camera observation is copied and retained by the original policy wrapper. | Appended as a fourth image on subsequent requests; provides the visual frame to which the initial point prompt refers. |
| 3 | Current robot state: 22 values | Observed robot `qpos`: base 3, left arm 7, left gripper 1, right arm 7, right gripper 1, torso 3. The torso uses indices `[1, 2, 3]`. | Passed as the numerical state vector. Head joints are excluded from this baseline vector. |
| 4 | Task instruction with the initial handle point prompt | The task supplies “Push the door open” or “Pull the door open.” The native point sensor supplies initial head-camera handle coordinates, generated from the simulator's handle mask. | The original wrapper retains the initial annotations and formats at most **one** valid head-camera point into the task string using Molmo point tags. It does not refresh this prompt with current handle points. |

**Point annotation and point prompt are not two independent inputs.** The
annotation is the coordinate data; the prompt is its formatted representation
inside the instruction. Likewise, the initial image is a retained camera
observation, not an additional sensor.

Implementation: [`MolmoBotRBY1MultitaskPolicy`, `_format_point_prompt`, and
`MolmoBotRBY1DoorPlusOpenPolicyConfig`](../MolmoBot/MolmoBot/olmo/eval/configure_molmo_spaces.py).

## 2. Our pose predictor inputs

| # | Input group | Where it comes from | How our predictor uses it |
| --- | --- | --- | --- |
| 1 | Current RGB images from the same three cameras | Recorded RGB frames during training; current simulator RGB observations during online inference. | A separate frozen DINOv2-base encoder extracts global image features and local features sampled at the supplied handle points. |
| 2 | Current robot joint state: 21 values | Observed `qpos`: head 2, torso 3 at indices `[1, 2, 3]`, left arm 7, right arm 7, left gripper 1, right gripper 1. | Normalized using Train statistics and embedded by the pose head. Global base coordinates are excluded. This selection differs from the baseline's 22-value vector. |
| 3 | Push/pull task type | Parsed from the task instruction as a binary `is_pull` value. | Supplies a task embedding. The pose predictor does not consume the full language instruction or its initial point tags. |
| 4 | Current handle points and visibility in each of the three cameras | The native simulator point sensor, or its recorded per-frame annotations in the demonstrations. For each camera, our loader averages the valid sampled handle coordinates. | Supplies normalized `(x, y, valid)` for each view, guides local visual-feature sampling, and provides a point embedding. Missing views use `(0, 0, 0)` and have their local feature token masked. |

The pose predictor does **not** receive the retained initial image or initial
point prompt. DINOv2 is an internal feature extractor, not a fifth data input.
It does not generate the handle coordinates. The trainable pose head predicts
the grasp pose from these four groups.

Implementation: [`data.py`](../pose_learning/data.py),
[`vision.py`](../pose_learning/vision.py),
[`model.py`](../pose_learning/model.py), and
[`predict.py`](../pose_learning/predict.py).

## 3. How the handle masks and points are generated

The original MolmoSpaces framework already provides
[`ObjectImagePointsSensor`](../molmospaces/molmo_spaces/env/sensors.py):

1. The task identifies the target object as `door_handle`.
2. The simulator renders that object's segmentation mask separately for each camera.
3. The sensor samples up to 10 visible mask pixels per camera. Door handles skip mask erosion because they are thin objects.
4. Pixel row/column coordinates are converted to `(x, y)`, normalized to `[0, 1]`, and rounded to four decimal places. Unused slots are padded with `NaN`; `num_points` records the valid count. An invisible handle has zero valid points.

The same native annotation mechanism supports both paths:

```text
Simulator handle mask -> sampled 2D handle annotations
                         |
                         +-> Initial head-camera sample, at most one point
                         |   -> formatted baseline prompt, retained over time
                         |
                         +-> Current samples in all three cameras
                             -> per-camera mean + visibility for our pose predictor
```

Recorded annotations are read from
`obs/extra/object_image_points/door_handle/{camera}/{points,num_points}` in HDF5.
Online inference obtains fresh annotations from the simulator sensor. This does
not imply that full mask images are saved in every downloaded demonstration.

The current handle points are **not predicted** by MolmoBot, DINOv2, or our pose
head. They are simulator-derived, privileged **2D localization**. The baseline's
initial prompt also uses simulator-derived annotations; our addition is updated
localization in three views. Initial and current points need not have the same
coordinates, even initially, because the baseline selects a sample while our
predictor averages the available samples.

Neither model directly receives a full handle mask. Our current implementation
does not include mask pooling or a learned handle detector/tracker. Real-camera
use would need a perception component to supply the current coordinates and
visibility instead of simulator segmentation.

## 4. How the grasp-pose training targets are generated

The four pose-predictor inputs above are separate from the **supervision labels**.
Our target is an expert-derived future grasp TCP pose, where TCP means tool
center point; it is not the handle's own pose.

1. Start with the authors' recorded expert demonstrations, including robot
   states, measured left/right TCP poses, robot-base poses, policy phases,
   gripper positions, grasp flags, and door state.
2. Identify a completed `GRASP_HANDLE` event with measurable gripper closure and
   a true `handle_grasped` flag for the final three observations. The active arm
   is inferred from the larger gripper closure.
3. The early-grasp view chooses the earliest eligible event whose preceding
   trajectory has not exceeded 10% door opening. It selects four distinct
   conditioning observations before that event when enough frames exist.
4. Take the measured active-arm TCP at the event's target frame. Convert it to
   world coordinates using the base pose at that frame, then express that same
   world target relative to the base at each conditioning observation:

   ```text
   T_world_target = T_world_base(target) * T_base_tcp(target)
   T_base(condition)_target = inverse(T_world_base(condition)) * T_world_target
   ```

Each conditioning example therefore pairs **current** RGB, joints, task type,
and handle points with a **future expert grasp target**. Additional labels are
the active arm and the sum of absolute finger displacements, used as a
finger-span proxy. The source event criteria do not establish force-certified
continuous grasp retention for every demonstration.

The head predicts TCP translation, rotation, active arm, and finger span.
Rotation uses a continuous six-value representation internally; the inference
interface returns translation plus a `wxyz` quaternion in the current robot-base
frame. Finger span is a prediction target, not the native gripper actuator
command.

Implementation: [`collect_interaction_pose_dataset.py`](../data_preparation/collect_interaction_pose_dataset.py)
extracts measured events;
[`build_early_interaction_pose_view.py`](../data_preparation/build_early_interaction_pose_view.py)
selects early events and converts targets into each conditioning frame;
[`data.py`](../pose_learning/data.py) separates predictor inputs from labels.

## 5. Inference inputs, labels, and controller information

| Information | Role in the current system |
| --- | --- |
| Future expert TCP pose, expert active arm, and measured finger span | Supervision during training; not inputs to ordinary learned pose inference. |
| Expert phase, grasp flags, and door-opening history | Used to construct/filter training labels; not pose-head inputs. |
| Recorded expert actions | Source demonstration data; not replayed by the learned pose predictor or its online comparison. |
| Current simulator-derived 2D handle points | Inputs to ordinary learned pose inference; disclose their privileged origin. |
| Current robot-base world transform | Used outside the neural predictor to convert its base-relative output into an executable world target. |
| Contact/force feedback | Used by the local execution controller and grasp metrics; not an additional neural input to either model described above. |
| Recorded expert 3D TCP target in `oracle_pose` mode | An explicit diagnostic substitution for the learned pose output, with expert arm selection. This is distinct from the 2D annotations used by the ordinary predictor. |

The combined online system retains MolmoBot for approach/opening and uses the
pose predictor to guide bounded local arm corrections. Consequently, the
comparison includes a new predictor, a local controller, and updated 2D handle
localization; it should not be described as an architecture comparison with
identical input information.

Execution details: [`evaluate_policy.py`](../pose_learning/evaluate_policy.py)
and [`native_adapter.py`](../grasp/native_adapter.py). The separate expert-pose
diagnostic is defined in [`ORACLE_EVALUATION.md`](../pose_learning/ORACLE_EVALUATION.md).
