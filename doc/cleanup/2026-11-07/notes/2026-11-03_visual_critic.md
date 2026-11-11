# Visual action critic training

The user requested a capable critic rather than a minimal numeric regressor.
Use a frozen pretrained DINOv2 ViT-B/14 and a trainable multimodal action
transformer ensemble. This is supervised imitation of the fixed short-horizon
simulator score; it is not a bootstrapped long-horizon RL value function.

## Model

- Official `facebook/dinov2-base`, pinned revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, downloaded to project-owned storage.
  Pretrained encoder weights and source hashes are recorded in the run manifest.
- Encode current head/left-wrist/right-wrist RGB and the original conditioning
  image. Resize full frames to 336×336 without cropping; ImageNet normalization.
  Each view retains CLS plus an 8×8 grid pooled from the 24×24 patch grid.
- Project the four views to width 256, adding view and spatial embeddings. Add
  robot-state and goal tokens, then a context transformer layer.
- The 22D robot state exactly follows native RBY1 extraction: base 3, arms 7
  each, gripper first scalar 1 each, torso indices [1,2,3]. Raw grippers contain
  two joints; taking both was caught and fixed in real-data preflight.
- Goal features encode push/pull, the native initial target point and presence
  flag, known action horizon, and current action count. No future states,
  simulator distance, contact result or future termination length are inputs.
- Eight 20D action tokens receive temporal embeddings and three transformer
  decoder layers with self-attention and cross-attention into visual/state/goal
  context. Width 256, eight heads, dropout .1.
- Predict short-horizon success, opening gain, grasp and approach gain. Combine
  predicted components with the existing fixed weights [10,5,.25,1]. Each member
  has **4,512,004 trainable parameters**; the visual encoder remains frozen.
- Train three independent seeds [42,43,44]. Their score average is the predefined
  ensemble, sharing the same frozen visual features.

The encoder is documented in the [official DINOv2 repository](https://github.com/facebookresearch/dinov2).

## Data and losses

The completed dataset has 72 Train episodes / 9,505 labels / 3,805 decision groups,
and 24 Dev episodes / 3,598 labels / 1,450 decision groups. All 24 Eval episodes
remain outside feature extraction, training, normalization and model selection.
Train and Dev houses are disjoint. Indices are hashed before extraction/training.

Keep candidates grouped by their shared observation. Baseline groups have one
candidate; simulator groups have four. Candidate index is not a model input.
Sample episodes equally, then decisions within episodes, so long failures do
not dominate. Fit state/action normalization, score scaling and binary priors
using Train only. Future simulator values appear only as targets. In particular,
candidate action masks use the known lookahead, never the observed future
`simulated_steps` (which would leak termination/success).

Loss: normalized score Smooth L1 + .5 within-state pairwise ranking loss + .2
success BCE + .1 grasp BCE + .2 opening-gain Smooth L1 + .2 approach-gain Smooth L1.
Ignore ranking pairs whose true scores differ by <= .001. Average candidate
losses within each group. AdamW, LR .0003, weight decay .05, five-epoch warmup,
cosine decay, gradient clipping 1.0, mixed precision, batch 32. Up to 80 epochs,
early stopping after 15 epochs without improvement in Dev house-balanced regret.

## Validation and interpretation

CPU tests verify padded actions do not change predictions, candidate permutation
permutes scores, future labels cannot change inputs, Eval is rejected by the
loader, and ranking/regression losses produce finite action-dependent gradients
and decrease in a small fitting test. Real-data parsing verifies 22D state and
candidate/target shapes. The pretrained encoder outputs finite (1,577,768) tokens
for a 336×336 image. A full-size, mixed-precision GPU training smoke runs on real
Train groups during feature extraction before training jobs are allowed to start.

Checkpoint selection uses Dev **house-balanced candidate regret**: the simulator
score of the best candidate minus that of the model-selected candidate, averaged
within episodes and then houses. Also report score RMSE, tie-aware best selection
rate, and first-candidate/random-selection regret. No checkpoint or seed is chosen
using Eval. These offline metrics do not establish task success. The next stage
after training is independent closed-loop Dev rollout testing, followed by final
held-out Eval after configuration freeze.

## Execution

Run: `/bigtemp/xna8aw/part-aware-policy-learning/runs/visual_critic_20261103_02`.

Feature extraction **6803592** -> ensemble training array **6803620**, `0-2%2`
-> final ensemble report **6803621**. At most two training GPUs concurrently.
Failed feature validation blocks training. Each member records best checkpoint,
Dev predictions, epoch history and status. `ensemble_manifest.json` is created
only when all three members complete. `index.html` and `aggregate.json` report
progress; no closed-loop critic rollout has been run by this pipeline.

Run `_01` was a preparation snapshot superseded before any jobs were submitted,
after the raw gripper-state preflight check. No dataset or encoder weights changed.

## Completed training and offline Dev results

Feature extraction, all three members and final report completed with exit 0.
The feature stage took 12m13s; member jobs took 2m58s, 2m42s and 3m24s.

| Member | Epochs run | Selected epoch | Dev house-balanced regret |
| --- | ---: | ---: | ---: |
| Seed 42 | 27 | 12 | 0.05425 |
| Seed 43 | 32 | 17 | 0.03980 |
| Seed 44 | 35 | 20 | 0.03692 |

The predefined ensemble's Dev house-balanced regret is **0.03735**. On the same
716 four-candidate Dev groups, its unweighted mean regret is **0.04712**, versus
**0.12371** for always choosing candidate 0 and **0.10896** for uniform random
selection. Compare these unweighted means with each other; baseline means must
not be compared directly with house-balanced regret. Tie-aware best selection
rate is 51.54% (tolerance .001); random chance is not assumed to be 25%, since
some candidate scores tie. Overall score RMSE is 1.5633 across all Dev groups.

These are offline ranking metrics on Dev states collected by the original two
methods, not learned-critic task success. Eval data was never loaded. All source
and Train/Dev index hashes were rechecked unchanged after training. Checkpoints
are 17.2 MiB each; strict checkpoint reload and finite Dev inference were tested.
`ensemble_manifest.json` identifies all three selected checkpoints and pinned
encoder. Closed-loop Dev rollout evaluation is the next required experiment.
