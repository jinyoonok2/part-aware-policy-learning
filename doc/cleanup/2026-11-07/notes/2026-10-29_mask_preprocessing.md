# Saved-result mask preprocessing validation

October 29, 2026. CPU job `6754713` checked the existing native pilot's saved
RGB/segmentation/mask data. No new simulation, policy inference or training ran.

## Scope and result

All 12 saved pairs (three cameras at steps 0, 20, 100 and 191) passed the
consistency checks at offline warp seeds 42, 43 and 44: 36 cases in total.
There were 21 visible cases and 15 empty-mask cases. Fractional coverage retained
foreground in every visible case. Empty masks produced finite all-zero weights.

The checkpoint uses SigLIP resize to **378×378**, **14-pixel patches** (27×27),
then **2×2 pooling groups**, padded to 14×14 groups. The preprocessor returns
uint8 RGB because normalization happens inside the model. Mask weights remain
in [0, 1]; they must not receive RGB normalization.

Checks used the actual checkpoint config and source preprocessor:

- Saved binary masks exactly matched their recorded segmentation geometry IDs
  and geometry-type channel.
- RGB transformed with a shared sampled fisheye grid exactly matched the
  upstream RGB function; nearest-label masks matched transformed segmentation IDs.
- Actual pixel-to-patch packing reconstructed the processed RGB exactly.
- Fractional support used the source's pooling index map, including padded IDs.
- A synthetic red region independently traced the upstream RGB path: aligned
  Dice 1.0, deliberately shifted negative-control Dice 0.0. Maximum floating-mask
  versus quantized-RGB interpolation error was 0.00489 (less than 3/255).
- A synthetic absent mask stayed zero and finite through warping, resizing,
  patch coverage and normalized pooling weights.

## Finding for the adapter

Requiring a patch to contain ≥50% foreground would erase the entire part in
**15 of 21 visible seed cases**: five of the seven visible saved camera/frame
pairs at each tested seed. Keep fractional foreground support instead of a
majority binary threshold. Carry an explicit visibility indicator and use zero
part weights plus a global-feature fallback when no support remains.

Nearest labels, soft bilinear masks and area-resized fractional coverage are
saved separately. Area coverage is a geometric support approximation, not a
measurement of model attention or proof of useful learned features.

## Artifacts and limits

The existing pilot's `index.html` now includes a validation section, a per-frame
table and 12 comparison panels. Details and NPZ weights are under:

`/bigtemp/xna8aw/part-aware-policy-learning/runs/native_door_smoke_20261028_01/mask_validation_20261029_01/`

Retain `summary.json`, `cases.json`, `cards.json`, source input hashes, the
validation script snapshot and Slurm accounting. The previous page is backed up
as `index_before_mask_validation_20261029.html`. `mask_validation.json` selects
the validation section when regenerating the main report.

These are four saved timesteps, not every rollout frame. The original rollout
did not save its sampled fisheye parameters, so the offline cases exercise the
same path with controlled samples rather than reconstructing historical inputs.
Training-time augmentations, multi-image/text integration, actual encoder
features, adapter gradients and policy effects remain unvalidated. Pair the
frozen initial conditioning image with its initial mask, not a live-frame mask.
No new masks were added to the original policy's inputs.
