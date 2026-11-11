# Isolated project environment

Workspace: `/u/xna8aw/workspace/part-aware-policy-learning`.
Large storage: `/bigtemp/xna8aw/part-aware-policy-learning`.
The workspace `data` and `runs` links point into that project store.

The independent Python environment is
`/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces`.
It is separate from Live Robotics Lab. Do not replace it with that project's
environment or upgrade its Torch/MuJoCo dependencies during routine work.

```bash
source env.sh
"$PART_PYTHON" --version
"$PART_PYTHON" -m pose_learning.train --help
```

`env.sh` sets the package/submodule search paths, isolated caches, resource paths,
thread counts and offline model mode. Models already available locally are:

- Frozen MolmoBot: `/bigtemp/xna8aw/LRL/hf_models/MolmoBot-RBY1Multitask`
  (shared read-only pretrained weights).
- Frozen DINOv2: `/bigtemp/xna8aw/part-aware-policy-learning/models/dinov2-base/f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

Pinned runtime versions include Torch 2.7.1+cu128 and MuJoCo 3.4.0. Slurm GPU
wrappers load CUDA 12.8.1 and GCC 12.4.0. Native expert generation additionally
uses the installed CuRobo revision recorded in
`data_preparation/fixtures/planner_build/curobo_build_receipt.json`.
The environment receipts in the project store and the two upstream submodule
commits remain the source of exact dependency identity.

CPU checks use the project interpreter. Rendering and native MolmoBot execution
require an allocated GPU; use the wrappers under the relevant package's
`slurm/` directory. No jobs are submitted automatically by preparation scripts.

Current entry points:

- `grasp.reproduce`: retained local contact or guarded opening diagnostic.
- `data_preparation.collect_interaction_pose_dataset`: official Train subset.
- `data_preparation.prepare_train_test_dataset`: source-backed Train/Test manifests.
- `data_preparation.prepare_test_set`: metadata plan and separately requested assets.
- `pose_learning.prepare_features`: frozen RGB/part features for one declared split.
- `pose_learning.train`: fixed-update pose training on Train only.
- `pose_learning.evaluate`: offline pose prediction errors on Test labels.
- `pose_learning.evaluate_policy`: matched full-start simulator evaluation.

Dataset source folders contain essential symlink targets and frozen provenance.
Use `data_preparation/README.md` before changing their layout.
