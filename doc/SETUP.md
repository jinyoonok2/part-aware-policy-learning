# Project setup

This project compares global-only and part-aware visual imitation-learning
policies for drawer manipulation. The proposal is [available here](IR_course_Project_Proposal.pdf).

## Stack and scope

- Simulator: ManiSkill3 3.0.1 with SAPIEN 3.0.3.
- Task: `OpenCabinetDrawer-v1`, initially using the supplied Fetch robot.
- Object assets: PartNet-Mobility cabinets. These are articulated models, not
  robot demonstrations.
- Learning: PyTorch, initially an MLP trained with behavior cloning.
- Encoder: frozen DINOv2 ViT-S/14. Pool global patch features and, for the
  part-aware policy, patch features weighted by the target-part mask.
- Extension: ACT once the initial comparison and dataset pipeline work.

Use the same encoder, demonstrations, controller, and evaluation episodes for
both policies. Compare capacity-matched heads so that extra parameters do not
explain the result. Keep privileged object poses and handle goal coordinates
out of policy inputs; the part mask is the explicit privileged input being studied.

## Platform choice

The development laptop runs Windows with an NVIDIA GeForce RTX 3050 Laptop GPU
(6 GB VRAM; driver 592.82 at setup). Native Windows supports CPU physics and
GPU rendering in ManiSkill; parallel GPU physics is supported on Linux/NVIDIA.
CUDA policy training is separate from the physics backend and can run on Windows.
Use one environment and small batches initially. Prefer a native Linux GPU
machine for large-scale expert collection and parallel evaluation.

Python 3.10 is used here. CUDA-enabled PyTorch 2.5.1/torchvision 0.20.1 wheels
are selected as a conservative starting pair. A separate CUDA toolkit installation
is not needed for these wheels; a compatible NVIDIA driver is required.

## Windows installation

Executable checks currently live in `scripts/local/`. `scripts/slurm/` is reserved
for a later UVA CS deployment; this stage implements and validates local scripts
only. See [the script layout](../scripts/README.md).

Run PowerShell commands from the repository root. `uv` is a package/environment
manager; an existing Python 3.10 plus `python -m venv .venv` also works.

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip check --python .venv\Scripts\python.exe
```

The local workspace uses `uv --cache-dir C:\Users\jinyo\workspace\.tools\uv-cache`
and its existing workspace Python 3.10 interpreter to stay within writable paths.
Use `.venv\Scripts\python.exe` directly; activation is optional.

Keep downloaded data and caches inside the ignored project directories:

```powershell
$env:MS_ASSET_DIR = Join-Path $PWD 'data\assets'
$env:TORCH_HOME = Join-Path $PWD '.cache\torch'
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
New-Item -ItemType Directory -Force data\assets,outputs,.cache | Out-Null
.venv\Scripts\python.exe scripts\local\smoke_test.py
.venv\Scripts\python.exe scripts\local\smoke_test.py --render
```

The first test checks imports, CUDA availability, and CPU-physics stepping with
`PickCube-v1`. The second checks RGB and segmentation rendering. Random actions
only verify execution; they are not expert demonstrations or success tests.

On Windows the smoke test explicitly selects `render_backend="cpu"`. This is
the working image-readback path here: the default CUDA image-buffer path crashes
inside SAPIEN's `get_picture_cuda`. CUDA tensor computation and policy training
still work. Linux defaults to the GPU render backend. Override the choice with
`--render-backend cpu` or `--render-backend gpu` when diagnosing another machine.

`requirements-windows.lock.txt` records the complete installed Windows package
versions, including the `+cu121` PyTorch builds. It is an environment inventory,
not a portable Linux lockfile; obtain those CUDA wheels from the index above.

## Linux installation for parallel experiments

The following is a future installation reference, not a validated UVA CS Slurm
configuration. Cluster-specific setup and job scripts will be added on the server.

On a native Linux/NVIDIA machine with Python 3.10, a compatible NVIDIA driver,
and Vulkan configured according to the official guide:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
export MS_ASSET_DIR="$PWD/data/assets"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
python scripts/local/smoke_test.py --render
```

The smoke test deliberately uses CPU physics on both systems. Validate GPU
physics separately on Linux before collecting data in parallel. Do not assume
WSL provides equivalent rendering/physics support; consult ManiSkill's platform table.

## Drawer assets and data collection

After the basic rendering test succeeds:

```powershell
.venv\Scripts\python.exe -m mani_skill.utils.download_asset OpenCabinetDrawer-v1
.venv\Scripts\python.exe scripts\local\smoke_test.py --drawer --render
```

Asset downloads can be large. Request only the needed cabinet asset group,
not every ManiSkill asset. Inspect the downloader's prompt before accepting.
The drawer smoke test validates loading and stepping, not successful opening.

**Current blocker (2026-10-25):** the official cabinet asset server
`storage1.ucsd.edu` timed out on the first download. No cabinet meshes were
installed, so drawer loading, handle masks, and drawer demonstrations are not
verified. The same outage is tracked in [ManiSkill issue #1489](https://github.com/mani-skill/ManiSkill/issues/1489).
Retry the command when the upstream service recovers, or use a maintainer-provided
mirror of the processed ManiSkill assets. Raw PartNet-Mobility files are not
automatically interchangeable with ManiSkill's processed collision assets.
Use `-y` only for unattended downloads into a fresh destination; the downloader
can replace existing model folders. The task resolves to 57 cabinet model assets.

ManiSkill's current task table lists no supplied demonstrations for its cabinet
door/drawer tasks. Do not assume `download_demo OpenCabinetDrawer-v1` will work.
Older ManiSkill2 trajectories require a separate compatibility check. For this
project, plan to collect successful demonstrations with a scripted controller or
a state-based RL expert using the same pinned simulator and action controller.

Record RGB, robot proprioception, actions, success labels, cabinet model ID,
target link ID, seeds, timestep, controller settings, and simulator version.
Save simulator states for replay and target masks for representation extraction.
ManiSkill trajectories use HDF5 plus JSON metadata; replay can regenerate visual
observations when the environment/version is compatible.

Segmentation IDs identify actors/links, which may include both a drawer front
and its handle. Verify the mask visually on each selected model. A handle-only
mask may require visual-part metadata or an additional rendering pass. Label a
whole-link mask honestly if that is the representation used. Define an explicit
fallback for an invisible target (for example, a zero part feature plus a
visibility flag).

## Encoder and experimental milestones

Install and verify the encoder independently of cabinet availability:

```powershell
.venv\Scripts\python.exe scripts\local\setup_encoder.py
```

This caches the source and weights locally, checks a CUDA forward pass when
available, and records revision/checksum details in `outputs/encoder_setup.json`.
The equivalent model-loading operation is:

```python
import torch
encoder = torch.hub.load(
    'facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'dinov2_vits14', trust_repo=True,
)
encoder = encoder.eval().requires_grad_(False).to('cuda')
```

This downloads executable upstream model code and weights. The script pins the
source revision and records the weight checksum. Align mask resizing with the image transform
and the encoder patch grid. Cache frozen features to reduce repeated GPU work.

1. Verify one cabinet loads and its intended target mask is correct.
2. Produce and replay one successful expert demonstration.
3. Collect a small pilot dataset and fit both behavior-cloning policies.
4. Scale to multiple training cabinets and disjoint validation/test model IDs.
5. Report success on seen and unseen cabinets, robustness perturbations, and
   multiple training seeds. Keep full episodes together when splitting data.
6. Add ACT only after the simpler experiment is established.

The installation alone does not supply a drawer expert, dataset, or trained policy.
Do not commit assets, trajectories, model weights, or local virtual environments.

## References

- [ManiSkill installation and platform support](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [Drawer task and demonstration availability](https://maniskill.readthedocs.io/en/latest/tasks/mobile_manipulation/index.html)
- [Current demonstration repository](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)
- [Observations and segmentation IDs](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)
- [Trajectory replay](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)
- [PyTorch versioned installation commands](https://pytorch.org/get-started/previous-versions/)
- [DINOv2 models](https://github.com/facebookresearch/dinov2)
- [ManiSkill ACT baseline](https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/act)

## Local installation results

Verified on 2026-10-25 using the project `.venv`:

| Check | Result |
| --- | --- |
| Python | 3.10.22 |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| PyTorch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| NumPy / OpenCV / h5py | 1.26.4 / 4.11.0.86 / 3.16.0 |
| Dependency consistency | All 84 installed packages compatible |
| CUDA | RTX 3050 detected; tensor computation passed |
| CPU physics | PickCube reset and five random-action steps passed |
| RGB + segmentation | Passed with Windows CPU readback; RGB shape 128 x 128 x 3 |
| DINOv2 | Frozen CUDA forward pass passed; patch features 1 x 256 x 384 |
| Cabinet asset download | Blocked by upstream connection timeout |
| Drawer opening / handle mask | Not yet verified |

The encoder checkpoint SHA-256 is
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
Camera output is saved in `outputs/PickCube-v1_base_camera.png`.
SAPIEN warns that optional Pinocchio is unavailable; DINOv2 warns that optional
xFormers is unavailable. Neither prevented the checks above.

The Windows execution sandbox initially stalled native-library loading; hardware
checks completed when run with driver access outside that restricted sandbox.
No system driver or operating-system changes were made.
