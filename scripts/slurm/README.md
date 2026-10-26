# UVA CS Slurm deployment (planned)

This directory is reserved for future UVA CS server launchers and Slurm batch
scripts. No cluster setup, submission scripts, or jobs are implemented yet.
The current implementation is under [`../local/`](../local/).

When moving to the server, confirm its actual partition/account requirements,
GPU and driver availability, Python/module setup, storage paths, network access,
and support for headless Vulkan rendering. Then create and validate a Linux
environment there; do not copy the Windows `.venv` or treat the Windows package
inventory as a Linux lockfile.

The first cluster milestone will be a scheduled smoke test covering CUDA,
physics, RGB/segmentation rendering, and the encoder before data collection or
training jobs. Keep resource requests and storage/cache paths in cluster
launchers, and reuse the same project algorithms and experiment configuration.
