# Scripts

Run commands from the repository root.

```text
scripts/
  local/
    smoke_test.py       # CUDA, CPU physics, optional RGB/segmentation
    setup_encoder.py    # Pinned DINOv2 download and feature verification
  slurm/
    README.md          # Reserved for the later UVA CS deployment
```

Only local scripts are implemented at this stage. Continue local development
and validation before introducing cluster job submission.

```powershell
.venv\Scripts\python.exe scripts\local\smoke_test.py
.venv\Scripts\python.exe scripts\local\smoke_test.py --render
.venv\Scripts\python.exe scripts\local\setup_encoder.py
```

The scripts resolve the repository root from their file locations. Outputs,
assets, and caches stay in the repository's ignored `outputs/`, `data/`, and
`.cache/` directories unless environment variables override the cache/asset paths.
See [the setup guide](../doc/SETUP.md) for installation and current limitations.

Future reusable training and data-processing code should live in a shared Python
package. Local commands and Slurm jobs should call that code rather than maintain
separate implementations of the research methods.
