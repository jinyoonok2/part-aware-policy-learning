# Retained grasp and opening controller

This package keeps the combined 20 mm closure gate and bounded base recovery
that improved local bilateral grasp contact from 7/10 to 10/10 on the same ten
Train pilot cases. It also keeps force-gated opening and grip recovery, the
native MolmoBot action bridge, and the explicit grasp-retention metric.

The 10/10 result is a local contact diagnostic after the expert approach. It
uses expert TCP labels and live simulator handle/contact state. It does not
mean 10/10 door opening from the original episode start. The later full-start
approach was harder; neither learned pose prediction nor a full-task improvement
has been established by this result.

The count comes from the separate post-hoc numerical review in
`fixtures/recovery_10/results.json`, whose state is `review_passed`. Original
workers 02, 03 and 05 saved all four complete 120-decision traces, then failed
their reference-reproduction assertion: final TCP-error agreement had to be
closer than 1e-8 m. Their original status remains `runtime_or_label_error`.
The review required identical per-step contact flags, phases and attempts,
source/configuration hashes, and all 120 decisions. It also checked complete
reference trajectories within 0.1 mm position/base, 0.05 degrees TCP orientation
and 0.0001 radians base yaw. All ten passed that disclosed revised audit;
the largest measured TCP-position difference was approximately 0.023 mm.
`audit_policy.json` records the original failures, revised tolerances and input
hashes. The reviewed 7/10 and 10/10 contact counts do not relabel failed workers
as successful jobs or change the original contact-success definition.

## Files

- `servo.py`: combined controller; the older 12 mm reference is retained only
  for reproducing the 7/10 comparison. The two intermediate ablations are gone.
- `firm.py`: approach, force verification, opening, and grip-loss recovery.
- `native_adapter.py`: bounded absolute local targets to native MolmoBot commands.
- `contacts.py`, `runtime.py`, `opening.py`: physical observations and shared helpers.
- `metrics.py`: historical retention policy and the separately named onset policy.
- `fixtures/recovery_10/`: frozen ten-case episode/configuration/source hashes,
  reviewed result counts, and audit policy.
- `reproduce.py`: contact reproduction and a full-start opening diagnostic.

Run commands from the project root with the dedicated project Python environment.
MuJoCo scene rendering requires an allocated GPU and the project asset cache;
`grasp/slurm/reproduce.sh` sets those environment variables. No job is submitted
by importing this package or preparing data.

```bash
# Reproduce the local reference-versus-combined contact comparison.
PART_GRASP_RUN=/bigtemp/xna8aw/part-aware-policy-learning/runs/grasp_contact_reproduction \
  sbatch grasp/slurm/reproduce.sh --mode contact --start expert-pregrasp --with-reference

# Test guarded opening from the original closed-door start, without approach replay.
PART_GRASP_RUN=/bigtemp/xna8aw/part-aware-policy-learning/runs/grasp_full_start \
  sbatch grasp/slurm/reproduce.sh --mode opening --start original
```

Supply `--dataset PATH` if the preserved raw-source root changes. The manifest
stores relative HDF5 paths and verifies each file's original SHA-256. A separate
output case directory is written for each array task. Results distinguish local
contact, simultaneous grasp/opening, historical retention, and onset retention.
The full opening budget is 500 decisions, including an expert approach prefix
when one is explicitly requested.

CPU checks:

```bash
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python \
  -m unittest discover -s grasp/tests -v
```

Recount the retained original contact traces without simulation:

```bash
"$PART_PYTHON" -m grasp.audit_results
```
