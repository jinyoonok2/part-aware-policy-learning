# Setup status and UVA CS handoff

Local framework installation was canceled on 2026-10-26. The project-local
ManiSkill/SAPIEN virtual environment, verification scripts, downloaded model
cache, and generated outputs have been removed. No MolmoSpaces or MolmoBot
installation was performed on this laptop.

## Planned direction

Study whether task-relevant part features improve visual robustness in
articulated manipulation. Compare global-only and part-aware policies using
matched demonstrations, encoders, training budgets, and evaluation episodes.
Evaluate controlled visual changes and imperfect part localization.

- Environment: MolmoSpaces with MuJoCo.
- Data: a small compatible subset of MolmoBot demonstrations or trajectories
  generated using MolmoSpaces.
- Base policy: a compatible pretrained MolmoBot checkpoint, initially with its
  backbone frozen.
- Adaptation: a part-aware adapter, a residual action module, and phase-specific
  experts, implemented and compared in stages with matched controls.
- Execution: UVA CS Linux server; Slurm scripts will be implemented there.

The [proposal PDF](IR_course_Project_Proposal.pdf) describes the earlier
ManiSkill-based plan and is retained as the original proposal.

## Continue on the server

1. Inspect the user's existing MolmoSpaces/MolmoBot installation and source revisions.
2. Confirm Python, CUDA, rendering, storage paths, and Slurm resource requirements.
3. Reuse or create an isolated Linux environment and record compatible versions.
4. Verify one articulated task, RGB rendering, and target-handle identification.
5. Inspect a demonstration and validate observation/action alignment.
6. Verify pixel-level part masks; visibility metadata alone is insufficient.
7. Implement server smoke tests before collecting data or training policies.
8. Inspect action-generation interfaces, GPU memory requirements, and available
   phase labels before implementing the three designs in the [research plan](RESEARCH_PLAN.md).

No package pins or runnable installation commands are prescribed here yet.
Choose them against the existing server setup and upstream compatibility
requirements. No cluster jobs have been submitted by this project.

## Upstream references

- [MolmoSpaces](https://github.com/allenai/molmospaces)
- [MolmoBot](https://github.com/allenai/MolmoBot)
- [MolmoSpaces data generation](https://github.com/allenai/molmospaces/blob/main/scripts/datagen/README.md)
- [MolmoSpaces trajectory format](https://github.com/allenai/molmospaces/blob/main/docs/data_format.md)
