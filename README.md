# Part-Aware Policy Learning for Robotic Manipulation

**Status: Ongoing — proposal stage**  
**Author: Jinyoon Kim · UVA IR course project**

Project proposal: [PDF](doc/IR_course_Project_Proposal.pdf).

Current topic and experiments: [Research plan](doc/RESEARCH_PLAN.md).

Setup status and server handoff: [Setup notes](doc/SETUP.md).

Future server scripts: [Planned Slurm layout](scripts/README.md).

This project investigates whether explicitly representing a task-relevant object part improves robotic manipulation robustness and generalization compared with policies using only global visual observations.

For tasks such as opening a drawer or cabinet, the robot primarily needs to interact with a functional part such as a handle. The study asks whether focusing visual features on that part helps a policy transfer to unseen object instances and configurations.

## Planned approach

- Adapt a compatible pretrained MolmoBot policy using three staged designs: a part-aware adapter, a residual action module, and phase-specific experts for approach, grasping, and opening.
- Use simulator ground-truth segmentation to identify the relevant part and pool features from its image region.
- Initially freeze the MolmoBot backbone and train added modules using compatible demonstration supervision. Compare against the original policy and matched adaptations without part information.
- Use accessible visual features or a separate frozen encoder such as DINOv2 for part extraction, subject to architecture verification.
- Plan to use MolmoSpaces with MuJoCo for articulated manipulation and compatible MolmoBot demonstrations or generated trajectories. Installation and verification will continue on the UVA CS Linux server.

The initial study fixes the task and target object to isolate the effect of part-level information. Language grounding is a future extension.

## Planned evaluation

- Measure task success on seen and unseen object instances.
- Evaluate robustness to changes in appearance, lighting, camera viewpoint, object geometry, part location, and initial configuration, with additional tests using imperfect target-part masks.
- Compare all three designs and their controls under matched training and evaluation conditions, including phase-dependent performance and routing errors.

## Future extensions

Language instructions could enable target-object and target-part selection in multi-object environments.

The proposal PDF records the original ManiSkill-based plan. The current direction is part-aware visual robustness in MolmoSpaces. The earlier Windows environment and local verification scripts have been removed; framework installation is deferred to the UVA CS server.

Implementation, setup instructions, and experimental results will be added as the project develops.
