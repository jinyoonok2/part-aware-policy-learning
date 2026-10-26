# Part-Aware Policy Learning for Robotic Manipulation

**Status: Ongoing — proposal stage**  
**Author: Jinyoon Kim · UVA IR course project**

Project proposal: [PDF](doc/IR_course_Project_Proposal.pdf).

This project investigates whether explicitly representing a task-relevant object part improves robotic manipulation robustness and generalization compared with policies using only global visual observations.

For tasks such as opening a drawer or cabinet, the robot primarily needs to interact with a functional part such as a handle. The study asks whether focusing visual features on that part helps a policy transfer to unseen object instances and configurations.

## Planned approach

- Compare a baseline policy using global visual features and robot state with a part-aware policy that also receives target-part features.
- Use simulator ground-truth segmentation to identify the relevant part and pool features from its image region.
- Extract visual features with a pretrained encoder such as DINOv2 and train imitation-learning policies from demonstrations, starting with behavior cloning and potentially extending to an Action Chunking Transformer (ACT).
- Use ManiSkill3 articulated-object manipulation tasks involving drawers or cabinets.

The initial study fixes the task and target object to isolate the effect of part-level information. Language grounding is a future extension.

## Planned evaluation

- Measure task success on seen and unseen object instances.
- Evaluate robustness to changes in object geometry, part location, object pose, and initial configuration.
- Compare global-only and part-aware policies under matched training and evaluation conditions.

## Future extensions

Language instructions could enable target-object and target-part selection in multi-object environments. MolmoSpaces may also be explored for greater object and geometry diversity.

Implementation, setup instructions, and experimental results will be added as the project develops.
