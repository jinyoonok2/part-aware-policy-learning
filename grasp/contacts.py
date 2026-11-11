"""Finger force, target handle and robot-state observations."""
import numpy as np
from .runtime import pose_error

def descendants(model, root):
    result = set()
    for body in range(model.nbody):
        parent = body
        while parent:
            if parent == root:
                result.add(body)
                break
            parent = int(model.body_parentid[parent])
    return result

class ContactProbe:
    def __init__(self, task, arm):
        model = task.env.mj_model
        self.selected_name = task.get_task_objects()['door_handle']
        self.selected = model.body(self.selected_name).id
        self.handles = {task.door_object.handle_name(i): descendants(model, model.body(task.door_object.handle_name(i)).id)
                        for i in range(task.door_object.num_handles)}
        self.finger_names = [f'robot_0/ee_finger_{arm[0]}{i}' for i in (1, 2)]
        self.finger_roots = [model.body(name).id for name in self.finger_names]
        self.fingers = [descendants(model, body) for body in self.finger_roots]

    def measure(self, task, arm, target):
        import mujoco
        model, data = task.env.mj_model, task.env.mj_datas[0]
        view = task.env.robots[0].robot_view
        handle_pose = np.eye(4)
        handle_pose[:3, 3] = data.xpos[self.selected]
        handle_pose[:3, :3] = data.xmat[self.selected].reshape(3, 3)
        hits = {name: [False, False] for name in self.handles}
        exact = [False, False]
        rows = []
        for i in range(data.ncon):
            contact = data.contact[i]
            bodies = [int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])]
            for finger, body_set in enumerate(self.fingers):
                if bodies[0] in body_set:
                    other, active = bodies[1], bodies[0]
                elif bodies[1] in body_set:
                    other, active = bodies[0], bodies[1]
                else:
                    continue
                exact[finger] |= active == self.finger_roots[finger] and other == self.selected
                for name, handle_set in self.handles.items():
                    hits[name][finger] |= other in handle_set
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                rows.append({'finger': finger + 1, 'finger_body': model.body(active).name,
                             'other_body': model.body(other).name,
                             'geom_pair': [model.geom(contact.geom1).name, model.geom(contact.geom2).name],
                             'distance_m': float(contact.dist), 'normal_force_N': float(force[0]),
                             'active_constraint': bool(contact.efc_address >= 0)})
        return {'tcp_error': pose_error(view.get_move_group(arm + '_gripper').leaf_frame_to_world, target),
                'tcp_world': view.get_move_group(arm + '_gripper').leaf_frame_to_world.copy(),
                'selected_handle_world': handle_pose, 'selected_handle': self.selected_name,
                'benchmark_grasped': bool(task.handle_grasped()),
                'active_arm_exact_both_fingers': bool(all(exact)),
                'selected_handle_descendant_contacts': hits[self.selected_name],
                'both_on_selected_descendants': bool(all(hits[self.selected_name])),
                'handle_contacts': hits, 'finger_contacts': rows,
                'gripper_joint_pos': view.get_move_group(arm + '_gripper').joint_pos.copy(),
                'gripper_aperture_m': view.get_move_group(arm + '_gripper').inter_finger_dist,
                'gripper_ctrl': view.get_move_group(arm + '_gripper').ctrl.copy(),
                'base_qpos': view.get_move_group('base').joint_pos.copy(),
                'opening_fraction': float(task.get_door_open_percentage())}

def summarize(trace, closure_step=None):
    selected = trace if closure_step is None else trace[closure_step:]
    early = selected[:10]
    return {'steps': len(trace), 'close_after_steps': closure_step,
            'any_benchmark_grasp': any(t['benchmark_grasped'] for t in trace),
            'first_grasp_step': next((t['step'] for t in trace if t['benchmark_grasped']), None),
            'first_second_after_close_grasp_fraction': float(np.mean([t['benchmark_grasped'] for t in early])) if early else None,
            'final_10_grasp_fraction': float(np.mean([t['benchmark_grasped'] for t in trace[-10:]])),
            'final_benchmark_grasp': trace[-1]['benchmark_grasped'],
            'final_descendant_grasp': trace[-1]['both_on_selected_descendants'],
            'final_tcp_error': trace[-1]['tcp_error'],
            'final_aperture_m': trace[-1]['gripper_aperture_m'],
            'handle_displacement_m': float(np.linalg.norm(np.asarray(trace[-1]['selected_handle_world'])[:3, 3] - np.asarray(trace[0]['selected_handle_world'])[:3, 3])),
            'exact_flag_disagreement_steps': sum(t['benchmark_grasped'] != t['active_arm_exact_both_fingers'] for t in trace),
            'descendant_only_grasp_steps': sum(t['both_on_selected_descendants'] and not t['active_arm_exact_both_fingers'] for t in trace)}
