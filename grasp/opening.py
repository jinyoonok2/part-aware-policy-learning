"""Hinge geometry, trace persistence and the historical joint-contact metric."""
import numpy as np
from scipy.spatial.transform import Rotation

OPENING = dict(joint_success_hold_steps=10)

def hinge_arc(handle, pivot, axis, radians):
    axis = np.asarray(axis, dtype=float).copy()
    axis /= np.linalg.norm(axis)
    rotation = Rotation.from_rotvec(axis * radians).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(pivot) - rotation @ np.asarray(pivot)
    return transform @ handle

def joint_metrics(trace, threshold, open_before_grasp=False):
    grasp_count, joint_count = 0, 0
    verified, joint, sustained = False, False, False
    first_verified, first_joint = None, None
    for t in trace:
        contact = t['active_arm_exact_both_fingers']
        grasp_count = grasp_count + 1 if contact else 0
        if grasp_count >= 13 and not verified:
            verified, first_verified = True, t['step']
        simultaneous = verified and contact and t['opening_fraction'] >= threshold
        joint_count = joint_count + 1 if simultaneous else 0
        if simultaneous:
            joint = True
            if first_joint is None:
                first_joint = t['step']
        sustained |= joint_count >= OPENING['joint_success_hold_steps']
    return dict(steps=len(trace), grasp_verified=verified, first_verified_step=first_verified,
                door_threshold_ever=any(t['opening_fraction'] >= threshold for t in trace),
                final_door_open=trace[-1]['opening_fraction'] >= threshold if trace else False,
                simultaneous_grasp_and_open=joint, sustained_grasp_and_open=bool(sustained),
                first_simultaneous_step=first_joint, door_open_before_grasp_stage=bool(open_before_grasp),
                peak_opening_fraction=max((t['opening_fraction'] for t in trace), default=0.),
                final_opening_fraction=trace[-1]['opening_fraction'] if trace else None)

def keep_state(task, states, decision):
    data = task.env.mj_datas[0]
    states.append(dict(qpos=data.qpos.copy(), ctrl=data.ctrl.copy(), time=float(data.time), decision=decision))

def save_states(path, states):
    if states:
        np.savez_compressed(path, qpos=np.asarray([s['qpos'] for s in states]),
                            ctrl=np.asarray([s['ctrl'] for s in states]), time=np.asarray([s['time'] for s in states]),
                            decision=np.asarray([s['decision'] for s in states]))

def opening_success_ready(grasp_history, opening_trace, threshold):
    """Use the reported joint criterion, excluding any expert approach prefix."""
    return joint_metrics(grasp_history + opening_trace, threshold)['sustained_grasp_and_open']
