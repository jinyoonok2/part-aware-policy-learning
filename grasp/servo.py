"""The retained 20 mm closure gate plus bounded base-recovery controller.

REFERENCE_PARAMETERS reproduces the earlier 7/10 local contact comparison.
BEST_PARAMETERS is the combined controller that reached 10/10 on those ten
expert-pregrasp Train starts. This is not a full-task 10/10 opening claim.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .runtime import physics_step, clone_task, integration_state
from .contacts import summarize

REFERENCE_PARAMETERS = {'steps': 120, 'position_threshold_m': 0.012, 'orientation_threshold_deg': 5.0, 'arm_step_rad': 0.04, 'base_step_m': 0.015, 'max_base_displacement_m': 0.6, 'close_timeout_steps': 10, 'reopen_steps': 5, 'hold_steps': 10, 'max_attempts': 3, 'damping': 0.001, 'gain': 0.7}
PARAMETERS = REFERENCE_PARAMETERS
BEST_PARAMETERS = {'steps': 120, 'position_threshold_m': 0.02, 'orientation_threshold_deg': 5.0, 'arm_step_rad': 0.04, 'base_step_m': 0.015, 'max_base_displacement_m': 0.6, 'close_timeout_steps': 10, 'reopen_steps': 5, 'hold_steps': 10, 'max_attempts': 3, 'damping': 0.001, 'gain': 0.7, 'base_recovery_enabled': True, 'recovery_stall_steps': 15, 'recovery_min_error_m': 0.04, 'recovery_max_move_m': 0.25, 'recovery_max_base_displacement_m': 0.85, 'recovery_max_repositions': 2}

def body_pose(data, body):
    out = np.eye(4)
    out[:3, 3] = data.xpos[body]
    out[:3, :3] = data.xmat[body].reshape(3, 3)
    return out

def moving_goal(handle, relative, attempt=0):
    target = handle @ relative
    # Fixed, small tool-frame corrections after an unsuccessful closure.
    offset = (0., .005, -.005)[min(attempt, 2)]
    target[:3, 3] += target[:3, :3] @ np.array([offset, 0., 0.])
    return target

def servo_delta(jacobian, actual, target, mobile, cfg):
    error = np.r_[target[:3, 3] - actual[:3, 3],
                  Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]
    error *= cfg['gain']
    # Bound task-space error as well as joint targets.
    for part, maximum in ((slice(0, 3), .03), (slice(3, 6), .15)):
        norm = np.linalg.norm(error[part])
        if norm > maximum:
            error[part] *= maximum / norm
    weights = np.array([1.] * 7 + ([.1, .1] if mobile else []))
    weighted = jacobian * weights
    delta = weighted.T @ np.linalg.solve(weighted @ jacobian.T + cfg['damping'] * np.eye(6), error)
    limits = np.array([cfg['arm_step_rad']] * 7 + ([cfg['base_step_m']] * 2 if mobile else []))
    return np.clip(delta, -limits, limits)

def bounded_base_goal(base_start, current, tcp, target, cfg):
    """Plan one bounded XY reposition toward the remaining TCP position error."""
    gap = np.asarray(target)[:2, 3] - np.asarray(tcp)[:2, 3]
    length = np.linalg.norm(gap)
    if length <= cfg.get('recovery_min_error_m', .04):
        return None
    goal = np.asarray(current).copy()
    distance = min(max(length - .025, 0.), cfg.get('recovery_max_move_m', .25))
    goal[:2] += gap * distance / length
    offset = goal[:2] - np.asarray(base_start)[:2]
    total = np.linalg.norm(offset)
    cap = cfg.get('recovery_max_base_displacement_m', .85)
    if total > cap:
        goal[:2] = np.asarray(base_start)[:2] + offset * cap / total
    goal[2] = base_start[2]
    if np.linalg.norm(goal[:2] - np.asarray(current)[:2]) < .01:
        return None
    return goal

class BaseRecovery:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []
        self.goal = None
        self.events = []

    def update(self, measurement, phase, step, base_start, tcp, target):
        current = np.asarray(measurement['base_qpos'])
        if phase != 'approach':
            self.goal = None
            self.history = []
            return None
        if self.goal is not None:
            if np.linalg.norm(current[:2] - self.goal[:2]) <= .02:
                self.events[-1]['finished_step'] = step
                self.goal, self.history = None, []
            else:
                return self.goal
        error = measurement['tcp_error']['position_m']
        self.history.append(error)
        window = self.cfg.get('recovery_stall_steps', 15)
        self.history = self.history[-window:]
        stalled = len(self.history) == window and self.history[0] - self.history[-1] < .005
        at_limit = np.linalg.norm(current[:2] - base_start[:2]) >= self.cfg['max_base_displacement_m'] - .02
        if (error > self.cfg.get('recovery_min_error_m', .04) and (stalled or at_limit)
                and len(self.events) < self.cfg.get('recovery_max_repositions', 2)):
            self.goal = bounded_base_goal(base_start, current, tcp, target, self.cfg)
            if self.goal is not None:
                self.events.append(dict(step=step, reason='normal_base_limit' if at_limit else 'stalled_approach',
                                        initial_error_m=error, from_base=current.copy(), target_base=self.goal.copy()))
                self.history = []
        return self.goal

class ApproachCloseVerify:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = 'approach'
        self.attempt = 0
        self.phase_steps = 0
        self.contact_steps = 0
        self.hold_steps = 0
        self.first_verified_step = None

    def update(self, measurement, step):
        if self.phase == 'exhausted':
            return False
        contact = measurement['active_arm_exact_both_fingers']
        self.contact_steps = self.contact_steps + 1 if contact else 0
        near = (measurement['tcp_error']['position_m'] <= self.cfg['position_threshold_m'] and
                measurement['tcp_error']['orientation_deg'] <= self.cfg['orientation_threshold_deg'])
        if self.phase == 'approach' and (near or contact):
            self.phase, self.phase_steps = 'close', 0
        elif self.phase == 'reopen' and self.phase_steps >= self.cfg['reopen_steps']:
            self.phase, self.phase_steps, self.contact_steps = 'approach', 0, 0
        elif self.phase == 'close':
            if self.contact_steps >= 3:
                self.phase, self.phase_steps, self.hold_steps = 'verify', 0, 0
            elif self.phase_steps >= self.cfg['close_timeout_steps']:
                self.attempt += 1
                self.phase = 'exhausted' if self.attempt >= self.cfg['max_attempts'] else 'reopen'
                self.phase_steps, self.contact_steps = 0, 0
        elif self.phase in ('verify', 'hold'):
            if contact:
                self.hold_steps += 1
                if self.hold_steps >= self.cfg['hold_steps']:
                    self.phase = 'hold'
                    if self.first_verified_step is None:
                        self.first_verified_step = step
            else:
                self.phase, self.phase_steps, self.contact_steps = 'close', 0, 0
        self.phase_steps += 1
        return self.phase in ('close', 'verify', 'hold')

def reconstruct_handle(task, probe, source_door, target):
    import mujoco
    from molmo_spaces.env.data_views import Door
    data = copy.copy(task.env.mj_datas[0])
    door = Door(task.config.task_config.door_body_name, data)
    index = door.get_hinge_joint_index()
    recorded_angle = float(np.asarray(source_door['joint_angle']).reshape(-1)[0])
    door.set_joint_position(index, recorded_angle)
    mujoco.mj_forward(task.env.mj_model, data)
    recorded_handle = body_pose(data, probe.selected)
    reconstructed_visual = door.get_handle_pose()[:3]
    mismatch = float(np.linalg.norm(reconstructed_visual - np.asarray(source_door['handle_position'])))
    audit = dict(recorded_hinge_rad=recorded_angle,
                 native_hinge_rad=float(task.door_object.get_joint_position(index)),
                 recorded_handle_position=source_door['handle_position'],
                 reconstructed_handle_position=reconstructed_visual,
                 position_reconstruction_error_m=mismatch,
                 recorded_selected_handle_world=recorded_handle,
                 tcp_in_handle=np.linalg.inv(recorded_handle) @ target,
                 current_handle_world=body_pose(task.env.mj_datas[0], probe.selected))
    assert mismatch < .005, f'Source handle FK mismatch {mismatch:.4f}m; cannot trust relative label'
    return audit

def run_contact(task, arm, probe, relative, cfg=None, return_branch=False, state_callback=None):
    cfg = dict(BEST_PARAMETERS if cfg is None else cfg)
    mobile = True
    parent_state = integration_state(task.env.mj_model, task.env.mj_datas[0])
    branch = clone_task(task)
    assert np.array_equal(parent_state, integration_state(branch.env.mj_model, branch.env.mj_datas[0]))
    robot = branch.env.robots[0]
    view = robot.robot_view
    arm_group, gripper = arm + '_arm', arm + '_gripper'
    command = {k: v.copy() for k, v in view.get_noop_ctrl_dict().items() if k in robot.controllers}
    base_start = view.get_move_group('base').joint_pos.copy()
    limits = view.get_move_group(arm_group).joint_pos_limits
    machine = ApproachCloseVerify(cfg)
    recovery = BaseRecovery(cfg) if mobile and cfg.get('base_recovery_enabled', False) else None
    trace = []
    for step in range(cfg['steps']):
        handle = body_pose(branch.env.mj_datas[0], probe.selected)
        target = moving_goal(handle, relative, machine.attempt)
        before = probe.measure(branch, arm, target)
        closed = machine.update(before, step)
        target = moving_goal(handle, relative, machine.attempt)
        command[gripper] = np.array([100. if closed else -100.])
        reposition = recovery.update(before, machine.phase, step, base_start,
                                     view.get_move_group(gripper).leaf_frame_to_world, target) if recovery else None
        if machine.phase != 'exhausted':
            groups = [arm_group, 'base'] if mobile else [arm_group]
            jac = view.get_jacobian(gripper, groups)
            if mobile:
                jac = jac[:, :9]  # Seven arm joints, base X/Y; yaw is held.
            delta = servo_delta(jac, view.get_move_group(gripper).leaf_frame_to_world, target, mobile, cfg)
            command[arm_group] = np.clip(view.get_move_group(arm_group).joint_pos + delta[:7], limits[:, 0], limits[:, 1])
            if mobile:
                base = view.get_move_group('base').joint_pos.copy()
                if reposition is None:
                    base[:2] += delta[7:9]
                else:
                    gap = reposition[:2] - base[:2]
                    norm = np.linalg.norm(gap)
                    base[:2] += gap * min(1., cfg['base_step_m'] / max(norm, 1e-12))
                offset = base[:2] - base_start[:2]
                length = np.linalg.norm(offset)
                cap = cfg.get('recovery_max_base_displacement_m', .85) if recovery and recovery.events else cfg['max_base_displacement_m']
                if length > cap:
                    base[:2] = base_start[:2] + offset * cap / length
                base[2] = base_start[2]
                command['base'] = base
        physics_step(branch, command)
        target_after = moving_goal(body_pose(branch.env.mj_datas[0], probe.selected), relative, machine.attempt)
        entry = probe.measure(branch, arm, target_after)
        entry.update(step=step + 1, phase=machine.phase,
                     attempt=machine.attempt + 1, base_displacement_m=float(np.linalg.norm(view.get_move_group('base').joint_pos[:2] - base_start[:2])))
        if recovery:
            entry.update(base_recovery_active=recovery.goal is not None, base_repositions=len(recovery.events))
        trace.append(entry)
        if state_callback is not None:
            state_callback(branch, entry)
        if cfg.get('stop_when_verified', False) and machine.phase == 'hold' and entry['active_arm_exact_both_fingers']:
            break
    result = summarize(trace)
    result.update(first_verified_step=machine.first_verified_step, final_phase=machine.phase,
                  attempts=min(machine.attempt + 1, cfg['max_attempts']),
                  max_base_displacement_m=max(t['base_displacement_m'] for t in trace),
                  final_10_active_grasp_fraction=float(np.mean([t['active_arm_exact_both_fingers'] for t in trace[-10:]])))
    if recovery:
        result['base_recovery_events'] = recovery.events
    assert np.array_equal(parent_state, integration_state(task.env.mj_model, task.env.mj_datas[0])), 'Branch modified parent'
    if return_branch:
        return trace, result, branch
    return trace, result
