"""Guarded physical approach, force verification, opening and grip recovery."""
import numpy as np
from scipy.spatial.transform import Rotation

from .metrics import POLICY, force_grasp, retained_grasp_opening
from .opening import hinge_arc, keep_state
from .servo import BaseRecovery, body_pose, moving_goal, servo_delta


CONTROL = dict(pregrasp_offset_m=.06, insertion_step_m=.005, position_gate_m=.02,
               orientation_gate_deg=5., close_timeout_steps=10, reopen_steps=5,
               max_attempts=3, opening_step_rad=.015, cautious_step_rad=.005,
               cautious_force_N=5., slip_pause_m=.005, slip_resume_m=.003,
               rotation_pause_deg=5., rotation_resume_deg=3., opening_base_cap_m=1.)


class FirmGraspControl:
    def __init__(self, cfg=CONTROL, policy=POLICY):
        self.cfg, self.policy = cfg, policy
        self.phase, self.offset = 'pregrasp', cfg['pregrasp_offset_m']
        self.ticks, self.force_streak, self.attempt = 0, 0, 0
        self.relative_anchor = None
        self.provisional = None
        self.first_verified_step = None
        self.failure_reason = None
        self.recoveries = 0
        self.last_forces = [0., 0.]

    def transition(self, phase):
        if phase != self.phase:
            self.phase, self.ticks = phase, 0

    def reopen(self):
        self.attempt += 1
        self.force_streak = 0
        if self.attempt >= self.cfg['max_attempts']:
            self.failure_reason = 'grasp_retries_exhausted'
            self.transition('failed')
        else:
            self.transition('reopen')

    def update(self, measurement, relative_now, step):
        if self.phase == 'failed':
            return
        good, self.last_forces = force_grasp(measurement, self.policy)
        self.force_streak = self.force_streak + 1 if good else 0
        near = (measurement['tcp_error']['position_m'] <= self.cfg['position_gate_m'] and
                measurement['tcp_error']['orientation_deg'] <= self.cfg['orientation_gate_deg'])
        if self.relative_anchor is None and measurement['opening_fraction'] > self.policy['max_opening_at_grasp']:
            self.failure_reason = 'door_moved_before_force_verified_grasp'
            self.transition('failed')
            return
        if self.phase == 'pregrasp':
            if near:
                self.transition('insert')
        elif self.phase == 'insert':
            if near:
                self.offset = max(0., self.offset - self.cfg['insertion_step_m'])
            if self.offset <= self.cfg['position_gate_m'] and near:
                self.offset = 0.
                self.transition('close')
        elif self.phase == 'close':
            if good:
                self.provisional = relative_now.copy()
                self.transition('verify')
            elif self.ticks >= self.cfg['close_timeout_steps']:
                self.reopen()
        elif self.phase == 'verify':
            if not good:
                self.provisional = None
                self.transition('close')
            elif self.force_streak >= self.policy['grasp_samples']:
                if self.relative_anchor is None:
                    self.relative_anchor = relative_now.copy()
                    self.first_verified_step = step
                self.transition('opening')
        elif self.phase == 'opening':
            slip, rotation = self.slip(relative_now)
            if not good or slip > self.cfg['slip_pause_m'] or rotation > self.cfg['rotation_pause_deg']:
                self.force_streak = 0
                self.recoveries += 1
                self.transition('recover_hold')
        elif self.phase == 'recover_hold':
            slip, rotation = self.slip(relative_now)
            if (good and self.force_streak >= self.policy['grasp_samples']
                    and slip <= self.cfg['slip_resume_m'] and rotation <= self.cfg['rotation_resume_deg']):
                self.transition('opening')
            elif not good and self.ticks >= self.cfg['close_timeout_steps']:
                self.reopen()
        elif self.phase == 'reopen' and self.ticks >= self.cfg['reopen_steps']:
            self.offset = .03
            self.provisional = None
            self.transition('pregrasp')
        self.ticks += 1

    def slip(self, relative):
        if self.relative_anchor is None:
            return 0., 0.
        translation = float(np.linalg.norm(relative[:3, 3] - self.relative_anchor[:3, 3]))
        rotation = float(np.degrees(Rotation.from_matrix(relative[:3, :3] @ self.relative_anchor[:3, :3].T).magnitude()))
        return translation, rotation

    @property
    def closed(self):
        return self.phase in ('close', 'verify', 'opening', 'recover_hold')

    @property
    def may_open(self):
        return self.phase == 'opening' and self.relative_anchor is not None

    def target(self, handle, expert_relative):
        relative = self.relative_anchor if self.relative_anchor is not None else self.provisional
        if relative is None:
            relative = expert_relative
        target = moving_goal(handle, relative, self.attempt)
        if self.phase in ('pregrasp', 'insert'):
            target[:3, 3] -= target[:3, 2] * self.offset
        return target

    def interrupted(self):
        if self.phase == 'opening':
            self.force_streak = 0
            self.recoveries += 1
            self.transition('recover_hold')


def guarded_physics_step(task, action, probe, arm, target, require_grip):
    """Cancel forward arm/base targets at the next 20ms control tick on loss."""
    import mujoco
    robot = task.env.robots[0]
    robot.update_control({k: np.asarray(v, dtype=float).copy() for k, v in action.items() if k in robot.controllers})
    observations, interrupted = [], False
    for control_step in range(task._n_ctrl_steps_per_policy):
        measurement = probe.measure(task, arm, target)
        good, forces = force_grasp(measurement)
        if require_grip and not good:
            hold = {k: v.copy() for k, v in robot.robot_view.get_noop_ctrl_dict().items() if k in robot.controllers}
            hold[arm + '_gripper'] = np.array([100.])
            robot.update_control(hold)
            interrupted = True
        robot.compute_control()
        mujoco.mj_step(task.env.mj_model, task.env.mj_datas[0], task._n_sim_steps_per_ctrl)
        after = probe.measure(task, arm, target)
        contact, forces = force_grasp(after)
        observations.append(dict(control_step=control_step + 1, force_grasp=contact, finger_forces_N=forces,
                                 opening_fraction=after['opening_fraction']))
    assert np.isfinite(task.env.mj_datas[0].qpos).all()
    return observations, interrupted


def run_guarded(task, arm, probe, expert_relative, servo_cfg, max_steps=500):
    from .runtime import clone_task, integration_state
    parent_state = integration_state(task.env.mj_model, task.env.mj_datas[0])
    active = clone_task(task)
    robot, model, data = active.env.robots[0], active.env.mj_model, active.env.mj_datas[0]
    view = robot.robot_view
    arm_group, gripper = arm + '_arm', arm + '_gripper'
    machine, recovery = FirmGraspControl(), BaseRecovery(servo_cfg)
    command = {k: v.copy() for k, v in view.get_noop_ctrl_dict().items() if k in robot.controllers}
    base_start = view.get_move_group('base').joint_pos.copy()
    opening_base = None
    limits = view.get_move_group(arm_group).joint_pos_limits
    joint = active.door_object.joint_ids[active.door_object.get_hinge_joint_index()]
    low, high = map(float, task.config.task_config.articulated_joint_range)
    threshold = float(task.config.task_config.door_openness_threshold)
    angle_goal = low + min(threshold + .05, 1.) * (high - low)
    trace, states = [], []
    keep_state(active, states, 0)
    for step in range(max_steps):
        handle = body_pose(data, probe.selected)
        target = machine.target(handle, expert_relative)
        before = probe.measure(active, arm, target)
        relative_now = np.linalg.inv(handle) @ view.get_move_group(gripper).leaf_frame_to_world
        machine.update(before, relative_now, step)
        if machine.phase == 'failed':
            break
        if machine.relative_anchor is not None and opening_base is None:
            opening_base = view.get_move_group('base').joint_pos.copy()
        target = machine.target(handle, expert_relative)
        advance = 0.
        if machine.may_open:
            advance = min(machine.cfg['opening_step_rad'] if min(machine.last_forces) >= machine.cfg['cautious_force_N']
                          else machine.cfg['cautious_step_rad'], max(angle_goal - float(data.qpos[model.jnt_qposadr[joint]]), 0.))
            desired = hinge_arc(handle, data.xanchor[joint], data.xaxis[joint], advance)
            target = machine.target(desired, expert_relative)
        command[gripper] = np.array([100. if machine.closed else -100.])
        delta = servo_delta(view.get_jacobian(gripper, [arm_group, 'base'])[:, :9],
                            view.get_move_group(gripper).leaf_frame_to_world, target, True, servo_cfg)
        command[arm_group] = np.clip(view.get_move_group(arm_group).joint_pos + delta[:7], limits[:, 0], limits[:, 1])
        reposition = recovery.update(before, 'approach' if machine.phase in ('pregrasp', 'insert') else machine.phase,
                                     step, base_start, view.get_move_group(gripper).leaf_frame_to_world, target)
        base = view.get_move_group('base').joint_pos.copy()
        if reposition is None:
            base[:2] += delta[7:9]
        else:
            gap = reposition[:2] - base[:2]
            base[:2] += gap * min(1., servo_cfg['base_step_m'] / max(np.linalg.norm(gap), 1e-12))
        origin = opening_base if opening_base is not None else base_start
        cap = machine.cfg['opening_base_cap_m'] if opening_base is not None else (.85 if recovery.events else .6)
        displacement = base[:2] - origin[:2]
        if np.linalg.norm(displacement) > cap:
            base[:2] = origin[:2] + displacement * cap / np.linalg.norm(displacement)
        base[2] = base_start[2]
        command['base'] = base
        substeps, interrupted = guarded_physics_step(active, command, probe, arm, target, machine.may_open)
        if interrupted:
            machine.interrupted()
        entry = probe.measure(active, arm, target)
        force, forces = force_grasp(entry)
        entry.update(step=step + 1, stage='guarded_control', phase=machine.phase,
                     force_grasp=force, finger_forces_N=forces, hinge_lookahead_rad=advance,
                     opening_command_interrupted=interrupted, control_substeps=substeps,
                     grasp_attempt=machine.attempt + 1)
        trace.append(entry)
        keep_state(active, states, step + 1)
        if retained_grasp_opening(trace, threshold)['retained_grasp_opening_success']:
            break
    result = retained_grasp_opening(trace, threshold)
    result.update(steps=len(trace), final_controller_phase=machine.phase, failure_reason=machine.failure_reason,
                  first_controller_verified_step=machine.first_verified_step, recoveries=machine.recoveries,
                  base_recovery_events=recovery.events, control_step_interruptions=sum(t['opening_command_interrupted'] for t in trace))
    assert np.array_equal(parent_state, integration_state(task.env.mj_model, task.env.mj_datas[0]))
    return trace, result, states
