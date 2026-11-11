"""Numerical/control contracts for the interaction-pose servo."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from grasp.servo import ApproachCloseVerify, BaseRecovery, PARAMETERS, bounded_base_goal, moving_goal, servo_delta


class ServoContracts(unittest.TestCase):
    def test_relative_goal_follows_handle_rotation_and_translation(self):
        source = np.eye(4)
        source[:3, 3] = [.8, -.3, 1.]
        source[:3, :3] = Rotation.from_euler('z', .7).as_matrix()
        tcp = source.copy()
        tcp[:3, 3] += [.02, .08, 0]
        relative = np.linalg.inv(source) @ tcp
        moved = source.copy()
        moved[:3, :3] = Rotation.from_euler('z', 1.2).as_matrix()
        moved[:3, 3] += [.2, -.1, 0]
        goal = moving_goal(moved, relative)
        np.testing.assert_allclose(np.linalg.inv(moved) @ goal, relative, atol=1e-12)
        self.assertGreater(np.linalg.norm(goal[:3, 3] - tcp[:3, 3]), .1)

    def test_bounded_servo_at_singularity(self):
        actual = np.eye(4)
        target = actual.copy()
        target[:3, 3] = [10., -5., 2.]
        for mobile in (False, True):
            jac = np.ones((6, 9 if mobile else 7))
            dq = servo_delta(jac, actual, target, mobile, PARAMETERS)
            self.assertTrue(np.isfinite(dq).all())
            self.assertLessEqual(np.max(np.abs(dq[:7])), PARAMETERS['arm_step_rad'])
            if mobile:
                self.assertLessEqual(np.max(np.abs(dq[7:])), PARAMETERS['base_step_m'])

    def test_close_retry_and_exhaustion_without_contact(self):
        state = ApproachCloseVerify(PARAMETERS)
        measurement = dict(active_arm_exact_both_fingers=False, tcp_error=dict(position_m=.003, orientation_deg=1.))
        self.assertTrue(state.update(measurement, 0))
        phases = set()
        for step in range(1, 120):
            state.update(measurement, step)
            phases.add(state.phase)
        self.assertIn('reopen', phases)
        self.assertEqual(state.phase, 'exhausted')
        self.assertIsNone(state.first_verified_step)

    def test_verification_requires_persistent_contact(self):
        state = ApproachCloseVerify(PARAMETERS)
        measurement = dict(active_arm_exact_both_fingers=True, tcp_error=dict(position_m=.003, orientation_deg=1.))
        for step in range(8):
            state.update(measurement, step)
        self.assertIsNone(state.first_verified_step)
        for step in range(8, 16):
            state.update(measurement, step)
        self.assertEqual(state.phase, 'hold')
        self.assertIsNotNone(state.first_verified_step)
        measurement['active_arm_exact_both_fingers'] = False
        state.update(measurement, 17)
        self.assertEqual(state.phase, 'close')

    def test_twenty_mm_gate_retains_orientation_guard(self):
        measurement = dict(active_arm_exact_both_fingers=False, tcp_error=dict(position_m=.016, orientation_deg=1.))
        self.assertFalse(ApproachCloseVerify(PARAMETERS).update(measurement, 0))
        cfg = dict(PARAMETERS, position_threshold_m=.020)
        self.assertTrue(ApproachCloseVerify(cfg).update(measurement, 0))
        measurement['tcp_error']['orientation_deg'] = 6.
        self.assertFalse(ApproachCloseVerify(cfg).update(measurement, 0))
        measurement['tcp_error'] = dict(position_m=.025, orientation_deg=1.)
        self.assertFalse(ApproachCloseVerify(cfg).update(measurement, 0))

    def test_base_plan_is_bounded_and_preserves_yaw(self):
        actual = np.eye(4)
        target = actual.copy()
        target[0, 3] = 2.
        start, current = np.array([0., 0., .5]), np.array([.59, 0., .5])
        cfg = dict(PARAMETERS, recovery_max_base_displacement_m=.85, recovery_max_move_m=.25)
        goal = bounded_base_goal(start, current, actual, target, cfg)
        self.assertGreater(goal[0], current[0])
        self.assertLessEqual(np.linalg.norm(goal[:2] - current[:2]), .25 + 1e-12)
        self.assertLessEqual(np.linalg.norm(goal[:2] - start[:2]), .85 + 1e-12)
        self.assertEqual(goal[2], start[2])
        near = actual.copy(); near[0, 3] = .02
        self.assertIsNone(bounded_base_goal(start, current, actual, near, cfg))

    def test_base_recovery_detects_stall_and_stops_during_grasp(self):
        cfg = dict(PARAMETERS, recovery_stall_steps=15)
        recovery = BaseRecovery(cfg)
        actual = np.eye(4); target = actual.copy(); target[0, 3] = .2
        start = np.zeros(3)
        m = dict(tcp_error=dict(position_m=.2), base_qpos=start)
        for step in range(14):
            self.assertIsNone(recovery.update(m, 'approach', step, start, actual, target))
        self.assertIsNotNone(recovery.update(m, 'approach', 14, start, actual, target))
        self.assertEqual(len(recovery.events), 1)
        self.assertEqual(recovery.events[0]['reason'], 'stalled_approach')
        self.assertIsNone(recovery.update(m, 'close', 15, start, actual, target))

    def test_base_recovery_does_not_interrupt_small_error_or_progress(self):
        recovery = BaseRecovery(PARAMETERS)
        actual = np.eye(4); target = actual.copy(); target[0, 3] = .2
        start = np.zeros(3)
        for step in range(30):
            m = dict(tcp_error=dict(position_m=.5 - step * .01), base_qpos=start)
            self.assertIsNone(recovery.update(m, 'approach', step, start, actual, target))
        for step in range(30, 60):
            m = dict(tcp_error=dict(position_m=.016), base_qpos=start)
            self.assertIsNone(recovery.update(m, 'approach', step, start, actual, target))
        self.assertFalse(recovery.events)


if __name__ == '__main__':
    unittest.main()
