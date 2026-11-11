"""Interface contracts for native MolmoBot and local grasp interventions."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from grasp.native_adapter import (AbsoluteJointTargets, GateConfig, LocalGraspGate,
    NativeDeltaLimits, absolute_to_native_action, account_local_step,
    hold_native_action, invalidate_action_buffer, validate_native_action)


def qpos():
    return dict(base=np.array([1., 2., .3]), left_arm=np.arange(7) * .1,
                right_arm=np.arange(7) * -.1, torso=np.array([0., .2, -.4, .2, 0., 0.]),
                left_gripper=np.array([-.03, .03]), right_gripper=np.array([-.04, .04]))


class BridgeContracts(unittest.TestCase):
    def test_absolute_arm_target_converts_once_and_preserves_policy(self):
        state = qpos()
        native = dict(left_arm=np.ones(7) * .005, right_arm=np.ones(7) * -.003,
                      torso=np.array([.25]), left_gripper=np.array([-100.]))
        original = copy.deepcopy(native)
        target = AbsoluteJointTargets(dict(left_arm=state['left_arm'] + .02))
        action = absolute_to_native_action(target, state, native)
        np.testing.assert_allclose(action['left_arm'], .02)
        np.testing.assert_array_equal(action['right_arm'], native['right_arm'])
        np.testing.assert_array_equal(action['torso'], [.25])
        for key in native:
            np.testing.assert_array_equal(native[key], original[key])
        with self.assertRaises(TypeError):
            absolute_to_native_action(action, state)
        np.testing.assert_array_equal(target.values['left_arm'], state['left_arm'] + .02)

    def test_relative_conversion_tracks_new_observation(self):
        state = qpos()
        target = AbsoluteJointTargets(dict(left_arm=state['left_arm'] + .02))
        state['left_arm'] += .01
        action = absolute_to_native_action(target, state)
        np.testing.assert_allclose(action['left_arm'], .01)

    def test_base_yaw_wraps_short_path(self):
        state = qpos(); state['base'][2] = np.pi - .01
        target = AbsoluteJointTargets(dict(base=np.array([1., 2., -np.pi + .01])))
        action = absolute_to_native_action(target, state)
        np.testing.assert_allclose(action['base'], [0., 0., .02], atol=1e-12)

    def test_optional_delta_bounds_do_not_clip_untouched_policy(self):
        state = qpos(); policy = dict(right_arm=np.ones(7) * .2)
        target = AbsoluteJointTargets(dict(base=state['base'] + [3., 4., 1.], left_arm=state['left_arm'] + 1.))
        action = absolute_to_native_action(target, state, policy, limits=NativeDeltaLimits())
        self.assertAlmostEqual(np.linalg.norm(action['base'][:2]), .015)
        self.assertLessEqual(abs(action['base'][2]), .03)
        self.assertLessEqual(np.max(abs(action['left_arm'])), .04)
        np.testing.assert_array_equal(action['right_arm'], policy['right_arm'])

    def test_torso_six_joint_hold_is_never_a_native_action(self):
        state = qpos()
        target = AbsoluteJointTargets(dict(torso=np.ones(6) * 8., left_gripper=np.array([100.])))
        action = absolute_to_native_action(target, state)
        np.testing.assert_array_equal(action['torso'], [.2])
        np.testing.assert_array_equal(action['left_gripper'], [100.])
        action = absolute_to_native_action(target, state, dict(torso=np.array([.3])), torso_mode='hold')
        np.testing.assert_array_equal(action['torso'], [.2])
        with self.assertRaises(ValueError):
            validate_native_action(dict(torso=np.ones(6)))
        with self.assertRaises(ValueError):
            AbsoluteJointTargets(dict(left_gripper=np.array([-.03, .03])))

    def test_hold_action_has_zero_relative_commands(self):
        action = hold_native_action(qpos(), dict(left_arm=np.ones(7)), close_arm='left')
        np.testing.assert_array_equal(action['left_arm'], np.zeros(7))
        np.testing.assert_array_equal(action['right_arm'], np.zeros(7))
        np.testing.assert_array_equal(action['base'], np.zeros(3))
        np.testing.assert_array_equal(action['torso'], [.2])
        np.testing.assert_array_equal(action['left_gripper'], [100.])
        self.assertEqual(action['right_gripper'].shape, (1,))

    def test_invalidation_preserves_all_conditioning_history_and_rng(self):
        image, points, history, generator = object(), object(), [object()], object()
        policy = SimpleNamespace(action_buffer=[{}, {}, {}], buffer_index=1, step_count=17,
            obs_history=history, _conditioning_image=image, _conditioning_points=points,
            _action_generator=generator)
        self.assertEqual(invalidate_action_buffer(policy), 2)
        self.assertEqual(policy.action_buffer, [])
        self.assertEqual(policy.buffer_index, 0)
        self.assertEqual(policy.step_count, 17)
        self.assertIs(policy.obs_history, history)
        self.assertIs(policy._conditioning_image, image)
        self.assertIs(policy._conditioning_points, points)
        self.assertIs(policy._action_generator, generator)

    def test_step_accounting_does_not_double_count_policy_query(self):
        policy = SimpleNamespace(obs_history=[], step_count=7)
        obs = {'qpos': qpos()}
        account_local_step(policy, [obs], model_was_queried=True)
        self.assertEqual(policy.step_count, 7)
        self.assertFalse(policy.obs_history)
        account_local_step(policy, [obs], model_was_queried=False)
        self.assertEqual(policy.step_count, 8)
        self.assertIs(policy.obs_history[0], obs)


class GateContracts(unittest.TestCase):
    def test_trigger_requires_closing_near_handle(self):
        gate = LocalGraspGate()
        action = dict(left_gripper=np.array([100.]))
        self.assertFalse(gate.decide(action, {'left': .5}, {}).use_local)
        self.assertFalse(gate.decide(dict(left_gripper=np.array([-100.])), {'left': .1}, {}).use_local)
        decision = gate.decide(action, {'left': .1}, {})
        self.assertTrue(decision.started)
        self.assertTrue(decision.use_local)
        self.assertEqual(decision.arm, 'left')

    def test_force_release_requires_consecutive_samples(self):
        gate = LocalGraspGate(GateConfig(verification_samples=3))
        gate.decide(dict(right_gripper=np.array([100.])), {'right': .1}, {})
        self.assertTrue(gate.decide(None, {}, {'right': True}).use_local)
        self.assertTrue(gate.decide(None, {}, {'right': False}).use_local)
        self.assertTrue(gate.decide(None, {}, {'right': True}).use_local)
        self.assertTrue(gate.decide(None, {}, {'right': True}).use_local)
        decision = gate.decide(None, {}, {'right': True})
        self.assertFalse(decision.use_local)
        self.assertTrue(decision.finished)
        self.assertEqual(decision.reason, 'force_verified')
        self.assertFalse(gate.active)

    def test_timeout_cooldown_and_total_budget_are_bounded(self):
        gate = LocalGraspGate(GateConfig(max_local_steps=2, max_events=1, cooldown_steps=1))
        action = dict(left_gripper=np.array([100.]))
        self.assertTrue(gate.decide(action, {'left': .1}, {}).use_local)
        self.assertTrue(gate.decide(None, {}, {}).use_local)
        decision = gate.decide(None, {}, {})
        self.assertTrue(decision.finished)
        self.assertEqual(decision.reason, 'intervention_timeout')
        self.assertEqual(gate.events[0]['local_steps'], 2)
        self.assertEqual(gate.decide(action, {'left': .1}, {}).reason, 'cooldown')
        self.assertEqual(gate.decide(action, {'left': .1}, {}).reason, 'intervention_budget_exhausted')

    def test_verified_grasp_does_not_trigger_unnecessary_intervention(self):
        gate = LocalGraspGate(GateConfig(verification_samples=2))
        gate.decide(None, {}, {'left': True})
        decision = gate.decide(dict(left_gripper=np.array([100.])), {'left': .05}, {'left': True})
        self.assertFalse(decision.use_local)


if __name__ == '__main__':
    unittest.main()
