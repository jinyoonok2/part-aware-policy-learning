import copy
import unittest
import numpy as np
from grasp.metrics import FINAL_POLICY, POLICY, retained_grasp_opening


def fixture():
    trace = []
    for i in range(45):
        opening = 0. if i < 13 else min((i - 12) * .04, .72)
        trace.append(dict(step=i + 1, active_arm_exact_both_fingers=True, selected_handle='handle',
                          finger_contacts=[dict(finger=f, other_body='handle', active_constraint=True, normal_force_N=10.) for f in (1,2)],
                          gripper_ctrl=[0.], gripper_aperture_m=.04, opening_fraction=opening,
                          selected_handle_world=np.eye(4).tolist(), tcp_world=np.eye(4).tolist()))
    return trace


class RetentionChecks(unittest.TestCase):
    def test_retained_grasp_then_opening_passes(self):
        self.assertTrue(retained_grasp_opening(fixture(), .67)['retained_grasp_opening_success'])

    def test_push_only_or_late_grasp_fails(self):
        trace = fixture()
        for t in trace:
            t['opening_fraction'] = .9
        self.assertFalse(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])
        for t in trace:
            t['active_arm_exact_both_fingers'] = False
        self.assertFalse(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])

    def test_force_contact_and_nonempty_grip_required(self):
        for field, value in [('finger_contacts', []), ('gripper_aperture_m', .0006), ('gripper_ctrl', [-.05])]:
            trace = fixture()
            for t in trace:
                t[field] = copy.deepcopy(value)
            self.assertFalse(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])

    def test_loss_or_slip_during_main_opening_fails(self):
        trace = fixture()
        for t in trace[20:22]:
            t['active_arm_exact_both_fingers'] = False
        self.assertFalse(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])
        trace = fixture()
        for t in trace[20:]:
            t['tcp_world'][0][3] = .03
        self.assertFalse(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])


class FinalRetentionChecks(unittest.TestCase):
    def evaluate(self, trace):
        return retained_grasp_opening(trace, .67, policy=FINAL_POLICY)

    def test_final_policy_keeps_historical_policy_unchanged(self):
        self.assertNotIn('require_no_prior_major_opening', POLICY)
        self.assertEqual(FINAL_POLICY['metric_version'], 'firm_grasp_opening_v3_onset')
        self.assertTrue(self.evaluate(fixture())['retained_grasp_opening_success'])

    def test_open_then_reclose_before_grasp_is_rejected(self):
        trace = fixture()
        trace[0]['opening_fraction'] = .3
        trace[0]['active_arm_exact_both_fingers'] = False
        # This historical behavior is preserved for old pilot reports.
        self.assertTrue(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])
        result = self.evaluate(trace)
        self.assertFalse(result['retained_grasp_opening_success'])
        self.assertEqual(result['reason'], 'major_opening_before_force_grasp_onset')
        self.assertEqual(result['disqualifying_step'], 1)

    def test_rapid_continuous_grasp_and_opening_needs_no_stationary_wait(self):
        trace = fixture()
        for i, row in enumerate(trace):
            row['opening_fraction'] = min(i * .1, .72)
        self.assertFalse(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])
        result = self.evaluate(trace)
        self.assertTrue(result['retained_grasp_opening_success'])
        self.assertEqual(result['grasp_onset_step'], 1)
        self.assertEqual(result['grasp_verified_step'], 13)
        self.assertGreater(result['opening_at_verification'], .1)
        self.assertEqual(result['opening_at_grasp'], 0.)

    def test_slip_during_initial_verification_is_measured_from_onset(self):
        trace = fixture()
        for i, row in enumerate(trace):
            row['opening_fraction'] = min(i * .1, .72)
        trace[5]['tcp_world'][0][3] = .03
        result = self.evaluate(trace)
        self.assertFalse(result['retained_grasp_opening_success'])
        self.assertEqual(result['reason'], 'handle_relative_slip_limit_exceeded')
        self.assertIsNone(result['grasp_verified_step'])
        self.assertTrue(result['regrasp_blocked_after_major_opening'])

    def test_verification_requires_consecutive_contact_even_for_single_gap(self):
        trace = fixture()
        for i, row in enumerate(trace):
            row['opening_fraction'] = min(i * .1, .72)
        trace[5]['active_arm_exact_both_fingers'] = False
        result = self.evaluate(trace)
        self.assertFalse(result['retained_grasp_opening_success'])
        self.assertEqual(result['reason'], 'force_grasp_verification_interrupted')

    def test_failed_early_verification_can_retry_while_door_remains_closed(self):
        prefix = fixture()[:6]
        prefix[-1]['active_arm_exact_both_fingers'] = False
        trace = prefix + fixture()
        for i, row in enumerate(trace):
            row['step'] = i
        result = self.evaluate(trace)
        self.assertTrue(result['retained_grasp_opening_success'])
        self.assertEqual(result['grasp_onset_step'], 6)
        self.assertEqual(len(result['prior_failed_attempts']), 1)

    def test_regrasp_after_loss_and_prior_opening_cannot_reset_history(self):
        trace = fixture()
        for row in trace[16:18]:
            row['active_arm_exact_both_fingers'] = False
        # Return closed after the grasp loss and offer another complete grasp
        # and opening trajectory; final eligibility still cannot be recovered.
        retry = fixture()
        for i, row in enumerate(retry, start=len(trace[:18]) + 1):
            row['step'] = i
        trace = trace[:18] + retry
        self.assertTrue(retained_grasp_opening(trace, .67)['retained_grasp_opening_success'])
        result = self.evaluate(trace)
        self.assertFalse(result['retained_grasp_opening_success'])
        self.assertEqual(result['reason'], 'grasp_contact_or_force_lost')
        self.assertTrue(result['regrasp_blocked_after_major_opening'])

    def test_short_verification_and_short_success_hold_do_not_pass(self):
        trace = fixture()[:12]
        for row in trace[1:]:
            row['opening_fraction'] = .72
        self.assertFalse(self.evaluate(trace)['retained_grasp_opening_success'])
        trace = fixture()[:38]
        # Threshold is reached at sample30; nine samples above threshold are
        # insufficient for the ten-observation joint force/opening hold.
        self.assertFalse(self.evaluate(trace)['retained_grasp_opening_success'])

    def test_force_both_fingers_and_retention_tolerances(self):
        for transform in ('weak_one_finger', 'two_missing_samples', 'translation', 'rotation'):
            trace = fixture()
            if transform == 'weak_one_finger':
                for row in trace:
                    row['finger_contacts'][1]['normal_force_N'] = .9
            elif transform == 'two_missing_samples':
                for row in trace[20:22]:
                    row['active_arm_exact_both_fingers'] = False
            elif transform == 'translation':
                for row in trace[20:]:
                    row['tcp_world'][0][3] = .021
            else:
                angle = np.deg2rad(11)
                rotation = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
                for row in trace[20:]:
                    for i in range(3):
                        row['tcp_world'][i][:3] = rotation[i]
            with self.subTest(transform=transform):
                self.assertFalse(self.evaluate(trace)['retained_grasp_opening_success'])

    def test_one_missing_sample_can_pass_with_sufficient_contact_fraction(self):
        trace = fixture()
        trace[20]['active_arm_exact_both_fingers'] = False
        result = self.evaluate(trace)
        self.assertTrue(result['retained_grasp_opening_success'])
        self.assertEqual(result['max_missing_contact_samples'], 1)

    def test_frequent_isolated_contact_losses_fail_total_retention(self):
        trace = fixture()
        for i in (16, 20, 24):
            trace[i]['active_arm_exact_both_fingers'] = False
        self.assertFalse(self.evaluate(trace)['retained_grasp_opening_success'])


if __name__ == '__main__':
    unittest.main()
