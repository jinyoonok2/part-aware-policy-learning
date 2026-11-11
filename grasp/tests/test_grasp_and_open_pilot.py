"""Geometry and metric checks for the joint grasp/opening pilot."""
import unittest
import numpy as np
from grasp.opening import hinge_arc, joint_metrics, opening_success_ready


class OpeningChecks(unittest.TestCase):
    def test_hinge_arc_preserves_radius_and_grasp_transform(self):
        handle = np.eye(4); handle[:3, 3] = [2., 1., .9]
        pivot = np.array([1., 1., .9])
        relative = np.eye(4); relative[:3, 3] = [.04, .02, 0]
        axis = np.array([0., 0., 2.])
        rotated = hinge_arc(handle, pivot, axis, np.pi / 2)
        np.testing.assert_allclose(rotated[:3, 3], [1., 2., .9], atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(rotated) @ (rotated @ relative), relative, atol=1e-12)
        np.testing.assert_allclose(handle[:3, 3], [2., 1., .9])
        np.testing.assert_array_equal(axis, [0., 0., 2.])

    def test_open_door_without_contact_is_not_joint_success(self):
        trace = [dict(step=i + 1, active_arm_exact_both_fingers=False, opening_fraction=.9) for i in range(30)]
        result = joint_metrics(trace, .67)
        self.assertTrue(result['door_threshold_ever'])
        self.assertFalse(result['grasp_verified'])
        self.assertFalse(result['sustained_grasp_and_open'])

    def test_grasp_without_opening_is_not_joint_success(self):
        trace = [dict(step=i + 1, active_arm_exact_both_fingers=True, opening_fraction=.2) for i in range(30)]
        result = joint_metrics(trace, .67)
        self.assertTrue(result['grasp_verified'])
        self.assertFalse(result['sustained_grasp_and_open'])

    def test_joint_success_requires_verified_grasp_and_ten_joint_samples(self):
        trace = [dict(step=i + 1, active_arm_exact_both_fingers=True, opening_fraction=.8) for i in range(21)]
        self.assertFalse(joint_metrics(trace, .67)['sustained_grasp_and_open'])
        trace.append(dict(step=22, active_arm_exact_both_fingers=True, opening_fraction=.8))
        self.assertTrue(joint_metrics(trace, .67)['sustained_grasp_and_open'])

    def test_already_open_handoff_waits_for_reported_joint_criterion(self):
        # The grasp controller verifies contact after approximately 14 samples.
        # An already-open door still needs ten samples after metric verification.
        grasp = [dict(step=311 + i, active_arm_exact_both_fingers=True, opening_fraction=.8) for i in range(14)]
        opening = [dict(step=325, active_arm_exact_both_fingers=True, opening_fraction=.8)]
        self.assertFalse(opening_success_ready(grasp, opening, .67))
        opening = [dict(step=325 + i, active_arm_exact_both_fingers=True, opening_fraction=.8) for i in range(8)]
        self.assertTrue(opening_success_ready(grasp, opening, .67))


if __name__ == '__main__':
    unittest.main()
