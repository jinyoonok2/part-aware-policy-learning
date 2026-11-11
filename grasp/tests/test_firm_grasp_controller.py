import unittest
import numpy as np
from grasp.firm import FirmGraspControl
from grasp.tests.test_retained_grasp_metrics import fixture


class FirmControllerChecks(unittest.TestCase):
    def measurement(self):
        m = fixture()[0]
        m['tcp_error'] = dict(position_m=.001, orientation_deg=.1)
        return m

    def verified(self):
        state = FirmGraspControl(); state.phase = 'close'
        m = self.measurement()
        for step in range(13):
            state.update(m, np.eye(4), step)
        self.assertTrue(state.may_open)
        return state, m

    def test_no_opening_before_force_verification(self):
        state = FirmGraspControl(); state.phase = 'close'
        m = self.measurement()
        for step in range(12):
            state.update(m, np.eye(4), step)
            self.assertFalse(state.may_open)
        state.update(m, np.eye(4), 12)
        self.assertTrue(state.may_open)

    def test_loss_and_slip_pause_opening_and_require_reverification(self):
        state, m = self.verified()
        m['finger_contacts'] = []
        state.update(m, np.eye(4), 14)
        self.assertFalse(state.may_open)
        self.assertEqual(state.phase, 'recover_hold')
        m = self.measurement()
        for step in range(15, 27):
            state.update(m, np.eye(4), step)
            self.assertFalse(state.may_open)
        state.update(m, np.eye(4), 27)
        self.assertTrue(state.may_open)
        slipped = np.eye(4); slipped[0, 3] = .01
        state.update(m, slipped, 28)
        self.assertFalse(state.may_open)

    def test_touch_or_empty_grip_does_not_unlock_opening(self):
        for field, value in [('finger_contacts', []), ('gripper_aperture_m', .0006)]:
            state = FirmGraspControl(); state.phase = 'close'
            m = self.measurement(); m[field] = value
            for step in range(13):
                state.update(m, np.eye(4), step)
                self.assertFalse(state.may_open)

    def test_push_open_before_grasp_is_failure(self):
        state = FirmGraspControl()
        m = self.measurement(); m['opening_fraction'] = .2
        state.update(m, np.eye(4), 0)
        self.assertEqual(state.phase, 'failed')
        self.assertFalse(state.may_open)

    def test_pregrasp_is_offset_away_from_handle(self):
        state = FirmGraspControl()
        target = state.target(np.eye(4), np.eye(4))
        np.testing.assert_allclose(target[:3, 3], [0, 0, -.06])
        self.assertFalse(state.closed)

    def test_control_tick_interrupt_stops_opening(self):
        state, _ = self.verified()
        state.interrupted()
        self.assertFalse(state.may_open)
        self.assertEqual(state.phase, 'recover_hold')


if __name__ == '__main__':
    unittest.main()
