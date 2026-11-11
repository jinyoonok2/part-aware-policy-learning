
# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from data_preparation.build_early_interaction_pose_view import conditioning_frames, conditioning_records, pose_matrix, select_event


class EarlyLabelChecks(unittest.TestCase):
    def test_earliest_grasp_replaces_later_target(self):
        events=[dict(start_frame=15,target_frame=19,arm='right'),dict(start_frame=3,target_frame=5,arm='left')]
        fractions=np.r_[np.zeros(10),np.ones(10)*.3]
        chosen,ledger=select_event(events,fractions)
        self.assertEqual(chosen['target_frame'],5)
        self.assertTrue(ledger[-1]['source_opening_fraction']>.1)

    def test_late_regrasp_after_prior_opening_is_excluded(self):
        event=dict(start_frame=6,target_frame=9,arm='left')
        fractions=[0,0,.4,.3,.1,.05,.03,.02,.02,.02]
        chosen,ledger=select_event([event],fractions)
        self.assertIsNone(chosen)
        self.assertTrue(ledger[0]['target_frame_early'])
        self.assertFalse(ledger[0]['no_prior_major_opening'])
        self.assertIsNotNone(select_event([event],fractions,require_no_prior=False)[0])

    def test_conditioning_uses_approach_for_selected_event(self):
        phase=np.array([0,1,1,1,1,3,3,3,4,4,1,1,1,1,3,3,3])
        frames,start=conditioning_frames(phase,1,5)
        self.assertEqual(start,1);self.assertEqual(frames,[1,2,3,4])
        self.assertTrue(all(i<5 for i in frames))

    def test_conditioning_transform_keeps_same_world_target(self):
        base=np.array([[0,0,0,1,0,0,0],[.4,.1,0,*Rotation.from_euler('z',.8).as_quat(scalar_first=True)]])
        world=np.eye(4);world[:3,3]=[.8,.2,1.]
        records=conditioning_records(base,world,[0,1],[0,.02])
        for row in records:
            np.testing.assert_allclose(pose_matrix(base[row['frame']])@pose_matrix(row['target_tcp_in_observation_base_xyz_wxyz']),world,atol=1e-12)


if __name__=='__main__':unittest.main()
