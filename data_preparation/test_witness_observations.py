import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from data_preparation.witness_observations import calibration, current_handle_points, save
from pose_learning.data import observed_points


class WitnessObservationTests(unittest.TestCase):
    def test_calibration_uses_actual_render_size_and_forward_axis(self):
        result = calibration(np.eye(4), 90., (360, 640, 3))
        self.assertAlmostEqual(result['intrinsic_cv'][0, 0], 180.)
        cv = result['extrinsic_cv'] @ np.array([0., 0., -2., 1.])
        pixel = result['intrinsic_cv'] @ cv
        np.testing.assert_allclose(pixel[:2]/pixel[2], [320., 180.])

    def test_current_points_keep_visibility_and_ignore_padding(self):
        points = np.array([[.3,.7],[np.nan,np.nan]],dtype=np.float32)
        obs = dict(object_image_points=dict(door_handle=dict(head_camera=dict(points=points,num_points=[1]))))
        result = current_handle_points(obs, 'head_camera')
        self.assertEqual(result['num_points'][0], 1)
        self.assertTrue(np.isnan(result['points'][1]).all())
        obs['object_image_points']['door_handle']['head_camera']['num_points'] = [2]
        with self.assertRaises(ValueError):current_handle_points(obs, 'head_camera')

    def test_saved_points_work_with_the_existing_pose_input_reader(self):
        camera = 'head_camera'
        frame = dict(simulation_time=0.,qvel=dict(head=[0.,0.]),
            action_sensors={name:{} for name in ('joint_pos','joint_pos_rel','ee_pose','ee_twist')},
            points={camera:dict(points=np.array([[.3,.7],[np.nan,np.nan]],np.float32),num_points=np.array([1]))},
            camera_parameters={camera:calibration(np.eye(4),90.,(360,640,3))})
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'sample.h5'
            def encoded(values):return np.asarray([json.dumps(value) for value in values],dtype=h5py.string_dtype('utf-8'))
            with h5py.File(path,'w') as file:
                group=file.create_group('traj_0');group.create_dataset('success',data=[False])
                save(group,[frame],[camera],encoded)
                np.testing.assert_allclose(observed_points(group,0)[0],[.3,.7,1.])
                with self.assertRaises(ValueError):save(group,[],[camera],encoded)


if __name__=='__main__':unittest.main()
