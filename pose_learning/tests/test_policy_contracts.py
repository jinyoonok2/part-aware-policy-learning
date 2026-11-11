"""Contracts that would otherwise silently change online/offline conditioning."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_learning.evaluate_policy import load_cases, observed_points, predicted_world_target


class PolicyContracts(unittest.TestCase):
    def test_native_object_camera_points_ignore_padding_and_grippers(self):
        obs = {'object_image_points': {
            'left_gripper': {'head_camera': {'points': [[.99, .99]], 'num_points': [1]}},
            'door_handle': {'head_camera': {'points': [[.2, .4], [.4, .6], [np.nan, np.nan]],
                                           'num_points': np.array([2])}}}}
        points = observed_points(obs)
        np.testing.assert_allclose(points['head_camera'], [.3, .5, 1])
        self.assertEqual(points['wrist_camera_l'], (0., 0., False))

    def test_hidden_handle_never_uses_padded_or_other_object_points(self):
        obs = {'object_image_points': {
            'door_handle': {'head_camera': {'points': [[.8, .8]], 'num_points': [0]}},
            'door': {'head_camera': {'points': [[.3, .3]], 'num_points': [1]}}}}
        self.assertEqual(observed_points(obs)['head_camera'], (0., 0., False))

    def test_base_frame_prediction_rotates_and_translates_once(self):
        base = np.array([[0., -1., 0., 3.], [1., 0., 0., 5.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
        target = predicted_world_target({'arm': 'right', 'tcp_base_xyz_wxyz': [1, 0, 1, 1, 0, 0, 0]}, base)
        np.testing.assert_allclose(target[:3, 3], [3, 6, 1])
        np.testing.assert_allclose(target[:3, :3], base[:3, :3])

    def test_manifest_refuses_changed_episode_and_unmaterialized_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'episode.json'; episode.write_text('{}')
            path = root / 'manifest.json'
            path.write_text(json.dumps({'cases': [{'case_id': 'test', 'episode_path': 'episode.json',
                                                  'episode_sha256': hashlib.sha256(b'{}').hexdigest()}]}))
            self.assertEqual(load_cases(path)[0]['episode_path'], str(episode))
            episode.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_cases(path)
            path.write_text(json.dumps({'cases': [{'case_id': 'test', 'h5': 'source.h5'}]}))
            with self.assertRaisesRegex(ValueError, 'materialized'):
                load_cases(path)


if __name__ == '__main__':
    unittest.main()
