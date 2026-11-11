import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pose_learning.data import (state_vector, point_vector, observed_points,
    load_examples, CachedPoseDataset, TARGET_FRAME)
from pose_learning.geometry import quaternion_to_matrix, rotation_6d_to_matrix, rotation_error_deg
from pose_learning.model import PoseHead, pose_loss


class DataContractTests(unittest.TestCase):
    def test_boolean_arm_accuracy_summary(self):
        from pose_learning.evaluate import summarize
        self.assertEqual(summarize([True,False]),dict(mean=.5,median=.5,p90=.9))

    def test_no_world_coordinate_shortcut(self):
        qpos=dict(head=[0,0],torso=[0]*6,left_arm=[0]*7,right_arm=[0]*7,
                  left_gripper=[-.02,.02],right_gripper=[-.03,.03],base=[1,2,3])
        original=state_vector(qpos)
        qpos['base']=[1000,-500,-3]
        np.testing.assert_array_equal(state_vector(qpos),original)
        self.assertEqual(len(original),21)

    def test_native_points_ignore_nan_padding_and_flag_occlusion(self):
        point=point_vector(dict(head_camera=dict(points=np.array([[.2,.4],[.4,.8],[np.nan,np.nan]]),num_points=[2])))
        np.testing.assert_allclose(point[0],[.3,.6,1])
        np.testing.assert_array_equal(point[1:],np.zeros((2,3)))
        absent=point_vector(dict(head_camera=dict(points=np.full((10,2),np.nan),num_points=[0])))
        np.testing.assert_array_equal(absent,np.zeros((3,3)))

    def test_invalid_visible_points_rejected(self):
        with self.assertRaises(ValueError):point_vector({'head_camera':[1.2,.5,1]})
        with self.assertRaises(ValueError):point_vector({'head_camera':dict(points=[[np.nan,.2]],num_points=[1])})

    def test_h5_reads_current_frame_only(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/'observation.h5','w') as f:
                group=f.create_group('traj_0')
                point=group.create_group('obs/extra/object_image_points/door_handle/head_camera')
                point.create_dataset('num_points',data=[[1],[1]])
                point.create_dataset('points',data=[[[.2,.3]],[[.9,.8]]])
                np.testing.assert_allclose(observed_points(group,0)[0],[.2,.3,1])

    def test_training_refuses_test_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.jsonl'
            path.write_text(json.dumps(dict(source_split='val',project_role='test'))+'\n')
            with self.assertRaisesRegex(ValueError,'different source'):load_examples(path,'train')

    def test_training_refuses_test_cache_before_arrays_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'metadata.json').write_text(json.dumps(dict(role='test',target_frame=TARGET_FRAME)))
            with self.assertRaisesRegex(ValueError,'role or pose-frame'):CachedPoseDataset(path,'train',{})


class PoseHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_quaternion_sign_invariance(self):
        q=torch.tensor([[.5,.5,.5,.5]])
        torch.testing.assert_close(quaternion_to_matrix(q),quaternion_to_matrix(-q))

    def test_rotation_is_proper_and_correct_angle(self):
        rotations=rotation_6d_to_matrix(torch.tensor([[1.,0,0,0,1,0],[0.,1,0,-1,0,0]]))
        torch.testing.assert_close(rotations.transpose(-1,-2)@rotations,torch.eye(3).repeat(2,1,1))
        torch.testing.assert_close(torch.det(rotations),torch.ones(2))
        torch.testing.assert_close(rotation_error_deg(rotations[0:1],rotations[1:2]),torch.tensor([90.]))

    def test_degenerate_rotation_output_still_yields_a_proper_pose(self):
        rotations=rotation_6d_to_matrix(torch.tensor([[0.,0,0,0,0,0],[1.,1,1,2,2,2]]))
        torch.testing.assert_close(rotations.transpose(-1,-2)@rotations,torch.eye(3).repeat(2,1,1),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(torch.det(rotations),torch.ones(2))

    def batch(self):
        return dict(visual=torch.randn(4,3,2,768),state=torch.randn(4,21),points=torch.tensor([[[.5,.5,1]]*3]*4),
            is_pull=torch.zeros(4),target=torch.tensor([[.5,.2,.9,1.,0,0,0]]*4),arm=torch.zeros(4,dtype=torch.long),
            gripper_width_m=torch.full((4,),.04))

    def test_missing_local_tokens_do_not_affect_prediction(self):
        torch.manual_seed(12)
        model=PoseHead(width=64,layers=1).eval();batch=self.batch();batch['points'][:,:,2]=0
        first=model(**batch)
        batch['visual'][:,:,1]+=100
        second=model(**batch)
        for name in first:torch.testing.assert_close(first[name],second[name])

    def test_finite_gradients_and_learnable_pose_supervision(self):
        torch.manual_seed(3)
        model=PoseHead(width=64,layers=1);batch=self.batch()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
        model.eval()
        initial=float(pose_loss(model(**batch),batch)[0].detach())
        for _ in range(30):
            optimizer.zero_grad();loss,_=pose_loss(model(**batch),batch);loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        final=float(pose_loss(model(**batch),batch)[0].detach())
        self.assertLess(final,initial*.4)
        self.assertGreater(float(model.point_projection.weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
