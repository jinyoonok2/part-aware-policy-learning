"""Check point/image alignment using a deterministic patch-grid test double."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pose_learning.data import CAMERAS
from pose_learning.vision import FrozenVision


class GridEncoder(torch.nn.Module):
    def forward(self,pixel_values):
        batch=len(pixel_values)
        tokens=torch.zeros(batch,257,768)
        yy,xx=torch.meshgrid(torch.arange(16.),torch.arange(16.),indexing='ij')
        tokens[:,1:,0]=xx.flatten()
        tokens[:,1:,1]=yy.flatten()
        tokens[:,0,0]=pixel_values[:,0].mean((-1,-2))
        return SimpleNamespace(last_hidden_state=tokens)


class VisionContracts(unittest.TestCase):
    def test_whole_image_point_alignment_and_camera_order(self):
        torch.set_num_threads(1)
        vision=FrozenVision.__new__(FrozenVision)
        torch.nn.Module.__init__(vision)
        vision.encoder=GridEncoder()
        vision.register_buffer('mean',torch.zeros(1,3,1,1))
        vision.register_buffer('std',torch.ones(1,3,1,1))
        images={camera:np.full((9,17,3),intensity,np.uint8) for camera,intensity in zip(CAMERAS,(0,128,255))}
        points=np.array([[[.25,.75,1],[.5,.5,0],[1,0,1]]],np.float32)
        result=vision([images],points)
        self.assertEqual(tuple(result.shape),(1,3,2,768))
        torch.testing.assert_close(result[0,:,0,0],torch.tensor([0,128/255,1.]))
        torch.testing.assert_close(result[0,0,1,:2],torch.tensor([3.5,11.5]))
        torch.testing.assert_close(result[0,1,1],torch.zeros(768))
        torch.testing.assert_close(result[0,2,1,:2],torch.tensor([15.,0.]))


if __name__=='__main__':unittest.main()
