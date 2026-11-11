"""Deployment API shared by offline pose evaluation and online grasp execution."""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .data import state_vector, point_vector, digest, TARGET_FRAME
from .model import PoseHead
from .vision import FrozenVision


class PosePredictor:
    def __init__(self,head,vision,normalization,device):
        self.head=head.eval();self.vision=vision
        self.normalization=normalization;self.device=torch.device(device)

    @classmethod
    def from_checkpoint(cls,path,device='cpu',encoder_path=None):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['target_frame']!=TARGET_FRAME or checkpoint['cache_metadata']['role']!='train':
            raise ValueError('Checkpoint pose-frame or training-role mismatch')
        encoder_path=Path(encoder_path or checkpoint['cache_metadata']['encoder_path'])
        if digest(encoder_path/'model.safetensors')!=checkpoint['cache_metadata']['encoder_sha256']:
            raise ValueError('Visual encoder differs from training')
        head=PoseHead(**checkpoint['model_configuration']).to(device)
        head.load_state_dict(checkpoint['model'])
        return cls(head,FrozenVision(encoder_path,device),checkpoint['normalization'],device)

    @torch.inference_mode()
    def predict(self,images,qpos,points,task_is_pull):
        """Use only current observations.

        images: {head_camera,wrist_camera_l,wrist_camera_r: uint8 HxWx3 RGB}.
        points: current-frame camera -> (normalized x,y,valid) OR native dict
        {points: Nx2, num_points: [N]}. NaN padding after N is ignored.
        Missing/occluded current points are invalid, not replaced by a future
        expert pose or a stale initial-image pixel in the current image.
        Returns xyz+wxyz in the current robot-base frame and predicted arm.
        """
        point=point_vector(points)
        state=(state_vector(qpos)-np.asarray(self.normalization['mean'],np.float32))/np.asarray(self.normalization['std'],np.float32)
        visual=self.vision([images],point[None])
        output=self.head(visual=visual,state=torch.tensor(state[None].clip(-10,10),device=self.device),
            points=torch.tensor(point[None],device=self.device),is_pull=torch.tensor([float(task_is_pull)],device=self.device))
        rotation=output['rotation'][0].cpu().numpy()
        if not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-3) or np.linalg.det(rotation)<.99:
            raise ValueError('Pose head produced an invalid rotation')
        quaternion=Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        pose=np.r_[output['position'][0].cpu().numpy(),quaternion]
        if not np.isfinite(pose).all():raise ValueError('Pose head produced a nonfinite pose')
        return dict(tcp_base_xyz_wxyz=pose.tolist(),arm=('left','right')[int(output['arm_logits'][0].argmax())],
            gripper_width_m=float(output['gripper_width'][0]),pose_frame=TARGET_FRAME,
            gripper_width_definition='sum of absolute measured finger displacements; proxy, not native actuator command',
            part_points_visible=int(point[:,2].sum()))
