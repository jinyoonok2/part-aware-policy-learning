"""Part-conditioned interaction-pose head over frozen visual representations."""
import torch
from torch import nn
import torch.nn.functional as F

from .data import STATE_DIM
from .geometry import rotation_6d_to_matrix, quaternion_to_matrix


class PoseHead(nn.Module):
    def __init__(self, width=256, layers=3):
        super().__init__()
        self.configuration = dict(width=width, layers=layers)
        self.visual_projection = nn.Linear(768, width)
        self.point_projection = nn.Linear(3, width)
        self.camera_embedding = nn.Parameter(torch.randn(3,width)*.02)
        self.kind_embedding = nn.Parameter(torch.randn(2,width)*.02)
        self.state_projection = nn.Sequential(nn.Linear(STATE_DIM,width), nn.GELU(), nn.Linear(width,width))
        self.task_projection = nn.Linear(1,width)
        self.query = nn.Parameter(torch.zeros(1,1,width))
        layer = nn.TransformerEncoderLayer(width, 8, width*4, dropout=.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Sequential(nn.Linear(width,width), nn.GELU(), nn.Linear(width,12))
        # Start from a proper identity rotation to avoid degenerate 6D vectors.
        with torch.no_grad():
            self.output[-1].bias[3:9] = torch.tensor([1.,0.,0.,0.,1.,0.])

    def forward(self, visual, state, points, is_pull, **unused):
        batch = len(visual)
        visual = self.visual_projection(visual.float())
        visual = visual + self.camera_embedding[None,:,None,:] + self.kind_embedding[None,None,:,:]
        visual = visual + self.point_projection(points.float())[:,:,None,:]
        state_token = self.state_projection(state.float())[:,None,:]
        task_token = self.task_projection(is_pull.float().reshape(batch,1))[:,None,:]
        tokens = torch.cat((self.query.expand(batch,-1,-1),visual.reshape(batch,6,-1),state_token,task_token),1)
        # Mask only unavailable local-part tokens, retaining global images.
        missing = torch.zeros((batch,9), device=visual.device, dtype=torch.bool)
        missing[:,[2,4,6]] = points[:,:,2] < .5
        hidden = self.norm(self.transformer(tokens, src_key_padding_mask=missing)[:,0])
        output = self.output(hidden).float()
        return dict(position=output[:,:3], rotation=rotation_6d_to_matrix(output[:,3:9]),
                    arm_logits=output[:,9:11], gripper_width=.15*torch.sigmoid(output[:,11]))


def pose_loss(output, batch):
    """Dimensionless scaled translation, chordal rotation, arm and finger-span losses."""
    target = batch['target'].float()
    position = F.smooth_l1_loss(output['position']/.1, target[:,:3]/.1)
    rotation = ((output['rotation']-quaternion_to_matrix(target[:,3:]))**2).sum((-1,-2)).mean()/8
    arm = F.cross_entropy(output['arm_logits'],batch['arm'].long())
    gripper = F.smooth_l1_loss(output['gripper_width']/.02,batch['gripper_width_m'].float()/.02)
    total = position + rotation + .25*arm + .1*gripper
    return total, dict(position=position, rotation=rotation, arm=arm, gripper=gripper)

