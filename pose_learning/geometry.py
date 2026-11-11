"""Differentiable rotations; all external poses use xyz + quaternion wxyz."""
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(value):
    """First two matrix columns, orthogonalized as a continuous rotation output."""
    a, b = value[..., :3], value[..., 3:]
    fallback_x = torch.zeros_like(a)
    fallback_x[...,0] = 1
    a = torch.where(a.norm(dim=-1,keepdim=True)<1e-6, fallback_x, a)
    x = F.normalize(a, dim=-1, eps=1e-6)
    orthogonal = b - (x * b).sum(-1, keepdim=True) * x
    # A collinear network output has no unique second axis. Choose the basis
    # axis least aligned to x, then orthogonalize it, rather than return a
    # non-rotation. The normal branch remains the standard 6D representation.
    basis = F.one_hot(x.abs().argmin(-1),3).to(x.dtype)
    fallback_y = basis - (x*basis).sum(-1,keepdim=True)*x
    orthogonal = torch.where(orthogonal.norm(dim=-1,keepdim=True)<1e-6,fallback_y,orthogonal)
    y = F.normalize(orthogonal, dim=-1, eps=1e-6)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


def quaternion_to_matrix(quaternion):
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotation_error_deg(predicted, target):
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))

