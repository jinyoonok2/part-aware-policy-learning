"""Frozen local DINOv2 features with matched RGB/point transforms."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import CAMERAS


class FrozenVision(torch.nn.Module):
    def __init__(self, encoder_path, device='cpu'):
        super().__init__()
        from transformers import Dinov2Model
        self.encoder_path = str(Path(encoder_path).resolve())
        self.encoder = Dinov2Model.from_pretrained(self.encoder_path, local_files_only=True)
        if self.encoder.config.hidden_size != 768 or self.encoder.config.patch_size != 14:
            raise ValueError('Expected the pinned DINOv2-base 768-dimensional encoder')
        self.encoder.requires_grad_(False).eval().to(device)
        self.register_buffer('mean', torch.tensor([.485, .456, .406], device=device)[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229, .224, .225], device=device)[None,:,None,None])

    @torch.inference_mode()
    def forward(self, images, points):
        """images: list of camera dicts, points: Bx3x3 normalized xy + visibility.

        Stretch the complete RGB image to 224x224; no crop removes the part.
        The identical normalized xy therefore indexes the resized patch grid.
        Missing part tokens are zeroed; global camera features remain available.
        """
        pixels = []
        for item in images:
            for camera in CAMERAS:
                image = np.asarray(item[camera])
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError('Expected uint8 RGB observations')
                pixels.append(np.asarray(Image.fromarray(image).resize((224,224), Image.Resampling.BICUBIC)))
        tensor = torch.from_numpy(np.stack(pixels)).to(self.mean.device, dtype=torch.float32).permute(0,3,1,2)/255
        tokens = self.encoder(pixel_values=(tensor-self.mean)/self.std).last_hidden_state.float()
        batch = len(images)
        patches = tokens[:,1:].transpose(1,2).reshape(batch*3, 768, 16,16)
        points = torch.as_tensor(points, device=tokens.device, dtype=torch.float32).reshape(batch*3,3)
        grid = (points[:,:2]*2-1).reshape(batch*3,1,1,2)
        local = F.grid_sample(patches, grid, mode='bilinear', padding_mode='border', align_corners=False).reshape(batch*3,768)
        local = local * points[:,2:3]
        return torch.stack((tokens[:,0], local), dim=1).reshape(batch,3,2,768)

