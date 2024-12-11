import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal


DINO_MODELS = {
    "small": "dinov2_vits14_reg",
    "base": "dinov2_vitb14_reg",
    "large": "dinov2_vitl12_reg",
    "giant": "dinov2_vitg12_reg",
}


class DinoPatchDiscriminator(nn.Module):
    def __init__(self, size: Literal["small", "base", "large", "giant"]):
        super().__init__()
        self.size = size
        self._dino_model = torch.hub.load(
            "facebookresearch/dinov2", DINO_MODELS[size], pretrained=True
        )
        self.proj = nn.Linear(self._dino_model.embed_dim, 1)
        
        nn.init.zeros_(self.proj.weight)
        
    def trainable_parameters(self, head_only: bool = True):
        if head_only:
            return self.proj.parameters()
        return self.parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        if H % self._dino_model.patch_size != 0 or W % self._dino_model.patch_size != 0:
            new_h = H + self._dino_model.patch_size - H % self._dino_model.patch_size
            new_w = W + self._dino_model.patch_size - W % self._dino_model.patch_size
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)
        outputs = self._dino_model.forward_features(x)
        outputs = torch.cat(
            [outputs["x_norm_clstoken"].unsqueeze(1), outputs["x_norm_patchtokens"]], dim=1
        )
        outputs = self.proj(outputs)
        return outputs
