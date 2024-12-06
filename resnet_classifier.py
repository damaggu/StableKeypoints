from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import timestep_embedding

import numpy as np
from einops import rearrange, reduce
from omegaconf import OmegaConf

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io.video import read_video



class ResNetTAD(nn.Module):
    def __init__(self):
        super().__init__()

        # Step 1: Initialize model with the best available weights
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights).to('cuda')

        # Step 2: Initialize the inference transforms
        self.preprocess = weights.transforms()

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> b t c h w')

        latents = []

        for i in range(b):
            curr_attn_maps = self._forward_one_window(x[i])
            latents.append(curr_attn_maps)

        latents = torch.stack(latents)
        return latents

    def _forward_one_window(self, x):
        batch = self.preprocess(x)

        # Step 4: Use the model and print the predicted category
        latents = self._get_resnet_latents(batch).squeeze(0)

        latents = rearrange(latents, 't c -> c t 1 1')

        return latents

    def _get_resnet_latents(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x