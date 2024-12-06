from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import timestep_embedding

import numpy as np
from einops import rearrange, reduce
from omegaconf import OmegaConf


class DiffTAD(nn.Module):
    def __init__(self,
                 img_size=512,
                 n_tokens=384,
                 num_frames=4,
                 total_frames=768,
                 timestep = 100,
                 layer_idxs = (2, 5, 8, 12),
                 ckpt="ldm/v1-5-pruned-emaonly.ckpt",
                 config_path="ldm/v1-inference.yaml",
                 context_ckpt=None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
                     dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
                 ],
                 **kwargs):
        super().__init__()

        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(config, f"{ckpt}")
        unet = self.model.model.diffusion_model
        # model = model.to(device)

        self.n_tokens = n_tokens
        self.num_frames = num_frames

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        timesteps = torch.tensor([timestep])
        timesteps = torch.cat([timesteps] * 2 * num_frames)
        t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to('cuda')
        self.emb = unet.time_embed(t_emb)

        # for param in unet.parameters():
        #     param.requires_grad = True

        modules = unet.input_blocks + [unet.middle_block]
        for idx, module in enumerate(modules):
            if idx >= min(layer_idxs):
                for param in module.parameters():
                    param.requires_grad = True

        context = torch.load(context_ckpt, weights_only=True) if context_ckpt is not None \
            else torch.randn(1, n_tokens, 768)
        self.context = nn.Parameter(context)

        self.layer_idxs = layer_idxs

        # del self.model.cond_stage_model
        del unet.output_blocks
        del unet.out

    def forward(self, x, return_attn_maps=False):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> b t c h w')

        attn_maps = []

        for i in range(b):
            curr_attn_maps = self._forward_one_window(x[i], return_attn_maps=return_attn_maps)
            attn_maps.append(curr_attn_maps)

        attn_maps = torch.stack(attn_maps)
        return attn_maps

    def _forward_one_window(self, x, return_attn_maps=False):
        x = x / 255.0
        x = 2 * x - 1

        x = self.model.get_first_stage_encoding(self.model.encode_first_stage(x))
        x = torch.cat([x] * 2)

        unet = self.model.model.diffusion_model

        emb = self.emb

        # unconditional_conditioning = self.context
        # cond = self.context
        # c_crossattn = [torch.cat([unconditional_conditioning, cond])]
        # context = torch.cat(c_crossattn, 1)

        unconditional_conditioning = self.model.get_learned_conditioning([""])
        context = torch.cat([unconditional_conditioning] * self.num_frames * 2)

        # context = torch.cat([self.context] * self.num_frames * 2)
        learned_context = torch.cat([self.context] * self.num_frames * 2)

        attn_maps = []

        h = x.type(unet.dtype)
        modules = unet.input_blocks + [unet.middle_block]
        for idx, module in enumerate(modules):
            if idx not in self.layer_idxs:
                h, attn = module(h, emb, context, return_attn_maps=torch.tensor(1.0))
            else:
                h, attn = module(h, emb, learned_context, return_attn_maps=torch.tensor(1.0))
                n_maps, n_patches, _ = attn.shape
                H, W = int(np.sqrt(n_patches)), int(np.sqrt(n_patches))
                attn = rearrange(attn, 't (h w) c -> t c h w', h=H, w=W)

                attn = attn[n_maps // 2:, :, :, :]

                n_heads = n_maps // (2 * self.num_frames)

                attn = rearrange(attn, '(t H) n h w -> t H n h w', H=n_heads)

                if not return_attn_maps:
                    attn = reduce(attn, 't H n h w -> t 1 n 1 1', 'mean')

                attn_maps.append(attn)

        if return_attn_maps:
            resized_attn_maps = []
            for attn_map in attn_maps:
                t, H, n, h, w = attn_map.shape
                resized_attn_map = rearrange(attn_map, 't H n h w -> (t H) n h w')
                resized_attn_map = F.interpolate(resized_attn_map, size=64, mode='bilinear', align_corners=True)
                resized_attn_map = rearrange(resized_attn_map, '(t H) n h w -> t H n h w', t=t)
                resized_attn_maps.append(resized_attn_map)
            attn_maps = resized_attn_maps
        attn_maps = torch.cat(attn_maps, dim=1)

        attn_maps = reduce(attn_maps, 't n c h w -> c t h w', 'mean')

        return attn_maps


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model
