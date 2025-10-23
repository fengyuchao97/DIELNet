# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))   #196+1,384
        # print(self.pos_embed.shape,"++++",self.embed_dim)

    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        # print("---",x.shape)#torch.Size([16, 3, 192, 256])
        B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape)#torch.Size([16, 192, 384])#torch.Size([16, 256, 384])
        pe = self.pos_embed
        # print("++++",pe.shape)#torch.Size([1, 192, 384])

        x = x + pe
        #print(x.shape)torch.Size([16, 192, 384])
        x = self.pos_drop(x)
        #print(x.shape)torch.Size([16, 192, 384])

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):  #deit_small_patch16_224=register_model
    model = DeiT(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_small_patch16_224.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    

    # pe = model.pos_embed[:, 1:, :].detach()#torch.Size([1, 196, 384])->torch.Size([1, 384, 196])
    # pe = pe.transpose(-1, -2)
    # pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))#torch.Size([1, 384, 14, 14])
    # pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)#torch.Size([1, 384, 12, 16])
    # pe = pe.flatten(2)#torch.Size([1, 384, 192])
    # pe = pe.transpose(-1, -2)#torch.Size([1, 192, 384])
    
    pe = model.pos_embed[:, 1:, :].detach()#torch.Size([1, 197, 384]) torch.Size([1, 196, 384]) torch.Size([1, 196, 384])  #the problem of error
    # print(pe)
    pe = pe.transpose(-1, -2)#torch.Size([1, 384, 196])
    # print(pe.shape)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))#torch.Size([1, 384, 14, 14])
    # print("L",pe.shape[0], pe.shape[1],pe.shape[2],"L")
    pe = F.interpolate(pe, size=(16,16), mode='bilinear', align_corners=True)#torch.Size([1, 384, 12, 16])
    pe = pe.flatten(2)#torch.Size([1, 384, 192])
    pe = pe.transpose(-1, -2)#torch.Size([1, 192, 384])
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model