import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from .GhostNetv2 import ghostnetv2
from .video_swin_transformer import SwinTransformer3D
from mmcv.cnn import build_norm_layer

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, chunted=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn([chunted, dim, shape]))

    def forward(self, x):
        B, _, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x

def shunted(x, chunk=4, dim=-1):
        B, C, H, W = x.shape
        if dim==(-1):
            x = x.reshape(B, C, H, W//chunk, chunk).permute(0,4,1,2,3).mean(dim)
        elif dim==(-2):
            x = x.reshape(B, C, chunk, H//chunk, W).permute(0,2,1,3,4).mean(dim)
        return x

def blocked(x, chunk=4):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//chunk, W//chunk, chunk, chunk).permute(0,4,5,1,2,3).mean(-1).mean(-2)
        return x

class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                 chunk_number=4,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.chunk_number = chunk_number

        self.to_q_1 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)

        self.proj_encode_column_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        
        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        q = torch.cat([q_1,q_2],dim=1)

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        k = torch.abs(k_1-k_2)

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2)
        
        # detail enhance
        qkv_1 = torch.cat([q, k, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q, k, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        # squeeze axial attention
        ## squeeze row
        # qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        # vrow_1 = v_1.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # vrow_2 = v_2.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        
        # B, chunt, C, H, W 
        qrow = self.pos_emb_rowq(shunted(q, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow = self.pos_emb_rowk(shunted(k, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        vrow_1 = shunted(v_1, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2 = shunted(v_2, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)


        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)

        xx_row_1 = torch.matmul(attn_row, vrow_1)  # B nH H C
        xx_row_1 = self.proj_encode_row_1(xx_row_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        xx_row_2 = torch.matmul(attn_row, vrow_2)
        xx_row_2 = self.proj_encode_row_2(xx_row_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        ## squeeze column
        # qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        # kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        # vcolumn_1 = v_1.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        # vcolumn_2 = v_2.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        qcolumn = self.pos_emb_columnq(shunted(q, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn = self.pos_emb_columnk(shunted(k, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        vcolumn_1 = shunted(v_1, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2 = shunted(v_2, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)

        xx_column_1 = torch.matmul(attn_column, vcolumn_1)  # B nH W C
        xx_column_1 = self.proj_encode_column_1(xx_column_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx_column_2 = torch.matmul(attn_column, vcolumn_2)  # B nH W C
        xx_column_2 = self.proj_encode_column_2(xx_column_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)


        xx = torch.abs(xx_row_1-xx_row_2).add(torch.abs(xx_column_1-xx_column_2)).reshape(B, self.dh, H, W)
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out1 = att * qkv_1
        out2 = att * qkv_2

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            # att = att.permute(0,2,1).view(B,C,H,W)
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)

            # print(qkv_1.shape)
            # print(label.shape)
            loss_res = self.loss_generator(qkv_1*(1-label), qkv_2*(1-label))
            return out1, out2, loss_att, loss_res
        else:
            return out1, out2
        
class Sea_Attention_cross(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                 chunk_number=4,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.chunk_number = chunk_number

        self.to_q_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)

        self.proj_encode_column_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        
        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        # q = torch.cat([q_1,q_2],dim=1)

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        k = torch.abs(k_1-k_2)

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2)
        
        # detail enhance
        qkv_1 = torch.cat([q_2, k_1, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q_1, k_2, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        # squeeze axial attention
        ## squeeze row
        # qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        # vrow_1 = v_1.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # vrow_2 = v_2.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        
        # B, chunt, C, H, W 
        qrow_1 = self.pos_emb_rowq(shunted(q_2, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        qrow_2 = self.pos_emb_rowq(shunted(q_1, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_1 = self.pos_emb_rowk(shunted(k_1, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        krow_2 = self.pos_emb_rowk(shunted(k_2, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        vrow_1 = shunted(v_1, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2 = shunted(v_2, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_1 = torch.matmul(qrow_1, krow_1) * self.scale
        attn_row_1 = attn_row_1.softmax(dim=-1)

        attn_row_2 = torch.matmul(qrow_2, krow_2) * self.scale
        attn_row_2 = attn_row_2.softmax(dim=-1)

        xx_row_1 = torch.matmul(attn_row_1, vrow_1)  # B nH H C
        xx_row_1 = self.proj_encode_row_1(xx_row_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        xx_row_2 = torch.matmul(attn_row_2, vrow_2)
        xx_row_2 = self.proj_encode_row_2(xx_row_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        ## squeeze column
        # qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        # kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        # vcolumn_1 = v_1.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        # vcolumn_2 = v_2.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        
        qcolumn_1 = self.pos_emb_columnq(shunted(q_2, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        qcolumn_2 = self.pos_emb_columnq(shunted(q_1, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_1 = self.pos_emb_columnk(shunted(k_1, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        kcolumn_2 = self.pos_emb_columnk(shunted(k_2, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        vcolumn_1 = shunted(v_1, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2 = shunted(v_2, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_1 = torch.matmul(qcolumn_1, kcolumn_1) * self.scale
        attn_column_1 = attn_column_1.softmax(dim=-1)

        attn_column_2 = torch.matmul(qcolumn_2, kcolumn_2) * self.scale
        attn_column_2 = attn_column_2.softmax(dim=-1)

        xx_column_1 = torch.matmul(attn_column_1, vcolumn_1)  # B nH W C
        xx_column_1 = self.proj_encode_column_1(xx_column_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx_column_2 = torch.matmul(attn_column_2, vcolumn_2)  # B nH W C
        xx_column_2 = self.proj_encode_column_2(xx_column_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx = torch.abs(xx_row_1-xx_row_2).add(torch.abs(xx_column_1-xx_column_2)).reshape(B, self.dh, H, W)
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out1 = att * qkv_1
        out2 = att * qkv_2

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            # att = att.permute(0,2,1).view(B,C,H,W)
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)
            return out1, out2, loss_att
        else:
            return out1, out2

class Sea_Attention_woksub(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                 chunk_number=4,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.chunk_number = chunk_number

        self.to_q_1 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)

        self.proj_encode_column_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        
        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        q = torch.cat([q_1,q_2],dim=1)

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        # k = torch.abs(k_1-k_2)

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2)
        
        # detail enhance
        # qkv_1 = torch.cat([q, k, v_1], dim=1)
        # qkv_1 = self.act(self.dwconv_1(qkv_1))
        # qkv_1 = self.pwconv_1(qkv_1)

        # qkv_2 = torch.cat([q, k, v_2], dim=1)
        # qkv_2 = self.act(self.dwconv_2(qkv_2))
        # qkv_2 = self.pwconv_2(qkv_2)

        qkv_1 = torch.cat([q, k_1, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q, k_2, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        # squeeze axial attention
        ## squeeze row
        # qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        # vrow_1 = v_1.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # vrow_2 = v_2.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        
        # B, chunt, C, H, W 
        qrow = self.pos_emb_rowq(shunted(q, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        # krow = self.pos_emb_rowk(shunted(k, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        krow_1 = self.pos_emb_rowk(shunted(k_1, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        krow_2 = self.pos_emb_rowk(shunted(k_2, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        vrow_1 = shunted(v_1, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2 = shunted(v_2, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_1 = torch.matmul(qrow, krow_1) * self.scale
        attn_row_1 = attn_row_1.softmax(dim=-1)

        attn_row_2 = torch.matmul(qrow, krow_2) * self.scale
        attn_row_2 = attn_row_2.softmax(dim=-1)

        xx_row_1 = torch.matmul(attn_row_1, vrow_1)  # B nH H C
        xx_row_1 = self.proj_encode_row_1(xx_row_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        xx_row_2 = torch.matmul(attn_row_2, vrow_2)
        xx_row_2 = self.proj_encode_row_2(xx_row_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        ## squeeze column
        # qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        # kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        # vcolumn_1 = v_1.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        # vcolumn_2 = v_2.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        qcolumn = self.pos_emb_columnq(shunted(q, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        # kcolumn = self.pos_emb_columnk(shunted(k, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        kcolumn_1 = self.pos_emb_columnk(shunted(k_1, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        kcolumn_2 = self.pos_emb_columnk(shunted(k_2, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        vcolumn_1 = shunted(v_1, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2 = shunted(v_2, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_1 = torch.matmul(qcolumn, kcolumn_1) * self.scale
        attn_column_1 = attn_column_1.softmax(dim=-1)

        attn_column_2 = torch.matmul(qcolumn, kcolumn_2) * self.scale
        attn_column_2 = attn_column_2.softmax(dim=-1)

        xx_column_1 = torch.matmul(attn_column_1, vcolumn_1)  # B nH W C
        xx_column_1 = self.proj_encode_column_1(xx_column_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx_column_2 = torch.matmul(attn_column_2, vcolumn_2)  # B nH W C
        xx_column_2 = self.proj_encode_column_2(xx_column_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)


        xx = torch.abs(xx_row_1-xx_row_2).add(torch.abs(xx_column_1-xx_column_2)).reshape(B, self.dh, H, W)
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out1 = att * qkv_1
        out2 = att * qkv_2

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            # att = att.permute(0,2,1).view(B,C,H,W)
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)
            return out1, out2, loss_att
        else:
            return out1, out2
    
def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d

def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU

def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)

def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
class DWConv_T(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_T, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC
        return x

class BasicConv(nn.Module):
    def __init__(
        self, in_ch, out_ch, 
        kernel_size, pad_mode='Zero', 
        bias='auto', norm=False, act=False, 
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize()+'Pad2d')(kernel_size//2))
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=0,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output

class CPAMEnc(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        self.pool5 = nn.AdaptiveAvgPool2d(9)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        
        # feat = self.conv1(x).view(b,c,-1)
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        feat5 = self.conv5(self.pool5(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4, feat5), 2)
        # return feat

class CPAMDec(nn.Module):
    def __init__(self,in_channels):
        super(CPAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) 
        self.conv_key = nn.Linear(in_channels, in_channels//2) 
        self.conv_value = nn.Linear(in_channels, in_channels) 
    def forward(self, x,y):
        m_batchsize,C,width ,height = x.size()
        m_batchsize,K,M = y.size()

        proj_query  = self.conv_query(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.conv_key(y).view(m_batchsize,K,-1).permute(0,2,1)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)

        proj_value = self.conv_value(y).permute(0,2,1) 
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = self.scale*out + x
        return out

class CCAMDec(nn.Module):
    def __init__(self):
        super(CCAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x,y):
        m_batchsize,C,width ,height = x.size()
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape 
        proj_key  = y_reshape.permute(0,2,1) 
        energy =  torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) 
        
        out = torch.bmm(attention,proj_value) 
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out

class DRAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DRAtt, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = CPAMEnc(out_channels, norm_layer) # en_s
        self.cpam_dec = CPAMDec(out_channels) # de_s

        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) 
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) 
        self.ccam_dec = CCAMDec()
        
    def forward(self, x):
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        cpam_b = self.conv_cpam_b(ccam_feat)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)
        return cpam_feat

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))
    
class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes=1):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch, out_ch) # DRAtt
        self.upsample = nn.Sequential(nn.Conv2d(out_ch, out_ch*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out = Conv1x1(out_ch//2, num_classes)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x2 = F.interpolate(x2, size=x1.shape[2:])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        out = self.conv_fuse(x)
        output = self.conv_out(self.upsample(out))
        return out, output

class CPAMDec_Mix_large(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix_large,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.loss_generator = nn.L1Loss()
        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

    def forward(self,x1,y1,x2,y2,label=None):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_query2  = self.conv_query2(x2).view(m_batchsize,-1,width*height).permute(0,2,1)
        # proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)

        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query1,proj_key1)
        energy2 =  torch.bmm(proj_query2,proj_key2)

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 
        
        if label is not None:
            label = F.interpolate(label, size=(width,height))
            att = attention.permute(0,2,1).view(m_batchsize,K,width,height)
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)
            return out1, out2, loss_att
        else:
            return out1, out2

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x.permute(0, 2, 3, 1) #NHWC
        return x
    
class ContrastiveAtt_large(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt_large, self).__init__()

        inter_channels = in_channels

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix_large(inter_channels) # de_s
        
    def forward(self, x, y, label=None):
        cpam_f_x = self.cpam_enc_x(x).permute(0,2,1)
        cpam_f_y = self.cpam_enc_y(y).permute(0,2,1)

        if label is not None:
            cpam_feat1, cpam_feat2, loss_att = self.cpam_dec_mix(x,cpam_f_x,y,cpam_f_y,label) 
            return cpam_feat1, cpam_feat2, loss_att
        else: 
            cpam_feat1, cpam_feat2 = self.cpam_dec_mix(x,cpam_f_x,y,cpam_f_y) 
            return cpam_feat1, cpam_feat2
            
class ContrastiveAtt_Block(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, chunk_number=4, mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
        super().__init__()     # drop_path=0., mlp_ratio=4, mlp_dwconv=False,

        
        dim = in_channels #// 2
        # self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
        #                            norm_layer(dim),
        #                            nn.GELU()) # conv5_s
        # self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
        #                            norm_layer(dim),
        #                            nn.GELU()) # conv5_s
        
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing

        self.attn = Sea_Attention(dim, chunk_number=chunk_number) # ContrastiveAtt_large(dim)  #Sea_Attention

        self.pre_norm = pre_norm
        # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
        #                          DWConv_T(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
        #                          nn.GELU(),
        #                          nn.Linear(int(mlp_ratio * dim), dim)
        #                          )
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, t1, t2, labels=None):
        # t1 = self.conv_cpam_b_x(t1)
        # t2 = self.conv_cpam_b_y(t2)
        # conv pos embedding  3×3卷积，一个残差连接
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        # attention & mlp
        if self.pre_norm:
            if labels is not None:
                x1, x2, loss_att, loss_res = self.attn(t1, t2, labels)
            else:
                x1, x2 = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)
            
            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)
        if labels is not None:
            return t1, t2, loss_att, loss_res
        else:
            return t1, t2

class Local_interaction(nn.Module):
    def __init__(self, in_channels, num_tokens=1, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        # head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = nn.Parameter(torch.zeros(1))
        self.q_ratio = 1
        self.k_ratio = 1

        inter_channels = in_channels

        self.init_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.GELU()) # conv5_s
        self.init_conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.GELU()) # conv5_
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.conv_query1 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value1 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2

        self.conv_query2 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value2 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2

    def forward_interaction_local(self, q1, k1, v1, q2, k2, v2, H, W):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, num_heads, N, C = v1.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q1 = q1.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q1 = q1.view(-1, num_heads, ws*ws, C//self.q_ratio)
        q2 = q2.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q2 = q2.view(-1, num_heads, ws*ws, C//self.q_ratio)

        k1 = k1.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k1 = k1.view(-1, num_heads, ws*ws, C//self.k_ratio)
        k2 = k2.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k2 = k2.view(-1, num_heads, ws*ws, C//self.k_ratio)

        v1 = v1.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v1 = v1.view(-1, num_heads, ws*ws, v1.shape[-1])
        v2 = v2.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v2 = v2.view(-1, num_heads, ws*ws, v2.shape[-1])

        # q_cat = torch.cat([q1,q2],3)

        attn1 = (q1 @ k1.transpose(-2, -1)) #* self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) #* self.scale

        # pos_bias = self._get_relative_positional_bias()
        attn = (torch.abs(attn1 - attn2)).softmax(dim=-1) # + pos_bias

        attn = self.attn_drop(attn)
        x1 = (attn @ v1).transpose(1, 2).reshape(v1.shape[0], ws*ws, -1)
        x2 = (attn @ v2).transpose(1, 2).reshape(v2.shape[0], ws*ws, -1)

        # reverse
        x1 = window_reverse(x1, (H, W), (ws, ws)).permute(0, 2, 1).reshape(B, -1, H, W)
        x2 = window_reverse(x2, (H, W), (ws, ws)).permute(0, 2, 1).reshape(B, -1, H, W)
        return x1, x2
    
    def get_qkv(self, x1, x2):
        B, C, H ,W = x1.size()
        q1  = self.conv_query1(x1).view(B, self.num_heads, C // (self.num_heads*self.q_ratio), -1).permute(0,1,3,2)
        k1 =  self.conv_key1(x1).view(B, self.num_heads, C // (self.num_heads*self.k_ratio), -1).permute(0,1,3,2)
        v1 = self.conv_value1(x1).view(B, self.num_heads, C // self.num_heads, -1).permute(0,1,3,2)

        q2  = self.conv_query2(x2).view(B, self.num_heads, C // (self.num_heads*self.q_ratio), -1).permute(0,1,3,2)
        k2 =  self.conv_key2(x2).view(B, self.num_heads, C // (self.num_heads*self.k_ratio), -1).permute(0,1,3,2)
        v2 = self.conv_value2(x2).view(B, self.num_heads, C // self.num_heads, -1).permute(0,1,3,2)
        return q1,k1,v1,q2,k2,v2
    
    def forward(self, x1, x2):
        x1 = self.init_conv1(x1)
        x2 = self.init_conv2(x2)

        B, C, H, W = x1.shape
        q1,k1,v1, q2,k2,v2 = self.get_qkv(x1,x2)
        x1_interact, x2_interact = self.forward_interaction_local(q1, k1, v1, q2, k2, v2, H, W)

        x1_interact = x1_interact + self.scale*x1 
        # # add-self.scale*x1_interact + x1: TDANet-0523-GZ： 84.67
        # # Without: TDANet-0525-GZ：0.85622
        # # x1_interact + self.scale*x1: TDANet-0524-GZ：85.115
        x2_interact = x2_interact + self.scale*x2  
        return x1_interact, x2_interact

class Local_LSKAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0_img1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_img1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_img1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2_img1 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze_img1 = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_img1 = nn.Conv2d(dim//2, dim, 1)

        self.conv0_img2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_img2 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_img2 = nn.Conv2d(dim, dim//2, 1)
        self.conv2_img2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze_img2 = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_img2 = nn.Conv2d(dim//2, dim, 1)

        self.loss_generator = nn.L1Loss()
        # self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, labels = None):   
        attn1_img1 = self.conv0_img1(x1)
        attn2_img1 = self.conv_spatial_img1(attn1_img1)
        attn1_img1 = self.conv1_img1(attn1_img1)
        attn2_img1 = self.conv2_img1(attn2_img1)
        attn_img1 = torch.cat([attn1_img1, attn2_img1], dim=1)
        avg_attn_img1 = torch.mean(attn_img1, dim=1, keepdim=True)
        max_attn_img1, _ = torch.max(attn_img1, dim=1, keepdim=True)
        agg_img1 = torch.cat([avg_attn_img1, max_attn_img1], dim=1)
        sig_img1 = self.conv_squeeze_img1(agg_img1).sigmoid()

        attn1_img2 = self.conv0_img2(x2)
        attn2_img2 = self.conv_spatial_img2(attn1_img2)
        attn1_img2 = self.conv1_img2(attn1_img2)
        attn2_img2 = self.conv2_img2(attn2_img2)
        attn_img2 = torch.cat([attn1_img2, attn2_img2], dim=1)
        avg_attn_img2 = torch.mean(attn_img2, dim=1, keepdim=True)
        max_attn_img2, _ = torch.max(attn_img2, dim=1, keepdim=True)
        agg_img2 = torch.cat([avg_attn_img2, max_attn_img2], dim=1)
        sig_img2 = self.conv_squeeze_img2(agg_img2).sigmoid()

        # sig_img = torch.abs(sig_img1-sig_img2)
        attn_img1 = attn1_img1 * sig_img1[:,0,:,:].unsqueeze(1) + attn2_img1 * sig_img1[:,1,:,:].unsqueeze(1)
        attn_img1 = self.conv_img1(attn_img1)

        attn_img2 = attn1_img2 * sig_img2[:,0,:,:].unsqueeze(1) + attn2_img2 * sig_img2[:,1,:,:].unsqueeze(1)
        attn_img2 = self.conv_img2(attn_img2)

        att = torch.abs(attn_img1-attn_img2)
        x1 = x1 * att #+ self.scale*x1 # (att + self.scale*attn_img1) 
        x2 = x2 * att #+ self.scale*x2 # (att + self.scale*attn_img2) #att # 

        if labels is not None:
            m_batchsize, C, width, height = x1.size()
            label = F.interpolate(labels, size=(width,height))
            loss_att = self.loss_generator(torch.mean(att,dim=1).sigmoid(),label)
            return x1, x2, loss_att
        else:
            return x1, x2

class Local_Block(nn.Module):
    def __init__(self, in_channels, window_size=8, drop_path=0.1, chunk_number=4, mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
        super().__init__()     # drop_path=0., mlp_ratio=4, mlp_dwconv=False,

        dim = in_channels #//2
        # self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
        #                            norm_layer(dim),
        #                            nn.GELU()) # conv5_s
        # self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
        #                            norm_layer(dim),
        #                            nn.GELU()) # conv5_s
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing

        # self.attn = Local_interaction(dim, window_size)
        self.attn = Sea_Attention(dim, chunk_number=chunk_number)# Local_LSKAtt(dim)

        self.pre_norm = pre_norm
        # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
        #                          DWConv_T(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
        #                          nn.GELU(),
        #                          nn.Linear(int(mlp_ratio * dim), dim)
        #                          )
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, t1, t2, labels = None):
        # t1 = self.conv_cpam_b_x(t1)
        # t2 = self.conv_cpam_b_y(t2)
        # conv pos embedding  3×3卷积，一个残差连接
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        # attention & mlp
        loss_att = 0
        if self.pre_norm:
            # x1, x2 = self.attn(t1, t2)
            if labels is not None:
                x1, x2, loss_att, loss_res = self.attn(t1, t2, labels)
            else:
                x1, x2 = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)

            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)

        if labels is not None:
            return t1, t2, loss_att, loss_res
        else:
            return t1, t2

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)
    
class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(x1 + x2 + x3 - y1 - y2 - y3) #torch.abs()
        return out

class Fusion(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(Fusion, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y
           
class Align(nn.Module):
    def __init__(self, input_dim, dim, key_dim=16, num_heads=8,
                 chunk_number=4,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.chunk_number = chunk_number
        
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(input_dim, dim, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU()) 
        self.ccam_enc = nn.Sequential(nn.Conv2d(dim, dim//16, 1, bias=False),
                                   nn.BatchNorm2d(dim//16),
                                   nn.ReLU()) 
        self.ccam_dec = CCAMDec()
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        
        self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
        
    def forward(self, x, label=None):
        # x = self.conv_ccam_b(x)
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        x = self.ccam_dec(ccam_b,ccam_f)
        
        B, C, H, W = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(shunted(q, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow = self.pos_emb_rowk(shunted(k, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        vrow = shunted(v, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)

        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        ## squeeze column
        qcolumn = self.pos_emb_columnq(shunted(q, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn = self.pos_emb_columnk(shunted(k, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        vcolumn = shunted(v, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)

        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx = xx_row.add(xx_column).reshape(B, self.dh, H, W)
        xx = v.add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out = att * qkv
        return out

class MLLA_Align(nn.Module):
    def __init__(self, input_dim, dim, chunk_number=16, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(input_dim)
        self.in_proj1 = nn.Linear(input_dim, input_dim)
        self.act_proj1 = nn.Linear(input_dim, dim)

        self.dwc1 = nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim)

        self.act = nn.SiLU()
        self.attn = Align(input_dim, dim, chunk_number=chunk_number) 
        self.out_proj1 = nn.Linear(dim, dim)

        self.conv2 = nn.Conv2d(input_dim, dim, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H*W
        C2 = self.dim

        x1 = x.permute(0, 2, 3, 1).reshape(B, L, C) # .permute(0, 2, 1)
        x1 = self.norm1(x1)
        act_res1 = self.act(self.act_proj1(x1))
        x1 = self.in_proj1(x1).view(B, H, W, C)
        x1 = self.act(self.dwc1(x1.permute(0, 3, 1, 2)))

        # Linear Attention
        x_att = self.attn(x1)
        # x1_1, x1_2 = torch.chunk(x1, chunks=2, dim=1)

        # x_out1 = self.local_single_unit(x1_1)
        # x_out2, _ = self.global_single_unit(x1_2)

        # x_out= torch.cat([x_out1, x_out2], dim=1)


        x1 = self.act(self.conv2(x)) + self.out_proj1(x_att.permute(0, 2, 3, 1).reshape(B, L, C2) * act_res1).view(B, H, W, C2).permute(0, 3, 1, 2)

        return x1
    
class Interaction_attention_various_head(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                #  chunk_number=4,
                 chunk_numbers=[4,8],
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        # self.chunk_number = chunk_number
        self.chunk_numbers = chunk_numbers

        self.to_q_1 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row_1_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])
        self.pos_emb_rowk_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])

        self.proj_encode_column_1_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])
        self.pos_emb_columnk_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])

        self.proj_encode_row_1_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        # part 2
        self.pos_emb_rowq_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])
        self.pos_emb_rowk_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])

        self.proj_encode_column_1_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])
        self.pos_emb_columnk_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])
        
        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        q = torch.cat([q_1,q_2],dim=1) # B, nh_kd, H, W

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        k =  torch.abs(k_1-k_2) # B, nh_kd, H, W

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2) # B, self.dh, H, W
        
        # detail enhance
        qkv_1 = torch.cat([q, k, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q, k, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        q_part1 = q[:,:(self.nh_kd//2),:,:]
        k_part1 = k[:,:(self.nh_kd//2),:,:]
        v1_part1 = v_1[:,:(self.dh//2),:,:]
        v2_part1 = v_2[:,:(self.dh//2),:,:]
        # squeeze axial attention
        ## squeeze row
        # B, chunt, C, H, W 
        qrow_part1 = self.pos_emb_rowq_part1(shunted(q_part1, chunk=self.chunk_numbers[0], dim=-1)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part1 = self.pos_emb_rowk_part1(shunted(k_part1, chunk=self.chunk_numbers[0], dim=-1)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H)
        vrow_1_part1 = shunted(v1_part1, chunk=self.chunk_numbers[0], dim=-1).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part1 = shunted(v2_part1, chunk=self.chunk_numbers[0], dim=-1).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part1 = torch.matmul(qrow_part1, krow_part1) * self.scale
        attn_row_part1 = attn_row_part1.softmax(dim=-1)
        xx_row_1_part1 = torch.matmul(attn_row_part1, vrow_1_part1)  # B nH H C
        xx_row_1_part1 = self.proj_encode_row_1_part1(xx_row_1_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], H//self.chunk_numbers[0])).unsqueeze(-1)
        xx_row_2_part1 = torch.matmul(attn_row_part1, vrow_2_part1)
        xx_row_2_part1 = self.proj_encode_row_2_part1(xx_row_2_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], H//self.chunk_numbers[0])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part1 = self.pos_emb_columnq_part1(shunted(q_part1, chunk=self.chunk_numbers[0], dim=-2)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part1 = self.pos_emb_columnk_part1(shunted(k_part1, chunk=self.chunk_numbers[0], dim=-2)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W)
        vcolumn_1_part1 = shunted(v1_part1, chunk=self.chunk_numbers[0], dim=-2).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part1 = shunted(v2_part1, chunk=self.chunk_numbers[0], dim=-2).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part1 = torch.matmul(qcolumn_part1, kcolumn_part1) * self.scale
        attn_column_part1 = attn_column_part1.softmax(dim=-1)
        xx_column_1_part1 = torch.matmul(attn_column_part1, vcolumn_1_part1)  # B nH W C
        xx_column_1_part1 = self.proj_encode_column_1_part1(xx_column_1_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], W//self.chunk_numbers[0])).unsqueeze(3)
        xx_column_2_part1 = torch.matmul(attn_column_part1, vcolumn_2_part1)  # B nH W C
        xx_column_2_part1 = self.proj_encode_column_2_part1(xx_column_2_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], W//self.chunk_numbers[0])).unsqueeze(3)

        # Part 2
        q_part2 = q[:,(self.nh_kd//2):,:,:]
        k_part2 = k[:,(self.nh_kd//2):,:,:]
        v1_part2 = v_1[:,(self.dh//2):,:,:]
        v2_part2 = v_2[:,(self.dh//2):,:,:]

        qrow_part2 = self.pos_emb_rowq_part2(shunted(q_part2, chunk=self.chunk_numbers[1], dim=-1)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part2 = self.pos_emb_rowk_part2(shunted(k_part2, chunk=self.chunk_numbers[1], dim=-1)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H)
        vrow_1_part2 = shunted(v1_part2, chunk=self.chunk_numbers[1], dim=-1).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part2 = shunted(v2_part2, chunk=self.chunk_numbers[1], dim=-1).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part2 = torch.matmul(qrow_part2, krow_part2) * self.scale
        attn_row_part2 = attn_row_part2.softmax(dim=-1)
        xx_row_1_part2 = torch.matmul(attn_row_part2, vrow_1_part2)  # B nH H C
        xx_row_1_part2 = self.proj_encode_row_1_part2(xx_row_1_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], H//self.chunk_numbers[1])).unsqueeze(-1)
        xx_row_2_part2 = torch.matmul(attn_row_part2, vrow_2_part2)
        xx_row_2_part2 = self.proj_encode_row_2_part2(xx_row_2_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], H//self.chunk_numbers[1])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part2 = self.pos_emb_columnq_part2(shunted(q_part2, chunk=self.chunk_numbers[1], dim=-2)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part2 = self.pos_emb_columnk_part2(shunted(k_part2, chunk=self.chunk_numbers[1], dim=-2)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W)
        vcolumn_1_part2 = shunted(v1_part2, chunk=self.chunk_numbers[1], dim=-2).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part2 = shunted(v2_part2, chunk=self.chunk_numbers[1], dim=-2).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part2 = torch.matmul(qcolumn_part2, kcolumn_part2) * self.scale
        attn_column_part2 = attn_column_part2.softmax(dim=-1)
        xx_column_1_part2 = torch.matmul(attn_column_part2, vcolumn_1_part2)  # B nH W C
        xx_column_1_part2 = self.proj_encode_column_1_part2(xx_column_1_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], W//self.chunk_numbers[1])).unsqueeze(3)
        xx_column_2_part2 = torch.matmul(attn_column_part2, vcolumn_2_part2)  # B nH W C
        xx_column_2_part2 = self.proj_encode_column_2_part2(xx_column_2_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], W//self.chunk_numbers[1])).unsqueeze(3)

        # xx_part1 = (xx_row_1_part1-xx_row_2_part1).add(xx_column_1_part1-xx_column_2_part1).reshape(B, self.dh//2, H, W)
        xx_part1 = torch.abs(xx_row_1_part1-xx_row_2_part1).add(torch.abs(xx_column_1_part1-xx_column_2_part1)).reshape(B, self.dh//2, H, W)
        # xx_part2 = (xx_row_1_part2-xx_row_2_part2).add(xx_column_1_part2-xx_column_2_part2).reshape(B, self.dh//2, H, W)
        xx_part2 = torch.abs(xx_row_1_part2-xx_row_2_part2).add(torch.abs(xx_column_1_part2-xx_column_2_part2)).reshape(B, self.dh//2, H, W)
        xx = torch.cat([xx_part1,xx_part2],dim=1)
        # xx = (v_1-v_2).add(xx) # 
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out1 = att * qkv_1
        out2 = att * qkv_2

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            b, c, h, w = att.shape
            loss_att = 0
            # for i in range(c):
            #     # att_pred = torch.where(att[:,i,:,:] > 0.5, torch.ones_like(att[:,i,:,:]), torch.zeros_like(att[:,i,:,:])).float()
            #     loss_att += self.loss_generator(att[:,i,:,:], label) # self.loss_generator  BCEDiceLoss  .unsqueeze(1)
            # loss_att /= c
            att_pred = torch.where(att > 0.5, torch.ones_like(att), torch.zeros_like(att)).float()
            loss_att = self.loss_generator(att_pred, label) # self.loss_generator  BCEDiceLoss
            loss_res = self.loss_generator(qkv_1*(1-label), qkv_2*(1-label))
            return out1, out2, att, loss_att, loss_res
        else:
            return out1, out2, torch.mean(att, 1).unsqueeze(1)

class Interaction_attention_various_head_four(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                #  chunk_number=4,
                 chunk_numbers=[2,4,8,16],
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        # self.chunk_number = chunk_number
        self.chunk_numbers = chunk_numbers

        self.to_q_1 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        # part 1
        self.proj_encode_row_1_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq_part1 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[0])
        self.pos_emb_rowk_part1 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[0])

        self.proj_encode_column_1_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part1 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[0])
        self.pos_emb_columnk_part1 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[0])
        
        # part 2
        self.proj_encode_row_1_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))

        self.pos_emb_rowq_part2 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[1])
        self.pos_emb_rowk_part2 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[1])

        self.proj_encode_column_1_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part2 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[1])
        self.pos_emb_columnk_part2 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[1])
        
        # part 3
        self.proj_encode_row_1_part3 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part3 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq_part3 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[2])
        self.pos_emb_rowk_part3 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[2])

        self.proj_encode_column_1_part3 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part3 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part3 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[2])
        self.pos_emb_columnk_part3 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[2])

        # part 4
        self.proj_encode_row_1_part4 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part4 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq_part4 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[3])
        self.pos_emb_rowk_part4 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[3])

        self.proj_encode_column_1_part4 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part4 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//4, self.dh//4, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part4 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[3])
        self.pos_emb_columnk_part4 = SqueezeAxialPositionalEmbedding(nh_kd//4, 16, chunted=self.chunk_numbers[3])

        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        q = torch.cat([q_1,q_2],dim=1) # B, nh_kd, H, W

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        k = torch.abs(k_1-k_2) # B, nh_kd, H, W

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2) # B, self.dh, H, W
        
        # detail enhance
        qkv_1 = torch.cat([q, k, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q, k, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        q_part1 = q[:,:(self.nh_kd//4),:,:]
        k_part1 = k[:,:(self.nh_kd//4),:,:]
        v1_part1 = v_1[:,:(self.dh//4),:,:]
        v2_part1 = v_2[:,:(self.dh//4),:,:]
        # squeeze axial attention
        ## squeeze row
        # B, chunt, C, H, W 
        qrow_part1 = self.pos_emb_rowq_part1(shunted(q_part1, chunk=self.chunk_numbers[0], dim=-1)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part1 = self.pos_emb_rowk_part1(shunted(k_part1, chunk=self.chunk_numbers[0], dim=-1)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H)
        vrow_1_part1 = shunted(v1_part1, chunk=self.chunk_numbers[0], dim=-1).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part1 = shunted(v2_part1, chunk=self.chunk_numbers[0], dim=-1).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part1 = torch.matmul(qrow_part1, krow_part1) * self.scale
        attn_row_part1 = attn_row_part1.softmax(dim=-1)
        xx_row_1_part1 = torch.matmul(attn_row_part1, vrow_1_part1)  # B nH H C
        xx_row_1_part1 = self.proj_encode_row_1_part1(xx_row_1_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[0]*self.chunk_numbers[0], H//self.chunk_numbers[0])).unsqueeze(-1)
        xx_row_2_part1 = torch.matmul(attn_row_part1, vrow_2_part1)
        xx_row_2_part1 = self.proj_encode_row_2_part1(xx_row_2_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[0]*self.chunk_numbers[0], H//self.chunk_numbers[0])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part1 = self.pos_emb_columnq_part1(shunted(q_part1, chunk=self.chunk_numbers[0], dim=-2)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part1 = self.pos_emb_columnk_part1(shunted(k_part1, chunk=self.chunk_numbers[0], dim=-2)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W)
        vcolumn_1_part1 = shunted(v1_part1, chunk=self.chunk_numbers[0], dim=-2).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part1 = shunted(v2_part1, chunk=self.chunk_numbers[0], dim=-2).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part1 = torch.matmul(qcolumn_part1, kcolumn_part1) * self.scale
        attn_column_part1 = attn_column_part1.softmax(dim=-1)
        xx_column_1_part1 = torch.matmul(attn_column_part1, vcolumn_1_part1)  # B nH W C
        xx_column_1_part1 = self.proj_encode_column_1_part1(xx_column_1_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[0]*self.chunk_numbers[0], W//self.chunk_numbers[0])).unsqueeze(3)
        xx_column_2_part1 = torch.matmul(attn_column_part1, vcolumn_2_part1)  # B nH W C
        xx_column_2_part1 = self.proj_encode_column_2_part1(xx_column_2_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[0]*self.chunk_numbers[0], W//self.chunk_numbers[0])).unsqueeze(3)

        # Part 2
        q_part2 = q[:,(self.nh_kd//4):(self.nh_kd//2),:,:]
        k_part2 = k[:,(self.nh_kd//4):(self.nh_kd//2),:,:]
        v1_part2 = v_1[:,(self.dh//4):(self.dh//2),:,:]
        v2_part2 = v_2[:,(self.dh//4):(self.dh//2),:,:]

        qrow_part2 = self.pos_emb_rowq_part2(shunted(q_part2, chunk=self.chunk_numbers[1], dim=-1)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part2 = self.pos_emb_rowk_part2(shunted(k_part2, chunk=self.chunk_numbers[1], dim=-1)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H)
        vrow_1_part2 = shunted(v1_part2, chunk=self.chunk_numbers[1], dim=-1).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part2 = shunted(v2_part2, chunk=self.chunk_numbers[1], dim=-1).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part2 = torch.matmul(qrow_part2, krow_part2) * self.scale
        attn_row_part2 = attn_row_part2.softmax(dim=-1)
        xx_row_1_part2 = torch.matmul(attn_row_part2, vrow_1_part2)  # B nH H C
        xx_row_1_part2 = self.proj_encode_row_1_part2(xx_row_1_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[1]*self.chunk_numbers[1], H//self.chunk_numbers[1])).unsqueeze(-1)
        xx_row_2_part2 = torch.matmul(attn_row_part2, vrow_2_part2)
        xx_row_2_part2 = self.proj_encode_row_2_part2(xx_row_2_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[1]*self.chunk_numbers[1], H//self.chunk_numbers[1])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part2 = self.pos_emb_columnq_part2(shunted(q_part2, chunk=self.chunk_numbers[1], dim=-2)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part2 = self.pos_emb_columnk_part2(shunted(k_part2, chunk=self.chunk_numbers[1], dim=-2)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W)
        vcolumn_1_part2 = shunted(v1_part2, chunk=self.chunk_numbers[1], dim=-2).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part2 = shunted(v2_part2, chunk=self.chunk_numbers[1], dim=-2).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part2 = torch.matmul(qcolumn_part2, kcolumn_part2) * self.scale
        attn_column_part2 = attn_column_part2.softmax(dim=-1)
        xx_column_1_part2 = torch.matmul(attn_column_part2, vcolumn_1_part2)  # B nH W C
        xx_column_1_part2 = self.proj_encode_column_1_part2(xx_column_1_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[1]*self.chunk_numbers[1], W//self.chunk_numbers[1])).unsqueeze(3)
        xx_column_2_part2 = torch.matmul(attn_column_part2, vcolumn_2_part2)  # B nH W C
        xx_column_2_part2 = self.proj_encode_column_2_part2(xx_column_2_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[1]*self.chunk_numbers[1], W//self.chunk_numbers[1])).unsqueeze(3)

        # Part 3
        q_part3 = q[:,(self.nh_kd//2):(self.nh_kd//4*3),:,:]
        k_part3 = k[:,(self.nh_kd//2):(self.nh_kd//4*3),:,:]
        v1_part3 = v_1[:,(self.dh//2):(self.dh//4*3),:,:]
        v2_part3 = v_2[:,(self.dh//2):(self.dh//4*3),:,:]

        qrow_part3 = self.pos_emb_rowq_part3(shunted(q_part3, chunk=self.chunk_numbers[2], dim=-1)).reshape(B, self.chunk_numbers[2], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part3 = self.pos_emb_rowk_part3(shunted(k_part3, chunk=self.chunk_numbers[2], dim=-1)).reshape(B, self.chunk_numbers[2], self.num_heads, -1, H)
        vrow_1_part3 = shunted(v1_part3, chunk=self.chunk_numbers[2], dim=-1).reshape(B, self.chunk_numbers[2], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part3 = shunted(v2_part3, chunk=self.chunk_numbers[2], dim=-1).reshape(B, self.chunk_numbers[2], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part3 = torch.matmul(qrow_part3, krow_part3) * self.scale
        attn_row_part3 = attn_row_part3.softmax(dim=-1)
        xx_row_1_part3 = torch.matmul(attn_row_part3, vrow_1_part3)  # B nH H C
        xx_row_1_part3 = self.proj_encode_row_1_part3(xx_row_1_part3.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[2]*self.chunk_numbers[2], H//self.chunk_numbers[2])).unsqueeze(-1)
        xx_row_2_part3 = torch.matmul(attn_row_part3, vrow_2_part3)
        xx_row_2_part3 = self.proj_encode_row_2_part3(xx_row_2_part3.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[2]*self.chunk_numbers[2], H//self.chunk_numbers[2])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part3 = self.pos_emb_columnq_part3(shunted(q_part3, chunk=self.chunk_numbers[2], dim=-2)).reshape(B, self.chunk_numbers[2], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part3 = self.pos_emb_columnk_part3(shunted(k_part3, chunk=self.chunk_numbers[2], dim=-2)).reshape(B, self.chunk_numbers[2], self.num_heads, -1, W)
        vcolumn_1_part3 = shunted(v1_part3, chunk=self.chunk_numbers[2], dim=-2).reshape(B, self.chunk_numbers[2], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part3 = shunted(v2_part3, chunk=self.chunk_numbers[2], dim=-2).reshape(B, self.chunk_numbers[2], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part3 = torch.matmul(qcolumn_part3, kcolumn_part3) * self.scale
        attn_column_part3 = attn_column_part3.softmax(dim=-1)
        xx_column_1_part3 = torch.matmul(attn_column_part3, vcolumn_1_part3)  # B nH W C
        xx_column_1_part3 = self.proj_encode_column_1_part3(xx_column_1_part3.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[2]*self.chunk_numbers[2], W//self.chunk_numbers[2])).unsqueeze(3)
        xx_column_2_part3 = torch.matmul(attn_column_part3, vcolumn_2_part3)  # B nH W C
        xx_column_2_part3 = self.proj_encode_column_2_part3(xx_column_2_part3.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[2]*self.chunk_numbers[2], W//self.chunk_numbers[2])).unsqueeze(3)

        # Part 4
        q_part4 = q[:,(self.nh_kd//4*3):(self.nh_kd),:,:]
        k_part4 = k[:,(self.nh_kd//4*3):(self.nh_kd),:,:]
        v1_part4 = v_1[:,(self.dh//4*3):(self.dh),:,:]
        v2_part4 = v_2[:,(self.dh//4*3):(self.dh),:,:]

        qrow_part4 = self.pos_emb_rowq_part4(shunted(q_part4, chunk=self.chunk_numbers[3], dim=-1)).reshape(B, self.chunk_numbers[3], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part4 = self.pos_emb_rowk_part4(shunted(k_part4, chunk=self.chunk_numbers[3], dim=-1)).reshape(B, self.chunk_numbers[3], self.num_heads, -1, H)
        vrow_1_part4 = shunted(v1_part4, chunk=self.chunk_numbers[3], dim=-1).reshape(B, self.chunk_numbers[3], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part4 = shunted(v2_part4, chunk=self.chunk_numbers[3], dim=-1).reshape(B, self.chunk_numbers[3], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part4 = torch.matmul(qrow_part4, krow_part4) * self.scale
        attn_row_part4 = attn_row_part4.softmax(dim=-1)
        xx_row_1_part4 = torch.matmul(attn_row_part4, vrow_1_part4)  # B nH H C
        xx_row_1_part4 = self.proj_encode_row_1_part4(xx_row_1_part4.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[3]*self.chunk_numbers[3], H//self.chunk_numbers[3])).unsqueeze(-1)
        xx_row_2_part4 = torch.matmul(attn_row_part4, vrow_2_part4)
        xx_row_2_part4 = self.proj_encode_row_2_part4(xx_row_2_part4.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[3]*self.chunk_numbers[3], H//self.chunk_numbers[3])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part4 = self.pos_emb_columnq_part4(shunted(q_part4, chunk=self.chunk_numbers[3], dim=-2)).reshape(B, self.chunk_numbers[3], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part4 = self.pos_emb_columnk_part4(shunted(k_part4, chunk=self.chunk_numbers[3], dim=-2)).reshape(B, self.chunk_numbers[3], self.num_heads, -1, W)
        vcolumn_1_part4 = shunted(v1_part4, chunk=self.chunk_numbers[3], dim=-2).reshape(B, self.chunk_numbers[3], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part4 = shunted(v2_part4, chunk=self.chunk_numbers[3], dim=-2).reshape(B, self.chunk_numbers[3], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part4 = torch.matmul(qcolumn_part4, kcolumn_part4) * self.scale
        attn_column_part4 = attn_column_part4.softmax(dim=-1)
        xx_column_1_part4 = torch.matmul(attn_column_part4, vcolumn_1_part4)  # B nH W C
        xx_column_1_part4 = self.proj_encode_column_1_part4(xx_column_1_part4.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[3]*self.chunk_numbers[3], W//self.chunk_numbers[3])).unsqueeze(3)
        xx_column_2_part4 = torch.matmul(attn_column_part4, vcolumn_2_part4)  # B nH W C
        xx_column_2_part4 = self.proj_encode_column_2_part4(xx_column_2_part4.permute(0, 2, 4, 1, 3).reshape(B, self.dh//4, self.chunk_numbers[3]*self.chunk_numbers[3], W//self.chunk_numbers[3])).unsqueeze(3)

        xx_part1 = torch.abs(xx_row_1_part1-xx_row_2_part1).add(torch.abs(xx_column_1_part1-xx_column_2_part1)).reshape(B, self.dh//4, H, W)
        xx_part2 = torch.abs(xx_row_1_part2-xx_row_2_part2).add(torch.abs(xx_column_1_part2-xx_column_2_part2)).reshape(B, self.dh//4, H, W)
        xx_part3 = torch.abs(xx_row_1_part3-xx_row_2_part3).add(torch.abs(xx_column_1_part3-xx_column_2_part3)).reshape(B, self.dh//4, H, W)
        xx_part4 = torch.abs(xx_row_1_part4-xx_row_2_part4).add(torch.abs(xx_column_1_part4-xx_column_2_part4)).reshape(B, self.dh//4, H, W)

        xx = torch.cat([xx_part1,xx_part2,xx_part3,xx_part4],dim=1)
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out1 = att * qkv_1
        out2 = att * qkv_2

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            b, c, h, w = att.shape
            loss_att = 0
            # for i in range(c):
            #     # att_pred = torch.where(att[:,i,:,:] > 0.5, torch.ones_like(att[:,i,:,:]), torch.zeros_like(att[:,i,:,:])).float()
            #     loss_att += self.loss_generator(att[:,i,:,:], label) # self.loss_generator  BCEDiceLoss  .unsqueeze(1)
            # loss_att /= c
            att_pred = torch.where(att > 0.5, torch.ones_like(att), torch.zeros_like(att)).float()
            loss_att = self.loss_generator(att_pred, label) # self.loss_generator  BCEDiceLoss
            loss_res = self.loss_generator(qkv_1*(1-label), qkv_2*(1-label))
            return out1, out2, att, loss_att, loss_res
        else:
            return out1, out2, att

class MLLABlock(nn.Module):
    def __init__(self, dim, chunk_numbers=[8,16], norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.in_proj1 = nn.Linear(dim, dim)
        self.act_proj1 = nn.Linear(dim, dim)

        self.in_proj2 = nn.Linear(dim, dim)
        self.act_proj2 = nn.Linear(dim, dim)

        self.dwc1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwc2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.act = nn.SiLU()
        self.attn = Interaction_attention_various_head(dim, chunk_numbers=chunk_numbers) 
        self.out_proj1 = nn.Linear(dim, dim)
        self.out_proj2 = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        L = H*W

        x1 = x1.permute(0, 2, 3, 1).reshape(B, L, C) # .permute(0, 2, 1)
        x1 = self.norm1(x1)
        act_res1 = self.act(self.act_proj1(x1))
        x1 = self.in_proj1(x1).view(B, H, W, C)
        x1 = self.act(self.dwc1(x1.permute(0, 3, 1, 2)))

        x2 = x2.permute(0, 2, 3, 1).reshape(B, L, C) # .permute(0, 2, 1)
        x2 = self.norm1(x2)
        act_res2 = self.act(self.act_proj2(x2))
        x2 = self.in_proj2(x2).view(B, H, W, C)
        x2 = self.act(self.dwc2(x2.permute(0, 3, 1, 2)))

        # Linear Attention
        x1_att, x2_att, att = self.attn(x1, x2)

        x1 = x1 + self.out_proj1(x1_att.permute(0, 2, 3, 1).reshape(B, L, C) * act_res1).view(B, H, W, C).permute(0, 3, 1, 2)
        x2 = x2 + self.out_proj2(x2_att.permute(0, 2, 3, 1).reshape(B, L, C) * act_res2).view(B, H, W, C).permute(0, 3, 1, 2)

        return x1, x2, att
            

class Interaction(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, 
                #  chunk_number=4, 
                 chunk_numbers=[4,8],
                 mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
        super().__init__()   

        dim = in_channels 
        
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  
        # self.attn = Interaction_attention(dim, chunk_number=chunk_number) 
        self.attn = Interaction_attention_various_head(dim, chunk_numbers=chunk_numbers) 

        self.pre_norm = pre_norm
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, t1, t2, labels=None, att=None):
        B, C, H, W = t1.shape

        if att is not None:
            att_val, _ = torch.max(F.interpolate(att, size=(H,W)), dim=1)
            # att = (torch.mean(F.interpolate(att, size=(H,W)), dim=1)+att_val).unsqueeze(1)
            t1 =  self.scale * att_val.unsqueeze(1) * t1 + t1
            t2 =  self.scale * att_val.unsqueeze(1) * t2 + t2

        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        # attention & mlp
        if self.pre_norm:
            if labels is not None:
                x1, x2, att, loss_att, loss_res = self.attn(t1, t2, labels)
            else:
                x1, x2, att = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)
            
            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)
        if labels is not None:
            return t1, t2, att, loss_att, loss_res
        else:
            return t1, t2, att
    
import warnings
warnings.filterwarnings("ignore")
class ASCNet(nn.Module): # 1020
    def __init__(self, num_classes=1, normal_init=True, pretrained=False):
        super(ASCNet, self).__init__()
        
        self.video_len = 8 
        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)
        
        self.interaction3 = Interaction(192, chunk_numbers=[8,16]) # Local_Block(192)# 
        self.interaction2 = Interaction(80, chunk_numbers=[8,16]) # Local_Block(80)# 
        self.interaction1 = Interaction(48, chunk_numbers=[8,16])
        self.interaction0 = Interaction(32, chunk_numbers=[8,16])

        self.backbone = SwinTransformer3D()
        # torch.Size([2, 24, 64, 64]) 32
        # torch.Size([2, 48, 32, 32]) 64
        # torch.Size([2, 96, 16, 16]) 64
        # torch.Size([2, 192, 8, 8]) 

        self.Translayer2_1 = BasicConv2d(96,64,1)
        self.fam32_1 = Align(112, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = Align(56, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

        self.fusion3 = Fusion(192,128,reduction=False) #Difference
        self.fusion2 = Fusion(80,64,reduction=False)
        self.fusion1 = Fusion(48,32,reduction=False)
        self.fusion0 = Fusion(32,16,reduction=False)

        self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out_v = Conv1x1(16, num_classes)

        if normal_init:
            self.init_weights()
    
    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        frames = rearrange(frames, "n l c h w -> n c l h w")
        return frames
    
    def generate_transition_video_tensor(self, frame1, frame2, num_frames=8):
        transition_frames = []

        for t in torch.linspace(0, 1, num_frames):
            weighted_frame1 = frame1 * (1 - t)
            weighted_frame2 = frame2 * t
            blended_frame = weighted_frame1 + weighted_frame2
            transition_frames.append(blended_frame.unsqueeze(0))

        transition_video = torch.cat(transition_frames, dim=0)
        frame = rearrange(transition_video, "l n c h w -> n c l h w")
        return frame
    
    # def forward(self, imgs1, imgs2, labels=None, return_aux=True):
    #     img1 = imgs1[:,:,2,:,:]
    #     img2 = imgs1[:,:,3,:,:]
    #     # x = self.pair_to_video(img1,img2)
    #     x = torch.cat([imgs1, imgs2],dim=2)

    def forward(self, imgs, labels=None, return_aux=True):
        
        img1 = imgs[:,:,2,:,:]
        img2 = imgs[:,:,3,:,:]
        # video = self.pair_to_video(img1,img2)
        # video = self.generate_transition_video_tensor(img1,img2)
        # print(x.shape)

        x, encoder_outputs = self.backbone(imgs)
        # print(encoder_outputs[2].shape)

        if labels is not None:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, att_swim3, loss_att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1),labels) #64
            out4, att_swim4, loss_att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1),labels) #32 ,att_swim3
        else:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
            out4, att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32 ,att=att_swim3

        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        # if labels is not None:
        #     cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1), labels) 
        #     cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), labels, att_3) #, att_3
        #     cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), labels, att_2) #, att_2
        #     cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), labels, att_1) #, att_1
            
        #     loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) + 0.4*(loss_att_swim3 + loss_att_swim4)
        #     loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        # else:
        #     cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
        #     cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), att=att_3) # 80, 32 , 32 -> 32, 32 , 32
        #     cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), att=att_2) # 48, 64 , 64 -> 24, 64 , 64
        #     cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), att=att_1) # 16, 128 , 128 -> 8, 128 , 128

        if labels is not None:
            cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1), labels) 
            cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), labels) #, att_3
            cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), labels) #, att_2
            cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), labels) #, att_1
            
            loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) + 0.4*(loss_att_swim3 + loss_att_swim4)
            loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        else:
            cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
            cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
            cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
            
        fuse3 = self.fusion3(cur1_3,cur2_3) # 128
        fuse2 = self.fusion2(cur1_2,cur2_2) # 64
        fuse1 = self.fusion1(cur1_1,cur2_1) # 32
        fuse0 = self.fusion0(cur1_0,cur2_0) # 16
 
        cat3 = torch.cat([fuse3,out2],1) # 96+128
        cat2 = torch.cat([fuse2,out3],1) # 64+64
        cat1 = torch.cat([fuse1,out4],1) # 24+32

        dec1,output_middle2 = self.decoder1(cat2,cat3) # 64+64 + 96+128
        dec2,output_middle1 = self.decoder2(cat1,dec1) # 24+32 + 128
        dec3,output = self.decoder3(fuse0,dec2) # 16 +

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(self.upsample_pixel(out4))
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att #, loss_res
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att #, loss_res
            else:
                return output


    def init_weights(self):
        # self.global_consrative2.apply(init_weights)
        # self.global_consrative1.apply(init_weights)
        # self.local_consrative2.apply(init_weights)
        # self.local_consrative1.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

class Interaction_Four(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, 
                #  chunk_number=4, 
                 chunk_numbers=[2,4,8,16],
                 mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
        super().__init__()   

        dim = in_channels 
        
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  
        # self.attn = Interaction_attention(dim, chunk_number=chunk_number) 
        self.attn = Interaction_attention_various_head_four(dim, chunk_numbers=chunk_numbers) 

        self.pre_norm = pre_norm
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, t1, t2, labels=None, att=None):
        B, C, H, W = t1.shape

        if att is not None:
            att_val, _ = torch.max(F.interpolate(att, size=(H,W)), dim=1)
            # att = (torch.mean(F.interpolate(att, size=(H,W)), dim=1)+att_val).unsqueeze(1)
            t1 =  self.scale * att_val.unsqueeze(1) * t1 + t1
            t2 =  self.scale * att_val.unsqueeze(1) * t2 + t2

        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        # attention & mlp
        if self.pre_norm:
            if labels is not None:
                x1, x2, att, loss_att, loss_res = self.attn(t1, t2, labels)
            else:
                x1, x2, att = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)
            
            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)
        if labels is not None:
            return t1, t2, att, loss_att, loss_res
        else:
            return t1, t2, att
        
class ASCNet_Four(nn.Module): # 1020
    def __init__(self, num_classes=1, normal_init=True, pretrained=False):
        super(ASCNet_Four, self).__init__()
        
        self.video_len = 8
        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)
        
        self.interaction3 = Interaction_Four(192, chunk_numbers=[2,4,8,16]) # Local_Block(192)# 
        self.interaction2 = Interaction_Four(80, chunk_numbers=[2,4,8,16]) # Local_Block(80)# 
        self.interaction1 = Interaction_Four(48, chunk_numbers=[2,4,8,16])
        self.interaction0 = Interaction_Four(32, chunk_numbers=[2,4,8,16])

        self.backbone = SwinTransformer3D()

        self.Translayer2_1 = BasicConv2d(96,64,1)
        self.fam32_1 = Align(112, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = Align(56, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

        self.fusion3 = Fusion(192,128,reduction=False) #Difference
        self.fusion2 = Fusion(80,64,reduction=False)
        self.fusion1 = Fusion(48,32,reduction=False)
        self.fusion0 = Fusion(32,16,reduction=False)

        self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out_v = Conv1x1(16, num_classes)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None, return_aux=True):
        
        img1 = imgs[:,:,2,:,:]
        img2 = imgs[:,:,3,:,:]

        x, encoder_outputs = self.backbone(imgs)

        if labels is not None:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, att_swim3, loss_att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1),labels) #64
            out4, att_swim4, loss_att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1),labels) #32 ,att_swim3
        else:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
            out4, att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32 ,att=att_swim3

        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        if labels is not None:
            cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1), labels) 
            cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), labels) #, att_3
            cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), labels) #, att_2
            cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), labels) #, att_1
            
            loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) + 0.4*(loss_att_swim3 + loss_att_swim4)
            loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        else:
            cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
            cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
            cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
            
        fuse3 = self.fusion3(cur1_3,cur2_3) # 128
        fuse2 = self.fusion2(cur1_2,cur2_2) # 64
        fuse1 = self.fusion1(cur1_1,cur2_1) # 32
        fuse0 = self.fusion0(cur1_0,cur2_0) # 16
 
        cat3 = torch.cat([fuse3,out2],1) # 96+128
        cat2 = torch.cat([fuse2,out3],1) # 64+64
        cat1 = torch.cat([fuse1,out4],1) # 24+32

        dec1,output_middle2 = self.decoder1(cat2,cat3) # 64+64 + 96+128
        dec2,output_middle1 = self.decoder2(cat1,dec1) # 24+32 + 128
        dec3,output = self.decoder3(fuse0,dec2) # 16 +

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(self.upsample_pixel(out4))
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att #, loss_res
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att #, loss_res
            else:
                return output


    def init_weights(self):
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

def swap_blocks(T1, T2, n, rario=2):
    N, C, H, W = T1.shape
    assert H % n == 0 and W % n == 0, "H and W must be divisible by n."
    
    block_H, block_W = H // n, W // n
    indices = [(i, j) for i in range(n) for j in range(n)]
    np.random.shuffle(indices)
    indices_to_swap = indices[:len(indices) // rario]
    
    T1_swapped = T1.clone()
    T2_swapped = T2.clone()
    
    for i, j in indices_to_swap:
        T1_block = T1[:, :, i*block_H:(i+1)*block_H, j*block_W:(j+1)*block_W]
        T2_block = T2[:, :, i*block_H:(i+1)*block_H, j*block_W:(j+1)*block_W]
        
        T1_swapped[:, :, i*block_H:(i+1)*block_H, j*block_W:(j+1)*block_W] = T2_block
        T2_swapped[:, :, i*block_H:(i+1)*block_H, j*block_W:(j+1)*block_W] = T1_block
    
    return T1_swapped, T2_swapped

def swap_channels(T1, T2, rario=2):
    N, C, H, W = T1.shape
    indices = np.arange(C)
    np.random.shuffle(indices)
    indices_to_swap = indices[:C // rario]
    
    T1_swapped = T1.clone()
    T2_swapped = T2.clone()
    
    for idx in indices_to_swap:
        T1_swapped[:, idx, :, :] = T2[:, idx, :, :]
        T2_swapped[:, idx, :, :] = T1[:, idx, :, :]
    
    return T1_swapped, T2_swapped

def swap_stripes_W(T1, T2, n, rario=2):
    N, C, H, W = T1.shape
    assert W % n == 0, "W must be divisible by n."
    
    stripe_width = W // n
    indices = np.arange(n)
    np.random.shuffle(indices)
    indices_to_swap = indices[:n // rario]
    
    T1_swapped = T1.clone()
    T2_swapped = T2.clone()
    
    for i in indices_to_swap:
        T1_stripe = T1[:, :, :, i*stripe_width:(i+1)*stripe_width]
        T2_stripe = T2[:, :, :, i*stripe_width:(i+1)*stripe_width]
        
        T1_swapped[:, :, :, i*stripe_width:(i+1)*stripe_width] = T2_stripe
        T2_swapped[:, :, :, i*stripe_width:(i+1)*stripe_width] = T1_stripe
    
    return T1_swapped, T2_swapped

def swap_stripes_H(T1, T2, n, rario=2):
    N, C, H, W = T1.shape
    assert H % n == 0, "H must be divisible by n."
    
    stripe_width = H // n
    indices = np.arange(n)
    np.random.shuffle(indices)
    indices_to_swap = indices[:n // rario]
    
    T1_swapped = T1.clone()
    T2_swapped = T2.clone()
    
    for i in indices_to_swap:
        T1_stripe = T1[:, :, i*stripe_width:(i+1)*stripe_width, :]
        T2_stripe = T2[:, :, i*stripe_width:(i+1)*stripe_width, :]
        
        T1_swapped[:, :, i*stripe_width:(i+1)*stripe_width, :] = T2_stripe
        T2_swapped[:, :, i*stripe_width:(i+1)*stripe_width, :] = T1_stripe
    
    return T1_swapped, T2_swapped

def swap_half_pixels(T1, T2, ratio=2):
    assert T1.shape == T2.shape, "T1 and T2 must have the same shape."
    N, C, H, W = T1.shape
    
    # 对于每一个通道分别处理
    for n in range(N):
        for c in range(C):
            # 生成所有像素的索引
            total_pixels = H * W
            indices = np.arange(total_pixels)
            np.random.shuffle(indices)
            
            # 选择一半的像素进行交换
            half = total_pixels // ratio
            indices_to_swap = indices[:half]
            
            # 计算行和列的索引
            rows_to_swap, cols_to_swap = np.unravel_index(indices_to_swap, (H, W))
            
            # 交换像素
            for i in range(half):
                row, col = rows_to_swap[i], cols_to_swap[i]
                # 交换T1和T2中的像素
                temp = T1[n, c, row, col].clone()
                T1[n, c, row, col] = T2[n, c, row, col]
                T2[n, c, row, col] = temp
    
    return T1, T2

class ASCNet_swap(nn.Module): # 1020
    def __init__(self, num_classes=1, normal_init=True, pretrained=False):
        super(ASCNet_swap, self).__init__()
        
        self.video_len = 8 
        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)
        
        self.interaction3 = Interaction(192, chunk_numbers=[8,16]) # Local_Block(192)# 
        self.interaction2 = Interaction(80, chunk_numbers=[8,16]) # Local_Block(80)# 
        self.interaction1 = Interaction(48, chunk_numbers=[8,16])
        self.interaction0 = Interaction(32, chunk_numbers=[8,16])

        # self.backbone = SwinTransformer3D()

        # self.Translayer2_1 = BasicConv2d(96,64,1)
        # self.fam32_1 = Align(112, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
        # self.Translayer3_1 = BasicConv2d(64,32,1)
        # self.fam43_1 = Align(56, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

        self.fusion3 = Fusion(192,128,reduction=False) #Difference
        self.fusion2 = Fusion(80,64,reduction=False)
        self.fusion1 = Fusion(48,32,reduction=False)
        self.fusion0 = Fusion(32,16,reduction=False)

        # self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
        # self.decoder2 = DecBlock(128+64, 64, num_classes)
        # self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.decoder1 = DecBlock(128+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+32, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out_v = Conv1x1(16, num_classes)

        if normal_init:
            self.init_weights()
    
    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        frames = rearrange(frames, "n l c h w -> n c l h w")
        return frames
    
    def generate_transition_video_tensor(self, frame1, frame2, num_frames=8):
        transition_frames = []

        for t in torch.linspace(0, 1, num_frames):
            weighted_frame1 = frame1 * (1 - t)
            weighted_frame2 = frame2 * t
            blended_frame = weighted_frame1 + weighted_frame2
            transition_frames.append(blended_frame.unsqueeze(0))

        transition_video = torch.cat(transition_frames, dim=0)
        frame = rearrange(transition_video, "l n c h w -> n c l h w")
        return frame

    def forward(self, imgs, labels=None, return_aux=True):
        
        img1 = imgs[:,:,2,:,:]
        img2 = imgs[:,:,3,:,:]
        # video = self.pair_to_video(img1,img2)
        # video = self.generate_transition_video_tensor(img1,img2)
        # print(x.shape)

        # x, encoder_outputs = self.backbone(imgs)
        # print(encoder_outputs[2].shape)

        # if labels is not None:
        #     out2 = self.Translayer2_1(encoder_outputs[2]) #64
        #     out3, att_swim3, loss_att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1),labels) #64
        #     out4, att_swim4, loss_att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1),labels) #32 ,att_swim3
        # else:
        #     out2 = self.Translayer2_1(encoder_outputs[2]) #64
        #     out3, att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
        #     out4, att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32 ,att=att_swim3

        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        # Nice
        # c7_swapped, c7_img2_swapped = swap_blocks(c7, c7_img2, 4, rario=16)
        # c6_swapped, c6_img2_swapped = swap_blocks(c6, c6_img2, 4, rario=16)
        # c5_swapped, c5_img2_swapped = swap_blocks(c5, c5_img2, 4, rario=16)
        # c4_swapped, c4_img2_swapped = swap_blocks(c4, c4_img2, 4, rario=16)
        # c3_swapped, c3_img2_swapped = swap_blocks(c3, c3_img2, 4, rario=16)
        # c2_swapped, c2_img2_swapped = swap_blocks(c2, c2_img2, 4, rario=16)
        # c1_swapped, c1_img2_swapped = swap_blocks(c1, c1_img2, 4, rario=16)
        # c0_swapped, c0_img2_swapped = swap_blocks(c0, c0_img2, 4, rario=16)

        # c7_swapped, c7_img2_swapped = swap_channels(c7, c7_img2, rario=64)
        # c6_swapped, c6_img2_swapped = swap_channels(c6, c6_img2, rario=64)
        # c5_swapped, c5_img2_swapped = swap_channels(c5, c5_img2, rario=32)
        # c4_swapped, c4_img2_swapped = swap_channels(c4, c4_img2, rario=32)
        # c3_swapped, c3_img2_swapped = swap_channels(c3, c3_img2, rario=16)
        # c2_swapped, c2_img2_swapped = swap_channels(c2, c2_img2, rario=16)
        # c1_swapped, c1_img2_swapped = swap_channels(c1, c1_img2, rario=16)
        # c0_swapped, c0_img2_swapped = swap_channels(c0, c0_img2, rario=16)


        c7_swapped, c7_img2_swapped = swap_stripes_W(c7, c7_img2, 4, rario=16)
        c6_swapped, c6_img2_swapped = swap_stripes_W(c6, c6_img2, 4, rario=16)
        c5_swapped, c5_img2_swapped = swap_stripes_W(c5, c5_img2, 4, rario=16)
        c4_swapped, c4_img2_swapped = swap_stripes_W(c4, c4_img2, 4, rario=16)
        c3_swapped, c3_img2_swapped = swap_stripes_W(c3, c3_img2, 4, rario=16)
        c2_swapped, c2_img2_swapped = swap_stripes_W(c2, c2_img2, 4, rario=16)
        c1_swapped, c1_img2_swapped = swap_stripes_W(c1, c1_img2, 4, rario=16)
        c0_swapped, c0_img2_swapped = swap_stripes_W(c0, c0_img2, 4, rario=16)

        if labels is not None:
            cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6_swapped,c7_swapped],1), torch.cat([c6_img2_swapped,c7_img2_swapped],1), labels) 
            cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4_swapped,c5_swapped],1), torch.cat([c4_img2_swapped,c5_img2_swapped],1), labels) #, att_3
            cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2_swapped,c3_swapped],1), torch.cat([c2_img2_swapped,c3_img2_swapped],1), labels) #, att_2
            cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0_swapped,c1_swapped],1), torch.cat([c0_img2_swapped,c1_img2_swapped],1), labels) #, att_1
            
            loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) # + 0.4*(loss_att_swim3 + loss_att_swim4)
            loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        else:
            cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6_swapped,c7_swapped],1), torch.cat([c6_img2_swapped,c7_img2_swapped],1)) # 192, 16 , 16 -> 64, 16 , 16
            cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4_swapped,c5_swapped],1), torch.cat([c4_img2_swapped,c5_img2_swapped],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2_swapped,c3_swapped],1), torch.cat([c2_img2_swapped,c3_img2_swapped],1)) # 48, 64 , 64 -> 24, 64 , 64
            cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0_swapped,c1_swapped],1), torch.cat([c0_img2_swapped,c1_img2_swapped],1)) # 16, 128 , 128 -> 8, 128 , 128
            
        # if labels is not None:
        #     cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1), labels) 
        #     cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), labels) #, att_3
        #     cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), labels) #, att_2
        #     cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), labels) #, att_1
            
        #     loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) #+ 0.4*(loss_att_swim3 + loss_att_swim4)
        #     loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        # else:
        #     cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
        #     cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
        #     cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
        #     cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128

        fuse3 = self.fusion3(cur1_3,cur2_3) # 128
        fuse2 = self.fusion2(cur1_2,cur2_2) # 64
        fuse1 = self.fusion1(cur1_1,cur2_1) # 32
        fuse0 = self.fusion0(cur1_0,cur2_0) # 16
 
        # cat3 = torch.cat([fuse3,out2],1) # 96+128
        # cat2 = torch.cat([fuse2,out3],1) # 64+64
        # cat1 = torch.cat([fuse1,out4],1) # 24+32

        # dec1,output_middle2 = self.decoder1(cat2,cat3) # 64+64 + 96+128
        # dec2,output_middle1 = self.decoder2(cat1,dec1) # 24+32 + 128
        # dec3,output = self.decoder3(fuse0,dec2) # 16 +

        dec1,output_middle2 = self.decoder1(fuse2,fuse3) 
        dec2,output_middle1 = self.decoder2(fuse1,dec1) 
        dec3,output = self.decoder3(fuse0,dec2) 

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            # pred_v = self.conv_out_v(self.upsample_pixel(out4))
            # pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            # pred_v = torch.sigmoid(pred_v)

            # if labels is not None:
            #     return output, output_middle1, output_middle2, pred_v, loss_att #, loss_res
            # else:
            #     return output, output_middle1, output_middle2, pred_v
            
            if labels is not None:
                return output, output_middle1, output_middle2, loss_att #, loss_res
            else:
                return output, output_middle1, output_middle2
        else:
            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att #, loss_res
            else:
                return output


    def init_weights(self):
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

class ASCNet_full_swap(nn.Module): # 1020
    def __init__(self, num_classes=1, normal_init=True, pretrained=False):
        super(ASCNet_full_swap, self).__init__()
        
        self.video_len = 8 
        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)
        
        self.interaction3 = Interaction(192, chunk_numbers=[8,16]) # Local_Block(192)# 
        self.interaction2 = Interaction(80, chunk_numbers=[8,16]) # Local_Block(80)# 
        self.interaction1 = Interaction(48, chunk_numbers=[8,16])
        self.interaction0 = Interaction(32, chunk_numbers=[8,16])

        self.backbone = SwinTransformer3D()

        self.Translayer2_1 = BasicConv2d(96,64,1)
        self.fam32_1 = Align(112, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = Align(56, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

        self.fusion3 = Fusion(192,128,reduction=False) #Difference
        self.fusion2 = Fusion(80,64,reduction=False)
        self.fusion1 = Fusion(48,32,reduction=False)
        self.fusion0 = Fusion(32,16,reduction=False)

        self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        # self.decoder1 = DecBlock(128+64, 128, num_classes) 
        # self.decoder2 = DecBlock(128+32, 64, num_classes)
        # self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out_v = Conv1x1(16, num_classes)

        if normal_init:
            self.init_weights()
    
    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        frames = rearrange(frames, "n l c h w -> n c l h w")
        return frames
    
    def generate_transition_video_tensor(self, frame1, frame2, num_frames=8):
        transition_frames = []

        for t in torch.linspace(0, 1, num_frames):
            weighted_frame1 = frame1 * (1 - t)
            weighted_frame2 = frame2 * t
            blended_frame = weighted_frame1 + weighted_frame2
            transition_frames.append(blended_frame.unsqueeze(0))

        transition_video = torch.cat(transition_frames, dim=0)
        frame = rearrange(transition_video, "l n c h w -> n c l h w")
        return frame

    def forward(self, imgs, labels=None, return_aux=True):
        
        img1 = imgs[:,:,2,:,:]
        img2 = imgs[:,:,3,:,:]
        # video = self.pair_to_video(img1,img2)
        # video = self.generate_transition_video_tensor(img1,img2)
        # print(x.shape)

        x, encoder_outputs = self.backbone(imgs)
        # print(encoder_outputs[2].shape)
# 
        if labels is not None:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, att_swim3, loss_att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1),labels) #64
            out4, att_swim4, loss_att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1),labels) #32 ,att_swim3
        else:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
            out4, att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32 ,att=att_swim3

        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        c7_swapped, c7_img2_swapped = swap_blocks(c7, c7_img2, 4, rario=16)
        c6_swapped, c6_img2_swapped = swap_blocks(c6, c6_img2, 4, rario=16)
        c5_swapped, c5_img2_swapped = swap_blocks(c5, c5_img2, 4, rario=16)
        c4_swapped, c4_img2_swapped = swap_blocks(c4, c4_img2, 4, rario=16)
        c3_swapped, c3_img2_swapped = swap_blocks(c3, c3_img2, 4, rario=16)
        c2_swapped, c2_img2_swapped = swap_blocks(c2, c2_img2, 4, rario=16)
        c1_swapped, c1_img2_swapped = swap_blocks(c1, c1_img2, 4, rario=16)
        c0_swapped, c0_img2_swapped = swap_blocks(c0, c0_img2, 4, rario=16)

        # c7_swapped, c7_img2_swapped = swap_channels(c7, c7_img2, rario=64)
        # c6_swapped, c6_img2_swapped = swap_channels(c6, c6_img2, rario=64)
        # c5_swapped, c5_img2_swapped = swap_channels(c5, c5_img2, rario=32)
        # c4_swapped, c4_img2_swapped = swap_channels(c4, c4_img2, rario=32)
        # c3_swapped, c3_img2_swapped = swap_channels(c3, c3_img2, rario=16)
        # c2_swapped, c2_img2_swapped = swap_channels(c2, c2_img2, rario=16)
        # c1_swapped, c1_img2_swapped = swap_channels(c1, c1_img2, rario=16)
        # c0_swapped, c0_img2_swapped = swap_channels(c0, c0_img2, rario=16)

        if labels is not None:
            cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6_swapped,c7_swapped],1), torch.cat([c6_img2_swapped,c7_img2_swapped],1), labels) 
            cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4_swapped,c5_swapped],1), torch.cat([c4_img2_swapped,c5_img2_swapped],1), labels) #, att_3
            cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2_swapped,c3_swapped],1), torch.cat([c2_img2_swapped,c3_img2_swapped],1), labels) #, att_2
            cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0_swapped,c1_swapped],1), torch.cat([c0_img2_swapped,c1_img2_swapped],1), labels) #, att_1
            
            loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) + 0.4*(loss_att_swim3 + loss_att_swim4)
            loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        else:
            cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6_swapped,c7_swapped],1), torch.cat([c6_img2_swapped,c7_img2_swapped],1)) # 192, 16 , 16 -> 64, 16 , 16
            cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4_swapped,c5_swapped],1), torch.cat([c4_img2_swapped,c5_img2_swapped],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2_swapped,c3_swapped],1), torch.cat([c2_img2_swapped,c3_img2_swapped],1)) # 48, 64 , 64 -> 24, 64 , 64
            cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0_swapped,c1_swapped],1), torch.cat([c0_img2_swapped,c1_img2_swapped],1)) # 16, 128 , 128 -> 8, 128 , 128
            
        # if labels is not None:
        #     cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1), labels) 
        #     cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), labels) #, att_3
        #     cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), labels) #, att_2
        #     cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), labels) #, att_1
            
        #     loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) #+ 0.4*(loss_att_swim3 + loss_att_swim4)
        #     loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
        # else:
        #     cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
        #     cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
        #     cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
        #     cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128

        fuse3 = self.fusion3(cur1_3,cur2_3) # 128
        fuse2 = self.fusion2(cur1_2,cur2_2) # 64
        fuse1 = self.fusion1(cur1_1,cur2_1) # 32
        fuse0 = self.fusion0(cur1_0,cur2_0) # 16
 
        cat3 = torch.cat([fuse3,out2],1) # 96+128
        cat2 = torch.cat([fuse2,out3],1) # 64+64
        cat1 = torch.cat([fuse1,out4],1) # 24+32

        dec1,output_middle2 = self.decoder1(cat2,cat3) # 64+64 + 96+128
        dec2,output_middle1 = self.decoder2(cat1,dec1) # 24+32 + 128
        dec3,output = self.decoder3(fuse0,dec2) # 16 +

        # dec1,output_middle2 = self.decoder1(fuse2,fuse3) 
        # dec2,output_middle1 = self.decoder2(fuse1,dec1) 
        # dec3,output = self.decoder3(fuse0,dec2) 

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(self.upsample_pixel(out4))
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att #, loss_res
            else:
                return output, output_middle1, output_middle2, pred_v
            
            # if labels is not None:
            #     return output, output_middle1, output_middle2, loss_att #, loss_res
            # else:
            #     return output, output_middle1, output_middle2
        else:
            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att #, loss_res
            else:
                return output


    def init_weights(self):
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

def get_random_indices(num_samples, H, W, except_indices=(0, 0)):
    indices = [(np.random.randint(0, H), np.random.randint(0, W)) for _ in range(num_samples)]
    # 确保随机选择的索引不包括当前中心像素点的索引
    return [(i, j) for i, j in indices if (i, j) != except_indices]

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class PixelwiseAttention(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                #  chunk_number=4,
                 chunk_numbers=[8,16],
                 attn_ratio=2,
                 activation=nn.ReLU,
                 mlp_ratio=4,
                 drop_path=0.1, 
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        # self.chunk_number = chunk_number
        self.chunk_numbers = chunk_numbers

        self.to_q_1 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row_1_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])
        self.pos_emb_rowk_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])

        self.proj_encode_column_1_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])
        self.pos_emb_columnk_part1 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[0])

        self.proj_encode_row_1_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        # part 2
        self.pos_emb_rowq_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])
        self.pos_emb_rowk_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])

        self.proj_encode_column_1_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2_part2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh//2, self.dh//2, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])
        self.pos_emb_columnk_part2 = SqueezeAxialPositionalEmbedding(nh_kd//2, 16, chunted=self.chunk_numbers[1])
        
        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.act2 = nn.ReLU6()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.DwConv_x1_1 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.DwConv_x2_1 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)

        self.f_x1_1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f_x2_1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f_x1_2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f_x2_2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)

        self.g_x1 = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.g_x2 = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.DwConv_x1_2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.DwConv_x2_2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        q = torch.cat([q_1,q_2],dim=1) # B, nh_kd, H, W

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        k = torch.abs(k_1-k_2) # B, nh_kd, H, W

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2) # B, self.dh, H, W
        
        # detail enhance
        qkv_1 = torch.cat([q, k, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q, k, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        q_part1 = q[:,:(self.nh_kd//2),:,:]
        k_part1 = k[:,:(self.nh_kd//2),:,:]
        v1_part1 = v_1[:,:(self.dh//2),:,:]
        v2_part1 = v_2[:,:(self.dh//2),:,:]
        # squeeze axial attention
        ## squeeze row
        # B, chunt, C, H, W 
        qrow_part1 = self.pos_emb_rowq_part1(shunted(q_part1, chunk=self.chunk_numbers[0], dim=-1)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part1 = self.pos_emb_rowk_part1(shunted(k_part1, chunk=self.chunk_numbers[0], dim=-1)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H)
        vrow_1_part1 = shunted(v1_part1, chunk=self.chunk_numbers[0], dim=-1).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part1 = shunted(v2_part1, chunk=self.chunk_numbers[0], dim=-1).reshape(B, self.chunk_numbers[0], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part1 = torch.matmul(qrow_part1, krow_part1) * self.scale
        attn_row_part1 = attn_row_part1.softmax(dim=-1)
        xx_row_1_part1 = torch.matmul(attn_row_part1, vrow_1_part1)  # B nH H C
        xx_row_1_part1 = self.proj_encode_row_1_part1(xx_row_1_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], H//self.chunk_numbers[0])).unsqueeze(-1)
        xx_row_2_part1 = torch.matmul(attn_row_part1, vrow_2_part1)
        xx_row_2_part1 = self.proj_encode_row_2_part1(xx_row_2_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], H//self.chunk_numbers[0])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part1 = self.pos_emb_columnq_part1(shunted(q_part1, chunk=self.chunk_numbers[0], dim=-2)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part1 = self.pos_emb_columnk_part1(shunted(k_part1, chunk=self.chunk_numbers[0], dim=-2)).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W)
        vcolumn_1_part1 = shunted(v1_part1, chunk=self.chunk_numbers[0], dim=-2).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part1 = shunted(v2_part1, chunk=self.chunk_numbers[0], dim=-2).reshape(B, self.chunk_numbers[0], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part1 = torch.matmul(qcolumn_part1, kcolumn_part1) * self.scale
        attn_column_part1 = attn_column_part1.softmax(dim=-1)
        xx_column_1_part1 = torch.matmul(attn_column_part1, vcolumn_1_part1)  # B nH W C
        xx_column_1_part1 = self.proj_encode_column_1_part1(xx_column_1_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], W//self.chunk_numbers[0])).unsqueeze(3)
        xx_column_2_part1 = torch.matmul(attn_column_part1, vcolumn_2_part1)  # B nH W C
        xx_column_2_part1 = self.proj_encode_column_2_part1(xx_column_2_part1.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[0]*self.chunk_numbers[0], W//self.chunk_numbers[0])).unsqueeze(3)

        # Part 2
        q_part2 = q[:,(self.nh_kd//2):,:,:]
        k_part2 = k[:,(self.nh_kd//2):,:,:]
        v1_part2 = v_1[:,(self.dh//2):,:,:]
        v2_part2 = v_2[:,(self.dh//2):,:,:]

        qrow_part2 = self.pos_emb_rowq_part2(shunted(q_part2, chunk=self.chunk_numbers[1], dim=-1)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow_part2 = self.pos_emb_rowk_part2(shunted(k_part2, chunk=self.chunk_numbers[1], dim=-1)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H)
        vrow_1_part2 = shunted(v1_part2, chunk=self.chunk_numbers[1], dim=-1).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2_part2 = shunted(v2_part2, chunk=self.chunk_numbers[1], dim=-1).reshape(B, self.chunk_numbers[1], self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row_part2 = torch.matmul(qrow_part2, krow_part2) * self.scale
        attn_row_part2 = attn_row_part2.softmax(dim=-1)
        xx_row_1_part2 = torch.matmul(attn_row_part2, vrow_1_part2)  # B nH H C
        xx_row_1_part2 = self.proj_encode_row_1_part2(xx_row_1_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], H//self.chunk_numbers[1])).unsqueeze(-1)
        xx_row_2_part2 = torch.matmul(attn_row_part2, vrow_2_part2)
        xx_row_2_part2 = self.proj_encode_row_2_part2(xx_row_2_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], H//self.chunk_numbers[1])).unsqueeze(-1)

        ## squeeze column     
        qcolumn_part2 = self.pos_emb_columnq_part2(shunted(q_part2, chunk=self.chunk_numbers[1], dim=-2)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn_part2 = self.pos_emb_columnk_part2(shunted(k_part2, chunk=self.chunk_numbers[1], dim=-2)).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W)
        vcolumn_1_part2 = shunted(v1_part2, chunk=self.chunk_numbers[1], dim=-2).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2_part2 = shunted(v2_part2, chunk=self.chunk_numbers[1], dim=-2).reshape(B, self.chunk_numbers[1], self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column_part2 = torch.matmul(qcolumn_part2, kcolumn_part2) * self.scale
        attn_column_part2 = attn_column_part2.softmax(dim=-1)
        xx_column_1_part2 = torch.matmul(attn_column_part2, vcolumn_1_part2)  # B nH W C
        xx_column_1_part2 = self.proj_encode_column_1_part2(xx_column_1_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], W//self.chunk_numbers[1])).unsqueeze(3)
        xx_column_2_part2 = torch.matmul(attn_column_part2, vcolumn_2_part2)  # B nH W C
        xx_column_2_part2 = self.proj_encode_column_2_part2(xx_column_2_part2.permute(0, 2, 4, 1, 3).reshape(B, self.dh//2, self.chunk_numbers[1]*self.chunk_numbers[1], W//self.chunk_numbers[1])).unsqueeze(3)

        xx_part1 = torch.abs(xx_row_1_part1-xx_row_2_part1).add(torch.abs(xx_column_1_part1-xx_column_2_part1)).reshape(B, self.dh//2, H, W)
        xx_part2 = torch.abs(xx_row_1_part2-xx_row_2_part2).add(torch.abs(xx_column_1_part2-xx_column_2_part2)).reshape(B, self.dh//2, H, W)
        xx = torch.cat([xx_part1,xx_part2],dim=1)
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        pixel_cor_x1 = self.DwConv_x1_1(x1)
        pixel_cor_x2 = self.DwConv_x2_2(x2)

        x1_1, x1_2 = self.f_x1_1(pixel_cor_x1), self.f_x1_2(pixel_cor_x1)
        x2_1, x2_2 = self.f_x2_1(pixel_cor_x2), self.f_x2_2(pixel_cor_x2)

        pixel_cor_x1 = self.act2(x2_1) * x1_1
        pixel_cor_x2 = self.act2(x1_2) * x2_2

        pixel_cor_x1 = self.DwConv_x1_2(self.g_x1(pixel_cor_x1))
        pixel_cor_x2 = self.DwConv_x2_2(self.g_x2(pixel_cor_x2))

        out1 = att * qkv_1 + self.drop_path(pixel_cor_x1)
        out2 = att * qkv_2 + self.drop_path(pixel_cor_x2)

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            b, c, h, w = att.shape
            loss_att = 0
            # for i in range(c):
            #     # att_pred = torch.where(att[:,i,:,:] > 0.5, torch.ones_like(att[:,i,:,:]), torch.zeros_like(att[:,i,:,:])).float()
            #     loss_att += self.loss_generator(att[:,i,:,:], label) # self.loss_generator  BCEDiceLoss  .unsqueeze(1)
            # loss_att /= c
            att_pred = torch.where(att > 0.5, torch.ones_like(att), torch.zeros_like(att)).float()
            loss_att = self.loss_generator(att_pred, label) # self.loss_generator  BCEDiceLoss
            loss_res = self.loss_generator(qkv_1*(1-label), qkv_2*(1-label))
            return out1, out2, att, loss_att, loss_res
        else:
            return out1, out2, att

from mmcv.cnn.bricks import ConvModule

import torch.fft

def split_freq_components(T1):
    # 执行二维傅里叶变换
    T1_fft = torch.fft.fft2(T1, dim=(2, 3))

    # 生成低频和高频掩膜
    N, C, H, W = T1.shape
    mask_low_freq = torch.zeros((H, W), dtype=torch.bool, device=T1.device)
    mask_high_freq = torch.ones((H, W), dtype=torch.bool, device=T1.device)
    center_h, center_w = H // 2, W // 2
    radius = min(center_h, center_w) // 4  # 可以调整半径大小以选择不同的频率分量

    # 定义低频区域
    mask_low_freq[center_h-radius:center_h+radius, center_w-radius:center_w+radius] = True
    mask_high_freq = ~mask_low_freq  # 高频掩膜为低频掩膜的补集

    # 将掩膜应用于傅里叶变换后的结果以分离频率成分
    low_freq_component = torch.zeros_like(T1_fft)
    high_freq_component = torch.zeros_like(T1_fft)
    for n in range(N):
        for c in range(C):
            low_freq_component[n, c] = T1_fft[n, c] * mask_low_freq
            high_freq_component[n, c] = T1_fft[n, c] * mask_high_freq

    # 执行逆傅里叶变换以回到空间域
    T1_low_freq = torch.fft.ifft2(low_freq_component, dim=(2, 3)).real
    T1_high_freq = torch.fft.ifft2(high_freq_component, dim=(2, 3)).real

    return T1_low_freq, T1_high_freq

class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q1 = nn.Conv2d(dim, dim//2, kernel_size=1)
        self.q2 = nn.Conv2d(dim, dim//2, kernel_size=1)

        self.kv1 = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.kv2 = nn.Conv2d(dim, dim*2, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr1 = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio+3,
                           stride=sr_ratio,
                           padding=(sr_ratio+3)//2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None,),)
            
            self.sr2 = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio+3,
                           stride=sr_ratio,
                           padding=(sr_ratio+3)//2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None,),)
        else:
            self.sr1 = nn.Identity()
            self.sr2 = nn.Identity()

        self.local_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.local_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.ReLU6()

    def forward(self, x1, x2, relative_pos_enc=None): # x1, x2, 
        B, C, H, W = x1.shape

        # x1_low, x1_high = split_freq_components(x1)
        # x2_low, x2_high = split_freq_components(x2)

        q1 = self.q1(x1)
        q2 = self.q2(x2)
        q = torch.cat([q1,q2],dim=1).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)

        kv1 = self.sr1(x1)
        kv1 = self.local_conv1(kv1) + kv1
        k1, v1 = torch.chunk(self.kv1(kv1), chunks=2, dim=1)

        kv2 = self.sr2(x2)
        kv2 = self.local_conv2(kv2) + kv2
        k2, v2 = torch.chunk(self.kv2(kv2), chunks=2, dim=1)

        k = torch.abs(k1-k2).reshape(B, self.num_heads, C//self.num_heads, -1)

        v1 = v1.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        v2 = v2.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)

        attn = (q @ k) * self.scale

        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], 
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_out1 = (attn @ v1).transpose(-1, -2)
        x_out2 = (attn @ v2).transpose(-1, -2)

        return x_out1.reshape(B, C, H, W), x_out2.reshape(B, C, H, W)

class DynamicConv2d(nn.Module): ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias

        self.dwconv1_1 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1,
                                groups=dim, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.dwconv1_2 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1,
                                groups=dim, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.dwconv2_1 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1,
                                groups=dim, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.dwconv2_2 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1,
                                groups=dim, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.act = nn.ReLU6()

        self.weight1 = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.weight2 = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj1 = nn.Sequential(
            ConvModule(dim, 
                       dim//reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'),),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)

        self.proj2 = nn.Sequential(
            ConvModule(dim, 
                       dim//reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'),),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)
        
        if bias:
            self.bias1 = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
            self.bias2 = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias1 = None
            self.bias2 = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight1, std=0.02)
        nn.init.trunc_normal_(self.weight2, std=0.02)
        if self.bias_type is not None:
            nn.init.trunc_normal_(self.bias1, std=0.02)
            nn.init.trunc_normal_(self.bias2, std=0.02)

    def forward(self, x1, x2):

        B, C, H, W = x1.shape
        x_fuse1 = self.act(self.dwconv1_1(x2)) * self.dwconv1_2(x1)
        x_fuse2 = self.act(self.dwconv2_1(x1)) * self.dwconv2_2(x2)

        scale1 = self.proj1(self.pool(x_fuse1)).reshape(B, self.num_groups, C, self.K, self.K)
        scale1 = torch.softmax(scale1, dim=1)
        weight1 = scale1 * self.weight1.unsqueeze(0)
        weight1 = torch.sum(weight1, dim=1, keepdim=False)
        weight1 = weight1.reshape(-1, 1, self.K, self.K)

        scale2 = self.proj2(self.pool(x_fuse2)).reshape(B, self.num_groups, C, self.K, self.K)
        scale2 = torch.softmax(scale2, dim=1)
        weight2 = scale2 * self.weight2.unsqueeze(0)
        weight2 = torch.sum(weight2, dim=1, keepdim=False)
        weight2 = weight2.reshape(-1, 1, self.K, self.K)

        if self.bias_type is not None:
            scale1 = self.proj1(torch.mean(x_fuse1, dim=[-2, -1], keepdim=True))
            scale1 = torch.softmax(scale1.reshape(B, self.num_groups, C), dim=1)
            bias1 = scale1 * self.bias1.unsqueeze(0)
            bias1 = torch.sum(bias1, dim=1).flatten(0)

            scale2 = self.proj2(torch.mean(x_fuse2, dim=[-2, -1], keepdim=True))
            scale2 = torch.softmax(scale2.reshape(B, self.num_groups, C), dim=1)
            bias2 = scale2 * self.bias2.unsqueeze(0)
            bias2 = torch.sum(bias2, dim=1).flatten(0)
        else:
            bias1 = None
            bias2 = None

        x_out1 = F.conv2d(x_fuse1.reshape(1, -1, H, W),
                     weight=weight1,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias1)
        
        x_out2 = F.conv2d(x_fuse2.reshape(1, -1, H, W),
                     weight=weight2,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias2)
        
        return x_out1.reshape(B, C, H, W), x_out2.reshape(B, C, H, W)

class HybridTokenMixer(nn.Module): ### D-Mixer
    def __init__(self, 
                 dim,
                 kernel_size=7,
                 num_groups=2,
                 num_heads=4,
                 sr_ratio=4,
                 chunk_numbers=[8,16],
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        # self.global_unit = Attention(
        #     dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)
        
        self.global_unit = MLLABlock(dim//2, chunk_numbers=chunk_numbers) 
        
        inner_dim = max(16, dim//reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)
        
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)

    def forward(self, x1, x2, relative_pos_enc=None):
        x1_1, x1_2 = torch.chunk(x1, chunks=2, dim=1)
        x2_1, x2_2 = torch.chunk(x2, chunks=2, dim=1)

        # x1_low, x1_high = split_freq_components(x1_2)
        # x2_low, x2_high = split_freq_components(x2_2)

        x_out1_1, x_out2_1 = self.local_unit(x1_1, x2_1)
        # x_out1_2, x_out2_2 = self.global_unit(x1_2, x2_2, relative_pos_enc)

        x_out1_2, x_out2_2, att = self.global_unit(x1_2, x2_2)

        x_out1 = torch.cat([x_out1_1, x_out1_2], dim=1)
        x_out2 = torch.cat([x_out2_1, x_out2_2], dim=1)

        x_out1 = self.proj(x_out1) + x_out1 ## STE
        x_out2 = self.proj2(x_out2) + x_out2 ## STE
        return x_out1, x_out2, att
    
class Interaction_multi_granularity(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, kernel_size=3, sr_ratio=1, num_groups=2, num_heads=1, chunk_numbers=[8,16],before_attn_dwconv=3, pre_norm=True):
        super().__init__()   

        dim = in_channels 
        
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  
        # self.attn = Interaction_attention(dim, chunk_number=chunk_number) 
        # self.attn = PixelwiseAttention(dim, chunk_numbers=chunk_numbers) 
        self.attn = HybridTokenMixer(dim, kernel_size=kernel_size, num_groups=num_groups, num_heads=num_heads, sr_ratio=sr_ratio, chunk_numbers=chunk_numbers)

        self.pre_norm = pre_norm
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, t1, t2, labels=None, att=None):
        B, C, H, W = t1.shape

        if att is not None:
            att_val, _ = torch.max(F.interpolate(att, size=(H,W)), dim=1)
            t1 =  self.scale * att_val.unsqueeze(1) * t1 + t1
            t2 =  self.scale * att_val.unsqueeze(1) * t2 + t2

        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        if self.pre_norm:
            
            x_out1, x_out2, att = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x_out1)
            t2 = t2 + self.drop_path(x_out2)
            
            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)
        return t1, t2, att
    
class Interaction_hybrid_minxer(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=2, chunk_numbers=[8,16], norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.in_proj1 = nn.Linear(dim, dim)
        self.act_proj1 = nn.Linear(dim, dim)

        self.in_proj2 = nn.Linear(dim, dim)
        self.act_proj2 = nn.Linear(dim, dim)

        self.dwc1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwc2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.act = nn.SiLU()

        self.local_unit = DynamicConv2d(dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Interaction_attention_various_head(dim//2, chunk_numbers=chunk_numbers) 

        self.out_proj1 = nn.Linear(dim, dim)
        self.out_proj2 = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        L = H*W

        x1 = x1.permute(0, 2, 3, 1).reshape(B, L, C) # .permute(0, 2, 1)
        x1 = self.norm1(x1)
        act_res1 = self.act(self.act_proj1(x1))
        x1 = self.in_proj1(x1).view(B, H, W, C)
        x1 = self.act(self.dwc1(x1.permute(0, 3, 1, 2)))

        x2 = x2.permute(0, 2, 3, 1).reshape(B, L, C) # .permute(0, 2, 1)
        x2 = self.norm1(x2)
        act_res2 = self.act(self.act_proj2(x2))
        x2 = self.in_proj2(x2).view(B, H, W, C)
        x2 = self.act(self.dwc2(x2.permute(0, 3, 1, 2)))

        # Linear Attention
        x1_1, x1_2 = torch.chunk(x1, chunks=2, dim=1)
        x2_1, x2_2 = torch.chunk(x2, chunks=2, dim=1)

        # x1_low, x1_high = split_freq_components(x1_2)
        # x2_low, x2_high = split_freq_components(x2_2)

        x_out1_1, x_out2_1 = self.local_unit(x1_1, x2_1)
        x_out1_2, x_out2_2, att = self.global_unit(x1_2, x2_2)

        x_out1 = torch.cat([x_out1_1, x_out1_2], dim=1)
        x_out2 = torch.cat([x_out2_1, x_out2_2], dim=1)

        x1 = x1 + self.out_proj1(x_out1.permute(0, 2, 3, 1).reshape(B, L, C) * act_res1).view(B, H, W, C).permute(0, 3, 1, 2)
        x2 = x2 + self.out_proj2(x_out2.permute(0, 2, 3, 1).reshape(B, L, C) * act_res2).view(B, H, W, C).permute(0, 3, 1, 2)

        return x1, x2, att
    
    
# class ASCNet_multi_granularity(nn.Module): # 1020
#     def __init__(self, num_classes=1, normal_init=True, pretrained=False):
#         super(ASCNet_multi_granularity, self).__init__()
        
#         self.video_len = 8 
#         self.model = ghostnetv2()
#         params=self.model.state_dict() 
#         save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
#         state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
#         self.model.load_state_dict(state_dict)

#         self.interaction3 = Interaction_multi_granularity(192, kernel_size=7, sr_ratio=1, num_groups=2, num_heads=8) # Local_Block(192)# 
#         self.interaction2 = Interaction_multi_granularity(80, kernel_size=7, sr_ratio=2, num_groups=2, num_heads=4) # Local_Block(80)# 
#         self.interaction1 = Interaction_multi_granularity(48, kernel_size=7, sr_ratio=4, num_groups=2, num_heads=2)
#         self.interaction0 = Interaction_multi_granularity(32, kernel_size=7, sr_ratio=8, num_groups=2, num_heads=1)

#         # self.backbone = SwinTransformer3D()

#         # self.Translayer2_1 = BasicConv2d(96,64,1)
#         # self.fam32_1 = Align(112, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
#         # self.Translayer3_1 = BasicConv2d(64,32,1)
#         # self.fam43_1 = Align(56, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

#         self.fusion3 = Fusion(192,128,reduction=False) #Difference
#         self.fusion2 = Fusion(80,64,reduction=False)
#         self.fusion1 = Fusion(48,32,reduction=False)
#         self.fusion0 = Fusion(32,16,reduction=False)

#         # self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
#         # self.decoder2 = DecBlock(128+64, 64, num_classes)
#         # self.decoder3 = DecBlock(64+16, 32, num_classes)

#         # self.decoder1 = DecBlock(192+80, 128, num_classes) 
#         # self.decoder2 = DecBlock(128+48, 64, num_classes)
#         # self.decoder3 = DecBlock(64+32, 32, num_classes)

#         self.decoder1 = DecBlock(128+64, 128, num_classes) 
#         self.decoder2 = DecBlock(128+32, 64, num_classes)
#         self.decoder3 = DecBlock(64+16, 32, num_classes)

#         self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
#         self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
#                                   nn.PixelShuffle(2))
#         self.conv_out_v = Conv1x1(16, num_classes)

#         if normal_init:
#             self.init_weights()

#     def forward(self, imgs, labels=None, return_aux=True):
        
#         img1 = imgs[:,:,2,:,:]
#         img2 = imgs[:,:,3,:,:]
#         # video = self.pair_to_video(img1,img2)
#         # video = self.generate_transition_video_tensor(img1,img2)
#         # print(x.shape)

#         # x, encoder_outputs = self.backbone(imgs)
#         # print(encoder_outputs[2].shape)

#         # if labels is not None:
#         #     out2 = self.Translayer2_1(encoder_outputs[2]) #64
#         #     out3, att_swim3, loss_att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1),labels) #64
#         #     out4, att_swim4, loss_att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1),labels) #32 ,att_swim3
#         # else:
#         #     out2 = self.Translayer2_1(encoder_outputs[2]) #64
#         #     out3, att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
#         #     out4, att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32 ,att=att_swim3

#         c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
#         c1 = self.model.blocks[0](c0) # 16, 128 , 128
#         c2 = self.model.blocks[1](c1) # 24, 64 , 64
#         c3 = self.model.blocks[2](c2) # 24, 64 , 64
#         c4 = self.model.blocks[3](c3) # 40, 32 , 32
#         c5 = self.model.blocks[4](c4) # 40, 32 , 32
#         c6 = self.model.blocks[5](c5) # 80, 16, 16
#         c7 = self.model.blocks[6](c6) # 112, 16, 16

#         c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
#         c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
#         c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
#         c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
#         c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
#         c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
#         c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
#         c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

#         # c7_swapped, c7_img2_swapped = swap_blocks(c7, c7_img2, 4, rario=16)
#         # c6_swapped, c6_img2_swapped = swap_blocks(c6, c6_img2, 4, rario=16)
#         # c5_swapped, c5_img2_swapped = swap_blocks(c5, c5_img2, 4, rario=16)
#         # c4_swapped, c4_img2_swapped = swap_blocks(c4, c4_img2, 4, rario=16)
#         # c3_swapped, c3_img2_swapped = swap_blocks(c3, c3_img2, 4, rario=16)
#         # c2_swapped, c2_img2_swapped = swap_blocks(c2, c2_img2, 4, rario=16)
#         # c1_swapped, c1_img2_swapped = swap_blocks(c1, c1_img2, 4, rario=16)
#         # c0_swapped, c0_img2_swapped = swap_blocks(c0, c0_img2, 4, rario=16)

#         # c7_swapped, c7_img2_swapped = swap_channels(c7, c7_img2, rario=64)
#         # c6_swapped, c6_img2_swapped = swap_channels(c6, c6_img2, rario=64)
#         # c5_swapped, c5_img2_swapped = swap_channels(c5, c5_img2, rario=32)
#         # c4_swapped, c4_img2_swapped = swap_channels(c4, c4_img2, rario=32)
#         # c3_swapped, c3_img2_swapped = swap_channels(c3, c3_img2, rario=16)
#         # c2_swapped, c2_img2_swapped = swap_channels(c2, c2_img2, rario=16)
#         # c1_swapped, c1_img2_swapped = swap_channels(c1, c1_img2, rario=16)
#         # c0_swapped, c0_img2_swapped = swap_channels(c0, c0_img2, rario=16)

#         # if labels is not None:
#         #     cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6_swapped,c7_swapped],1), torch.cat([c6_img2_swapped,c7_img2_swapped],1), labels) 
#         #     cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4_swapped,c5_swapped],1), torch.cat([c4_img2_swapped,c5_img2_swapped],1), labels) #, att_3
#         #     cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2_swapped,c3_swapped],1), torch.cat([c2_img2_swapped,c3_img2_swapped],1), labels) #, att_2
#         #     cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0_swapped,c1_swapped],1), torch.cat([c0_img2_swapped,c1_img2_swapped],1), labels) #, att_1
            
#         #     loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) # + 0.4*(loss_att_swim3 + loss_att_swim4)
#         #     loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
#         # else:
#         #     cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6_swapped,c7_swapped],1), torch.cat([c6_img2_swapped,c7_img2_swapped],1)) # 192, 16 , 16 -> 64, 16 , 16
#         #     cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4_swapped,c5_swapped],1), torch.cat([c4_img2_swapped,c5_img2_swapped],1)) # 80, 32 , 32 -> 32, 32 , 32
#         #     cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2_swapped,c3_swapped],1), torch.cat([c2_img2_swapped,c3_img2_swapped],1)) # 48, 64 , 64 -> 24, 64 , 64
#         #     cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0_swapped,c1_swapped],1), torch.cat([c0_img2_swapped,c1_img2_swapped],1)) # 16, 128 , 128 -> 8, 128 , 128
            
#         # if labels is not None:
#         #     cur1_3, cur2_3, att_3, loss_att3, loss_res3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1), labels) 
#         #     cur1_2, cur2_2, att_2, loss_att2, loss_res2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1), labels) #, att_3
#         #     cur1_1, cur2_1, att_1, loss_att1, loss_res1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1), labels) #, att_2
#         #     cur1_0, cur2_0, att_0, loss_att0, loss_res0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1), labels) #, att_1
            
#         #     loss_att = loss_att3 + loss_att2 + 0.4*(loss_att1 + loss_att0) #+ 0.4*(loss_att_swim3 + loss_att_swim4)
#         #     loss_res = loss_res3 + loss_res2 + 0.4*(loss_res1 + loss_res0)
#         # else:
#         #     cur1_3, cur2_3, att_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
#         #     cur1_2, cur2_2, att_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
#         #     cur1_1, cur2_1, att_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
#         #     cur1_0, cur2_0, att_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128

#         # cur3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
#         # cur2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
#         # cur1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
#         # cur0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128

#         cur1_3, cur2_3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
#         cur1_2, cur2_2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
#         cur1_1, cur2_1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
#         cur1_0, cur2_0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128

#         fuse3 = self.fusion3(cur1_3,cur2_3) # 128
#         fuse2 = self.fusion2(cur1_2,cur2_2) # 64
#         fuse1 = self.fusion1(cur1_1,cur2_1) # 32
#         fuse0 = self.fusion0(cur1_0,cur2_0) # 16
 
#         # cat3 = torch.cat([fuse3,out2],1) # 96+128
#         # cat2 = torch.cat([fuse2,out3],1) # 64+64
#         # cat1 = torch.cat([fuse1,out4],1) # 24+32

#         # dec1,output_middle2 = self.decoder1(cat2,cat3) # 64+64 + 96+128
#         # dec2,output_middle1 = self.decoder2(cat1,dec1) # 24+32 + 128
#         # dec3,output = self.decoder3(fuse0,dec2) # 16 +

#         # dec1,output_middle2 = self.decoder1(cur2,cur3) 
#         # dec2,output_middle1 = self.decoder2(cur1,dec1) 
#         # dec3,output = self.decoder3(cur0,dec2) 

#         dec1,output_middle2 = self.decoder1(fuse2,fuse3) 
#         dec2,output_middle1 = self.decoder2(fuse1,dec1) 
#         dec3,output = self.decoder3(fuse0,dec2) 

#         if return_aux:
#             output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
#             output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
#             # pred_v = self.conv_out_v(self.upsample_pixel(out4))
#             # pred_v = F.interpolate(pred_v, size=output.shape[2:])

#             output = F.interpolate(output, size=img1.shape[2:])
#             output = torch.sigmoid(output)
#             output_middle1 = torch.sigmoid(output_middle1)
#             output_middle2 = torch.sigmoid(output_middle2)
    
#             # pred_v = torch.sigmoid(pred_v)

#             # if labels is not None:
#             #     return output, output_middle1, output_middle2, pred_v, loss_att #, loss_res
#             # else:
#             #     return output, output_middle1, output_middle2, pred_v
            
#             # if labels is not None:
#             #     return output, output_middle1, output_middle2, loss_att #, loss_res
#             # else:
#             #     return output, output_middle1, output_middle2
#             return output, output_middle1, output_middle2
#         else:
#             output = F.interpolate(output, size=img1.shape[2:])
#             output = torch.sigmoid(output)
#             # if labels is not None:
#             #     return output, loss_att #, loss_res
#             # else:
#             #     return output
#             return output


#     def init_weights(self):
#         self.fusion3.apply(init_weights) 
#         self.fusion2.apply(init_weights) 
#         self.fusion1.apply(init_weights) 
#         self.fusion0.apply(init_weights) 
        
#         self.decoder1.apply(init_weights) 
#         self.decoder2.apply(init_weights) 
#         self.decoder3.apply(init_weights) 
#         self.conv_out_v.apply(init_weights) 

class Temporal_fusion(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(Temporal_fusion, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)
    
class MLLANet(nn.Module): # 1020
    def __init__(self, num_classes=3, binary_classes=1, normal_init=True, pretrained=False):
        super(MLLANet, self).__init__()
        
        self.video_len = 8 
        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.interaction3 = Interaction_multi_granularity(192, kernel_size=7, sr_ratio=1, num_groups=2, num_heads=8, chunk_numbers=[8,16]) # Local_Block(192)# 
        self.interaction2 = Interaction_multi_granularity(80, kernel_size=7, sr_ratio=2, num_groups=2, num_heads=4, chunk_numbers=[8,16]) # Local_Block(80)# 
        self.interaction1 = Interaction_multi_granularity(48, kernel_size=7, sr_ratio=4, num_groups=2, num_heads=2, chunk_numbers=[8,16])
        self.interaction0 = Interaction_multi_granularity(32, kernel_size=7, sr_ratio=8, num_groups=2, num_heads=1, chunk_numbers=[8,16])

        # self.interaction3 = Interaction_hybrid_minxer(192, kernel_size=7, chunk_numbers=[8,16]) 
        # self.interaction2 = Interaction_hybrid_minxer(80, kernel_size=7, chunk_numbers=[8,16]) 
        # self.interaction1 = Interaction_hybrid_minxer(48, kernel_size=7, chunk_numbers=[8,16])
        # self.interaction0 = Interaction_hybrid_minxer(32, kernel_size=7, chunk_numbers=[8,16])

        self.backbone = SwinTransformer3D()

        # self.Translayer2_1 = BasicConv2d(96,64,1)
        # self.fam32_1 = MLLA_Align(112, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
        # self.Translayer3_1 = BasicConv2d(64,32,1)
        # self.fam43_1 = MLLA_Align(56, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

        self.Translayer2_1 = BasicConv2d(144,64,1)
        self.fam32_1 = MLLA_Align(136, 64, chunk_number=16) #DRAtt(112,64) # SimpleResBlock DRAtt(112,64) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = MLLA_Align(68, 32, chunk_number=16) # DRAtt(56,32) DRAtt(56,32) # 

        self.fusion3 = Fusion(192,128,reduction=True) #Difference
        self.fusion2 = Fusion(80,64,reduction=True)
        self.fusion1 = Fusion(48,32,reduction=True)
        self.fusion0 = Fusion(32,16,reduction=True)

        # self.fusion3 = Temporal_fusion(192,128) #Difference
        # self.fusion2 = Temporal_fusion(80,64)
        # self.fusion1 = Temporal_fusion(48,32)
        # self.fusion0 = Temporal_fusion(32,16)

        self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.conv_out = Conv1x1(32, binary_classes)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        
        self.conv_out_v = Conv1x1(16, num_classes)
        self.conv_out_v_binary = Conv1x1(16, binary_classes)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None, return_aux=True):
        
        img1 = imgs[:,:,4,:,:]
        img2 = imgs[:,:,5,:,:]

        x, encoder_outputs = self.backbone(imgs)

        out2 = self.Translayer2_1(encoder_outputs[2]) #64
        out3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
        out4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32 ,att=att_swim3

        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        cur1_3, cur2_3, att3 = self.interaction3(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
        cur1_2, cur2_2, att2 = self.interaction2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
        cur1_1, cur2_1, att1 = self.interaction1(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
        cur1_0, cur2_0, att0 = self.interaction0(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128

        fuse3 = self.fusion3(cur1_3,cur2_3) # 128
        fuse2 = self.fusion2(cur1_2,cur2_2) # 64
        fuse1 = self.fusion1(cur1_1,cur2_1) # 32
        fuse0 = self.fusion0(cur1_0,cur2_0) # 16
 
        cat3 = torch.cat([fuse3,out2],1) # 96+128
        cat2 = torch.cat([fuse2,out3],1) # 64+64
        cat1 = torch.cat([fuse1,out4],1) # 24+32

        dec1,output_middle2 = self.decoder1(cat2,cat3) # 64+64 + 96+128
        dec2,output_middle1 = self.decoder2(cat1,dec1) # 24+32 + 128
        dec3,output = self.decoder3(fuse0,dec2) # 16 

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(self.upsample_pixel(out4))
            pred_v_binary = self.conv_out_v_binary(self.upsample_pixel(out4))

            pred_v = F.interpolate(pred_v, size=output.shape[2:])
            pred_v_binary = F.interpolate(pred_v_binary, size=output.shape[2:])

            output = F.interpolate(output, size=img1.shape[2:])
            # output = torch.sigmoid(output)
            # output_middle1 = torch.sigmoid(output_middle1)
            # output_middle2 = torch.sigmoid(output_middle2)
            # pred_v = torch.sigmoid(pred_v)
            pred_v_binary = torch.sigmoid(pred_v_binary)

            output_binary = self.conv_out(F.interpolate(dec3, size=img1.shape[2:]))
            output_binary = torch.sigmoid(output_binary)

            return output, output_middle1, output_middle2, pred_v, output_binary, pred_v_binary, F.interpolate(att0, size=img1.shape[2:]), F.interpolate(att1, size=img1.shape[2:]), F.interpolate(att2, size=img1.shape[2:]), F.interpolate(att3, size=img1.shape[2:])
        else:
            output = F.interpolate(output, size=img1.shape[2:])
            # output = torch.sigmoid(output)
            return output


    def init_weights(self):
        self.interaction3.apply(init_weights) 
        self.interaction2.apply(init_weights) 
        self.interaction1.apply(init_weights) 
        self.interaction0.apply(init_weights) 

        self.Translayer2_1.apply(init_weights) 
        self.fam32_1.apply(init_weights) 
        self.Translayer3_1.apply(init_weights) 
        self.fam43_1.apply(init_weights) 

        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 
        self.conv_out.apply(init_weights) 
        self.conv_out_v_binary.apply(init_weights) 