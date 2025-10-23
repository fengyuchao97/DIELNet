import torch
import torch.nn as nn
from models.resnet import resnet18, resnet34
import torch.nn.functional as F
import numpy as np
import math
from torch import nn, einsum
from thop import clever_format,profile
from einops import rearrange
from timm.models.layers import DropPath
from models.bra2 import BiLevelRoutingCrossAttention
# import misc1

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

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
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


class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))


class BasicConv3D(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        kernel_size,
        bias='auto',
        bn=False, act=False,
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.ConstantPad3d(kernel_size//2, 0.0))
        seq.append(
            nn.Conv3d(
                in_ch, out_ch, kernel_size,
                padding=0,
                bias=(False if bn else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.BatchNorm3d(out_ch))
        if act:
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv3x3x3(BasicConv3D):
    def __init__(self, in_ch, out_ch, bias='auto', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, bias=bias, bn=bn, act=act, **kwargs)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Conv3x3x3(itm_ch, itm_ch, bn=True, act=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y + res)
        return y


class VideoEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64,128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = 4
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3,9,9), stride=(1,4,4), padding=(1,4,4), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU()
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(
                enc_chs[0],
                enc_chs[0]*exps,
                enc_chs[0],
                ds=BasicConv3D(enc_chs[0], enc_chs[0]*exps, 1, bn=True)
            ),
            ResBlock3D(enc_chs[0]*exps, enc_chs[0]*exps, enc_chs[0])
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(
                enc_chs[0]*exps,
                enc_chs[1]*exps,
                enc_chs[1],
                stride=(2,2,2),
                ds=BasicConv3D(enc_chs[0]*exps, enc_chs[1]*exps, 1, stride=(2,2,2), bn=True)
            ),
            ResBlock3D(enc_chs[1]*exps, enc_chs[1]*exps, enc_chs[1])
        )

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            feats.append(x)

        return feats


class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        return self.conv_fuse(x)


class SpatialEncoder(nn.Module):
    def __init__(self, in_ch, topk):
        super().__init__()
        self.n_layers = 3
        self.block = Block(dim=in_ch, topk=topk)

    def forward(self, t1, t2):    # (1,64,64,64) (1,128,32,32) (1,256,16,16)
        feat = self.block(t1, t2)
        return feat


class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):  # 256, 128, 64, 32
        super().__init__()

        enc_chs = enc_chs[::-1]  # 256,128,64,6
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)     # linear interpolation sampling
            for in_ch1, in_ch2, out_ch in zip(enc_chs, (itm_ch,) + dec_chs[:-1], dec_chs)
        ])
        self.conv_out = Conv1x1(dec_chs[-1], 1)

    def forward(self, x, feats):
        feats = feats[::-1]  # 倒置

        # for key in feats:
        #     print(key.shape)

        x = self.conv_bottom(x)  # [1, 256, 16, 16]

        for feat, blk in zip(feats, self.blocks):   # 取 feats and blocks
            # print(feat.shape)
            # print(x.shape)
            x = blk(feat, x)
            # print(x.shape)
            # print('0')

        y = self.conv_out(x)

        return y


class RGBConcatConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(RGBConcatConv, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=3)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return F.relu(self.norm(x))

class ConvReduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvReduce, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=2, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingCrossAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        self.pre_norm = pre_norm
        # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
        #                          DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
        #                          nn.GELU(),
        #                          nn.Linear(int(mlp_ratio * dim), dim)
        #                          )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.concat_chs = RGBConcatConv(in_channels=dim, out_channels=dim)

    def forward(self, t1, t2):
        # conv pos embedding  3×3卷积，一个残差连接
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)
        # permute to NHWC tensor for attention & mlp
        t1 = t1.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        t2 = t2.permute(0, 2, 3, 1)
        x = self.concat_chs(t1, t2)

        # attention & mlp
        if self.pre_norm:
            x = x + self.drop_path(self.attn(self.norm1(t1), self.norm1(t2)))  # (N, H, W, C)
            # x = x + self.drop_path(self.mlp(self.norm1(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class MYNet3(nn.Module):
    def __init__(self, num_classes=2, normal_init=True, in_ch=(64, 128, 256), dec_chs=(256, 128, 64, 32),
                 topks=(4, 16, 32), pretrained=False):  # (4,16,32)
        super(MYNet3, self).__init__()

        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        self.resnet.layer4 = nn.Identity()

        self.crossAttention1 = SpatialEncoder(in_ch[0], topk=topks[0])
        self.crossAttention2 = SpatialEncoder(in_ch[1], topk=topks[1])
        self.crossAttention3 = SpatialEncoder(in_ch[2], topk=topks[2])

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')  # nearest、linear、bilinear

        self.sigmoid = nn.Sigmoid()

        self.encoder_v = VideoEncoder(3, enc_chs=(32, 64))

        self.conv_out_v = Conv1x1(256, 1)

        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2 * ch, ch, norm=True, act=True)
                for ch in (128, 256)
            ]
        )

        self.decoder = SimpleDecoder(in_ch[-1], (6,) + in_ch, dec_chs)

        self.ConvReduce1 = ConvReduce(in_channels=192, out_channels=64)
        self.ConvReduce2 = ConvReduce(in_channels=384, out_channels=128)

        self.concat_chs = RGBConcatConv(in_channels=3, out_channels=3)

        if normal_init:
            self.init_weights()


    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta  # 1/7  在时间维度上进行切割 (1,1,256,256)
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)  # tensor[0., 1., 2., 3., 4., 5., 6., 7.]
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, 8)
        return frames

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)

    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        frames = self.pair_to_video(imgs1, imgs2)
        feats_v = self.encoder_v(frames.transpose(1, 2))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        c1 = self.resnet.conv1(imgs1)
        c1 = self.resnet.bn1(c1)
        c1 = self.resnet.relu(c1)
        c1 = self.resnet.maxpool(c1)

        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c1_imgs2 = self.resnet.conv1(imgs2)
        c1_imgs2 = self.resnet.bn1(c1_imgs2)
        c1_imgs2 = self.resnet.relu(c1_imgs2)
        c1_imgs2 = self.resnet.maxpool(c1_imgs2)

        c1_imgs2 = self.resnet.layer1(c1_imgs2)  # [1, 64, 64, 64]
        c2_imgs2 = self.resnet.layer2(c1_imgs2)  # [1, 128, 32, 32]
        c3_imgs2 = self.resnet.layer3(c2_imgs2)  # [1, 256, 16, 16]

        att0 = torch.cat([imgs1, imgs2], dim=1)
        att1 = self.crossAttention1(c1, c1_imgs2)  # 1,64,64,64
        att2 = self.crossAttention2(c2, c2_imgs2)  # 1,128,32,32
        att3 = self.crossAttention3(c3, c3_imgs2)  # 1,256,16,16

        att1 = self.ConvReduce1(torch.cat([att1, feats_v[0]], dim=1))
        att2 = self.ConvReduce2(torch.cat([att2, feats_v[1]], dim=1))

        feats_p = [att0, att1, att2, att3]

        pred = self.decoder(feats_p[-1], feats_p)
        pred = self.sigmoid(pred)  # (1,1,256,256)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])  # (1,1,32,32)->(1,1,256,256)
            pred_v = self.sigmoid(pred_v)
            return pred, pred_v
        else:
            return pred


    def init_weights(self):
        # self.final.apply(init_weights)
        pass


def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


if __name__ == '__main__':

    input1 = torch.randn(1, 3, 256, 256).cuda(6)
    input2 = torch.randn(1, 3, 256, 256).cuda(6)

    model = MYNet3(pretrained=True).cuda(6)
    misc1.print_module_summary(model, [input1, input2])
    flops, params = profile(model, inputs=(input1, input2))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
    out, out_v = model(input1, input2)
    print(out_v.shape)