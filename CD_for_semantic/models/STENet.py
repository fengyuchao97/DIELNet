import torch
from models.resnet import resnet18
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from timm.models.layers import DropPath
from typing import Tuple
from torch import Tensor

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

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC
        return x


def make_norm(*args, **kwargs):
    norm_layer = nn.BatchNorm2d
    return norm_layer(*args, **kwargs)


def make_act(*args, **kwargs):
    act_layer = nn.ReLU
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

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv2d(in_ch, itm_ch, stride=stride)
        self.conv2 = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        self.conv3 = BasicConv2d(itm_ch, out_ch)
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


class TMEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(32, 64, 128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 3
        self.expansion = 2

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3, 9, 9), stride=(1, 4, 4), padding=(1, 4, 4), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU()
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(
                enc_chs[0],
                enc_chs[0] * exps,
                enc_chs[0],
                ds=BasicConv3D(enc_chs[0], enc_chs[0] * exps, 1, bn=True)
            ),
           ResBlock3D(enc_chs[0] * exps, enc_chs[0] * exps, enc_chs[0])
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(
                enc_chs[0] * exps,
                enc_chs[1] * exps,
                enc_chs[1],
                stride=(2, 2, 2),
                ds=BasicConv3D(enc_chs[0] * exps, enc_chs[1] * exps, 1, stride=(2, 2, 2), bn=True)
            ),
            ResBlock3D(enc_chs[1] * exps, enc_chs[1] * exps, enc_chs[1])
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(
                enc_chs[1] * exps,
                enc_chs[2] * exps,
                enc_chs[1] * exps,
                stride=(2, 2, 2),
                ds=BasicConv3D(enc_chs[1] * exps, enc_chs[2] * exps, 1, stride=(2, 2, 2), bn=True)
            ),
            # ResBlock3D(enc_chs[2] * exps, enc_chs[2] * exps, enc_chs[1] * exps)
        )

    def forward(self, x):
        feats = []

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i + 1}')
            x = layer(x)
            feats.append(x)

        return feats


class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)
        self.conv_out = Conv1x1(out_ch, 3)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        outs = self.conv_fuse(x)
        output = self.conv_out(outs)
        return outs, output

class DecBlock2(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)
        self.conv_out = Conv1x1(out_ch, 3)
        self.conv_out_binary = Conv1x1(out_ch, 1)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        outs = self.conv_fuse(x)
        output = self.conv_out(outs)
        output_binary = self.conv_out_binary(outs)
        return outs, output, output_binary
    
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

        self.decoder1 = DecBlock(256, 256, 256)
        self.decoder2 = DecBlock(128, 256, 128)
        self.decoder3 = DecBlock(64, 128, 64)
        self.decoder4 = DecBlock2(6, 64, 32)

        self.conv_out = Conv1x1(dec_chs[-1], 3)

    def forward(self, x, feats):
        feats = feats[::-1]  # 倒置

        x = self.conv_bottom(x)  # [1, 256, 16, 16]
        x1, middle_out1 = self.decoder1(feats[0], x)  # [1, 256, 16, 16]
        x2, middle_out2 = self.decoder2(feats[1], x1)  # [1, 128, 32, 32]
        x3, middle_out3 = self.decoder3(feats[2], x2)  # [1, 64, 64, 64]
        x4, output, pred_binary = self.decoder4(feats[3], x3)       # [1, 32, 256, 256]

        return output, pred_binary, middle_out1, middle_out2, middle_out3, x1, x2, x3, x4


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

class TopkRouting(nn.Module):
    def __init__(self, qk_dim, n_win, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.p = n_win ** 2
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c) (1,16,64)

        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)

        attn_logit[:, torch.arange(self.p), torch.arange(self.p)] = 1  # 1 to 1 set value on diagonal

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k) 路由权重矩阵

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               )
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  n_win=self.n_win,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x1, x2, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        N, H, W, C = x1.size()
        assert H % self.n_win == 0 and W % self.n_win == 0

        x1 = rearrange(x1, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
        x2 = rearrange(x2, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        q1, kv1 = self.qkv(x1)
        q2, kv2 = self.qkv(x2)

        q1_pix = rearrange(q1, 'n p2 h w c -> n p2 (h w) c')
        q2_pix = rearrange(q2, 'n p2 h w c -> n p2 (h w) c')

        kv_pix1 = self.kv_down(rearrange(kv1, 'n p2 h w c -> (n p2) c h w'))
        kv_pix2 = self.kv_down(rearrange(kv2, 'n p2 h w c -> (n p2) c h w'))
        kv_pix1 = rearrange(kv_pix1, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        kv_pix2 = rearrange(kv_pix2, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q1_win, k2_win = q1.mean([2, 3]), kv2[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk) region-level queries and keys (1,49,64)
        q2_win, k1_win = q2.mean([2, 3]), kv1[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk) region-level queries and keys (1,49,64)

        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe1 = self.lepe(rearrange(kv1[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                    i=self.n_win).contiguous())
        lepe1 = rearrange(lepe1, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
        lepe2 = self.lepe(rearrange(kv2[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe2 = rearrange(lepe2, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        r_weight1, r_idx1 = self.router(q2_win, k1_win)  # both are (n, p^2, topk)
        r_weight2, r_idx2 = self.router(q1_win, k2_win)  # both are (n, p^2, topk)

        kv_pix_sel1 = self.kv_gather(r_idx=r_idx1, r_weight=r_weight1, kv=kv_pix1)
        k_pix_sel1, v_pix_sel1 = kv_pix_sel1.split([self.qk_dim, self.dim], dim=-1)
        kv_pix_sel2 = self.kv_gather(r_idx=r_idx2, r_weight=r_weight2, kv=kv_pix2)
        k_pix_sel2, v_pix_sel2 = kv_pix_sel2.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel1 = rearrange(k_pix_sel1, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                               m=self.num_heads)
        v_pix_sel1 = rearrange(v_pix_sel1, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                               m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q2_pix = rearrange(q2_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                           m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
        k_pix_sel2 = rearrange(k_pix_sel2, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)
        v_pix_sel2 = rearrange(v_pix_sel2, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q1_pix = rearrange(q1_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        attn_weight1 = (q2_pix * self.scale) @ k_pix_sel1
        attn_weight1 = self.attn_act(attn_weight1)
        attn_weight2 = (q1_pix * self.scale) @ k_pix_sel2  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight2 = self.attn_act(attn_weight2)

        out1 = attn_weight1 @ v_pix_sel1
        out1 = rearrange(out1, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                         h=H // self.n_win, w=W // self.n_win)
        out2 = attn_weight2 @ v_pix_sel2
        out2 = rearrange(out2, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                         h=H // self.n_win, w=W // self.n_win)

        out1 = out1 + lepe1
        out2 = out2 + lepe2
        # output linear
        out1 = self.wo(out1)
        out2 = self.wo(out2)

        return out1, out2

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.1, layer_scale_init_value=-1,
                 num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=2, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=3, mlp_dwconv=True,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()     # drop_path=0., mlp_ratio=4, mlp_dwconv=False,
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
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.concat_chs = RGBConcatConv(in_channels=dim, out_channels=dim)

    def forward(self, t1, t2):
        # conv pos embedding  3×3卷积，一个残差连接
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)
        # permute to NHWC tensor for attention
        t1 = t1.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        t2 = t2.permute(0, 2, 3, 1)

        # attention & mlp
        if self.pre_norm:
            x1, x2 = self.attn(self.norm1(t1), self.norm1(t2))
            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)
            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C)
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        # permute back
        x = self.concat_chs(t1, t2)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x


class STENet(nn.Module):
    def __init__(self, len=8, normal_init=True, in_ch=(64, 128, 256), dec_chs=(256, 128, 64, 32),
                 topks=(4, 16, 32), pretrained=False):  # (4,16,32)
        super(STENet, self).__init__()

        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        self.resnet.layer4 = nn.Identity()

        self.crossAttention1 = SpatialEncoder(in_ch[0], topk=topks[0])
        self.crossAttention2 = SpatialEncoder(in_ch[1], topk=topks[1])
        self.crossAttention3 = SpatialEncoder(in_ch[2], topk=topks[2])

        self.sigmoid = nn.Sigmoid()

        self.len = len

        self.encoder_v = TMEncoder(3, enc_chs=(32, 64, 128))  # (32,64)

        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2 * ch, ch, norm=True, act=True)
                for ch in (64, 128, 256)
            ]
        )

        self.conv_out_v = Conv1x1(256, 3)
        self.decoder = SimpleDecoder(in_ch[-1], (6,) + in_ch, dec_chs)
        self.ConvReduce1 = ConvReduce(in_channels=64+64, out_channels=64)
        self.ConvReduce2 = ConvReduce(in_channels=128+128, out_channels=128)
        self.ConvReduce3 = ConvReduce(in_channels=256+256, out_channels=256)
        self.concat_chs = RGBConcatConv(in_channels=3, out_channels=3)


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

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)

    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        frames = self.generate_transition_video_tensor(imgs1, imgs2)
        feats_v = self.encoder_v(frames)

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

        att0 = torch.cat([imgs1, imgs2], dim=1)  # 1,6,256,256
        att1 = self.crossAttention1(c1, c1_imgs2)  # 1,64,64,64
        att2 = self.crossAttention2(c2, c2_imgs2)  # 1,128,32,32
        att3 = self.crossAttention3(c3, c3_imgs2)  # 1,256,16,16

        fusion1 = self.ConvReduce1(torch.cat([att1, feats_v[0]], dim=1))
        fusion2 = self.ConvReduce2(torch.cat([att2, feats_v[1]], dim=1))
        fusion3 = self.ConvReduce3(torch.cat([att3, feats_v[2]], dim=1))

        feats_p = [att0, fusion1, fusion2, fusion3]

        pred, pred_binary, middle_out1, middle_out2, middle_out3, x1, x2, x3, x4 = self.decoder(feats_p[-1], feats_p)  # (32,64)
        # pred = self.sigmoid(pred)  # (1,1,256,256)
        pred_binary = self.sigmoid(pred_binary)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])  # (1,1,16,16)->(1,1,256,256)
            # pred_v = self.sigmoid(pred_v)

            middle_out1 = F.interpolate(middle_out1, size=pred.shape[2:])
            middle_out2 = F.interpolate(middle_out2, size=pred.shape[2:])  # (1,1,256,256)
            middle_out3 = F.interpolate(middle_out3, size=pred.shape[2:])
            # middle_out1 = self.sigmoid(middle_out1)
            # middle_out2 = self.sigmoid(middle_out2)
            # middle_out3 = self.sigmoid(middle_out3)

            return pred, pred_binary, pred_v, middle_out1, middle_out2, middle_out3
        else:
            return pred, pred_binary