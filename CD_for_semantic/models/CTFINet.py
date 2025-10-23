from contextlib import ContextDecorator
import torch
import torch.nn as nn
from .resnet import resnet18,resnet34,resnet50,resnet101
from .pvtv2 import pvt_v2_b1, pvt_v2_b2
import torch.nn.functional as F
import numpy as np
import math
from torch import nn, einsum
from einops import rearrange

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

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)
        

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            window = {window: h}                                                         # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()            
        
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),                          
                groups=cur_head_split*Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W
        # print(q.shape,v.shape)
        # Convolutional relative position encoding.
        # q_img = q                                                             # Shape: [B, h, H*W, Ch].
        # v_img = v                                                             # Shape: [B, h, H*W, Ch].
        # print(q.shape,v.shape)
        v_img = rearrange(v, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)                      # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q* conv_v_img
        # print(EV_hat_img.shape)
        zero = torch.zeros((B, h, 0, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
        EV_hat = torch.cat((zero, EV_hat_img), dim=2)                                # Shape: [B, h, N, Ch].
        # print(EV_hat.shape)
        return EV_hat

class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(self, dim, num_heads=8, qkv_bias=False,  proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)                                       # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window={3:2, 5:3, 7:3})

    def forward(self, q,k,v, size):
        B, N, C = size[0],size[1],size[2]

        # # Generate Q, K, V.
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        # q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=[size[3],size[4]])                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)                                           # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x   

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS, ch_out, drop_rate=0.2,qkv_bias=False):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
            
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

        self.qkv = nn.Linear(channelS, channelS * 3, bias=qkv_bias)
        self.num_heads = 8
        head_dim = channelS// 8
        self.scale = head_dim ** -0.5
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(channelS,self.num_heads,qkv_bias=qkv_bias,proj_drop=drop_rate)
        self.residual = Residual(channelS*2, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size() 
        Yb, Yc, Yh, Yw = Y.size()
        
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        size=[Sb, Sh*Sw, Sc, Sh, Sw ]

        qkv_y= self.qkv(Y1)
        qkv_y=qkv_y.reshape(Sb, Sh*Sw, 3, self.num_heads, Sc // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_y, k_y, v_y = qkv_y[0], qkv_y[1], qkv_y[2] 

        qkv_s = self.qkv(S1)
        qkv_s=qkv_s.reshape(Sb, Sh*Sw, 3, self.num_heads, Sc // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2] 

        cur1 = self.factoratt_crpe(q_y, k_s, v_s, size).permute(0, 2, 1).reshape(Sb, Sc, Sh, Sw) 
        cur2 = self.factoratt_crpe(q_s, k_y, v_y, size).permute(0, 2, 1).reshape(Sb, Sc, Sh, Sw)

        fuse = self.residual(torch.cat([cur1,cur2], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse) ,cur1,cur2
        else:
            return fuse ,cur1,cur2

class MultiHeadCrossAttention2(MultiHeadAttention):
    def __init__(self, channelY, channelS, ch_out, drop_rate=0.2,qkv_bias=False):
        super(MultiHeadCrossAttention2, self).__init__()
        self.Sconv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
            
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

        self.qkv = nn.Linear(channelS, channelS * 3, bias=qkv_bias)
        self.num_heads = 8
        head_dim = channelS// 8
        self.scale = head_dim ** -0.5
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(channelS,self.num_heads,qkv_bias=qkv_bias,proj_drop=drop_rate)
        self.residual = Residual(channelS*2, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size() 
        Yb, Yc, Yh, Yw = Y.size()
        
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        size=[Sb, Sh*Sw, Sc, Sh, Sw ]

        qkv_y= self.qkv(Y1)
        qkv_y=qkv_y.reshape(Sb, Sh*Sw, 3, self.num_heads, Sc // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_y, k_y, v_y = qkv_y[0], qkv_y[1], qkv_y[2] 

        qkv_s = self.qkv(S1)
        qkv_s=qkv_s.reshape(Sb, Sh*Sw, 3, self.num_heads, Sc // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2] 

        cur1 = self.factoratt_crpe(q_y, k_s, v_s, size).permute(0, 2, 1).reshape(Sb, Sc, Sh, Sw) 
        cur2 = self.factoratt_crpe(q_s, k_y, v_y, size).permute(0, 2, 1).reshape(Sb, Sc, Sh, Sw)

        fuse = self.residual(torch.cat([cur1,cur2], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse) #,cur1,cur2
        else:
            return fuse #,cur1,cur2

class MultiHeadCrossAttention3(MultiHeadAttention):
    def __init__(self, channelY, channelS, ch_out, drop_rate=0.2,qkv_bias=False):
        super(MultiHeadCrossAttention3, self).__init__()
        self.Sconv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
            
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

        self.qkv = nn.Linear(channelS, channelS * 3, bias=qkv_bias)
        self.num_heads = 8
        head_dim = channelS// 8
        self.scale = head_dim ** -0.5
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(channelS,self.num_heads,qkv_bias=qkv_bias,proj_drop=drop_rate)

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size() 
        Yb, Yc, Yh, Yw = Y.size()
        
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        size=[Sb, Sh*Sw, Sc, Sh, Sw ]

        qkv_y= self.qkv(Y1)
        qkv_y=qkv_y.reshape(Sb, Sh*Sw, 3, self.num_heads, Sc // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_y, k_y, v_y = qkv_y[0], qkv_y[1], qkv_y[2] 

        qkv_s = self.qkv(S1)
        qkv_s=qkv_s.reshape(Sb, Sh*Sw, 3, self.num_heads, Sc // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2] 

        cur1 = self.factoratt_crpe(q_y, k_s, v_s, size).permute(0, 2, 1).reshape(Sb, Sc, Sh, Sw) 
        cur2 = self.factoratt_crpe(q_s, k_y, v_y, size).permute(0, 2, 1).reshape(Sb, Sc, Sh, Sw)

        return cur1,cur2

class AlignBlock(nn.Module):
    def __init__(self, features):
        super(AlignBlock, self).__init__()

        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(features*2, features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(features),
                        nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
                        )

        self.delta_gen2 = nn.Sequential(
                        nn.Conv2d(features*2, features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(features),
                        nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
                        )


        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    # https://github.com/speedinghzl/AlignSeg/issues/7
    # the normlization item is set to [w/s, h/s] rather than [h/s, w/s]
    # the function bilinear_interpolate_torch_gridsample2 is standard implementation, please use bilinear_interpolate_torch_gridsample2 for training.
    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w/s, h/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device) # not [h/s, w/s]
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        if high_stage.size()[2:] != low_stage.size()[2:]:
            high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage
            
class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel,norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
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

class SelfAtt(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)
 
        return self.gamma * out + input

class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU()) # conv5_s
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU()) # conv5_s

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        # input1 = self.conv1(input1)
        # input2 = self.conv1(input2)
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) #.view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix1 = self.softmax(attn_matrix1)#经过一个softmax进行缩放权重大小.
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix2 = self.softmax(attn_matrix2)#经过一个softmax进行缩放权重大小.
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BAB_Decoder(nn.Module):
    def __init__(self, channel_1=32, channel_2=32, channel_3=32, dilation_1=3, dilation_2=2):
        super(BAB_Decoder, self).__init__()
        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse = BasicConv2d(channel_2 * 3, channel_3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)
        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)
        x3 = self.conv3(x2)
        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))
        return x_fuse

class DMINet_Final(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_Final, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

        # self.resnet = resnet34()
        # # if pretrained:
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
        self.resnet.layer4 = nn.Identity()

        self.cross2 = CrossAtt(256, 256) 
        self.cross3 = CrossAtt(128, 128) 
        self.cross4 = CrossAtt(64, 64) 

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class CrossAtt50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv5_s
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv5_s

        self.query1 = nn.Conv2d(out_channels, out_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(out_channels, out_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(out_channels, out_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(out_channels, out_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(out_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        input1 = self.conv1(input1)
        input2 = self.conv1(input2)
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) #.view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix1 = self.softmax(attn_matrix1)#经过一个softmax进行缩放权重大小.
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix2 = self.softmax(attn_matrix2)#经过一个softmax进行缩放权重大小.
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2

class DMINet50(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet50, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        # self.resnet = resnet18()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

        self.resnet = resnet50()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet50-0676ba61.pth'))
        self.resnet.layer4 = nn.Identity()

        self.cross2 = CrossAtt50(1024, 512) 
        self.cross3 = CrossAtt50(512, 256) 
        self.cross4 = CrossAtt50(256, 128) 

        self.Translayer2_1 = BasicConv2d(512,256,1)
        self.fam32_1 = decode(256,256,256) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(256,128,1)
        self.fam43_1 = decode(128,128,128) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(512,256,1)
        self.fam32_2 = decode(256,256,256)
        self.Translayer3_2 = BasicConv2d(256,128,1)
        self.fam43_2 = decode(128,128,128)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        # print("fyc")
        # print(c1.shape)
        # print(c2.shape)
        # print(c3.shape)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2) # 1024
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2) # 512
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) # 256

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class DMINet101(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet101, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        # self.resnet = resnet18()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

        self.resnet = resnet101()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet101-5d3b4d8f.pth'))
        self.resnet.layer4 = nn.Identity()

        self.cross2 = CrossAtt50(1024, 512) 
        self.cross3 = CrossAtt50(512, 256) 
        self.cross4 = CrossAtt50(256, 128) 

        self.Translayer2_1 = BasicConv2d(512,256,1)
        self.fam32_1 = decode(256,256,256) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(256,128,1)
        self.fam43_1 = decode(128,128,128) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(512,256,1)
        self.fam32_2 = decode(256,256,256)
        self.Translayer3_2 = BasicConv2d(256,128,1)
        self.fam43_2 = decode(128,128,128)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        # print("fyc")
        # print(c1.shape)
        # print(c2.shape)
        # print(c3.shape)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2) # 1024
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2) # 512
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) # 256

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class DMINet_0916(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_0916, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        # self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        # self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        # self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        # self.fam43_2 = decode(64,64,64)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(128*2, 128, 128, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.fam32_2 = nn.Sequential(
            BAB_Decoder(128*2, 128, 128, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out2 = self.upsamplex2(self.Translayer2_1(cross_result2))
        out3 = self.fam32_1(torch.cat((cross_result3, out2), dim=1))
        out4 = self.fam43_1(torch.cat((cross_result4, self.Translayer3_1(out3)), dim=1))

        out2_2 = self.upsamplex2(self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out3_2 = self.fam32_2(torch.cat((torch.abs(cur1_3-cur2_3), out2_2), dim=1))
        out4_2 = self.fam43_2(torch.cat((torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)), dim=1))

        out4_up = self.upsamplex2(out4)
        out4_2_up = self.upsamplex2(out4_2)

        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex4(out3))
        out_2_2 = self.final2_2(self.upsamplex4(out3_2))
        return out_1, out_2, out_1_2, out_2_2 

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

# class CTFINet(nn.Module):
#     def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
#         super(CTFINet, self).__init__()

#         self.show_Feature_Maps = show_Feature_Maps
        
#         self.resnet = resnet18()
#         # if pretrained:
#         self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
#         # # self.resnet.fc = nn.Identity()
#         self.resnet.layer4 = nn.Identity()

#         # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
#         self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
#         self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
#         self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

#         self.Translayer2_1 = BasicConv2d(256,128,1)
#         self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
#         self.Translayer3_1 = BasicConv2d(128,64,1)
#         self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

#         self.Translayer2_2 = BasicConv2d(256,128,1)
#         self.fam32_2 = decode(128,128,128)
#         self.Translayer3_2 = BasicConv2d(128,64,1)
#         self.fam43_2 = decode(64,64,64)

#         self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

#         self.final = nn.Sequential(
#             Conv(64, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         self.final2 = nn.Sequential(
#             Conv(64, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )

#         # self.final_edge = nn.Sequential(
#         #     Conv(32, 32, 3, bn=True, relu=True),
#         #     Conv(32, num_classes, 3, bn=False, relu=False)
#         #     )
#         # self.final2_edge = nn.Sequential(
#         #     Conv(32, 32, 3, bn=True, relu=True),
#         #     Conv(32, num_classes, 3, bn=False, relu=False)
#         #     )

#         self.final_2 = nn.Sequential(
#             Conv(128, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         self.final2_2 = nn.Sequential(
#             Conv(128, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         if normal_init:
#             self.init_weights()

#     def forward(self, imgs1, imgs2, labels=None):

#         c0 = self.resnet.conv1(imgs1)
#         c0 = self.resnet.bn1(c0)
#         c0 = self.resnet.relu(c0)
#         c1 = self.resnet.maxpool(c0)
#         c1 = self.resnet.layer1(c1)
#         c2 = self.resnet.layer2(c1)
#         c3 = self.resnet.layer3(c2)

#         c0_img2 = self.resnet.conv1(imgs2)
#         c0_img2 = self.resnet.bn1(c0_img2)
#         c0_img2 = self.resnet.relu(c0_img2)
#         c1_img2 = self.resnet.maxpool(c0_img2)
#         c1_img2 = self.resnet.layer1(c1_img2)
#         c2_img2 = self.resnet.layer2(c1_img2)
#         c3_img2 = self.resnet.layer3(c2_img2)

#         cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
#         cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
#         cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

#         out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
#         out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

#         out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
#         out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

#         # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
#         # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

#         out4_up = self.upsamplex4(out4)
#         out4_2_up = self.upsamplex4(out4_2)

#         # out_1_edge = self.final_edge(out4_up)
#         # out_2_edge = self.final2_edge(out4_2_up)

#         # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
#         # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
#         out_1 = self.final(out4_up)
#         out_2 = self.final2(out4_2_up)

#         out_1_2 = self.final_2(self.upsamplex8(out3))
#         out_2_2 = self.final2_2(self.upsamplex8(out3_2))
#         return out_1, out_2, out_1_2, out_2_2 #, out_1_edge, out_2_edge

#     def init_weights(self):
#         # self.cross1.apply(init_weights)
#         self.cross2.apply(init_weights)
#         self.cross3.apply(init_weights)        
#         self.cross4.apply(init_weights)

#         self.fam32_1.apply(init_weights)
#         self.Translayer2_1.apply(init_weights)
#         self.fam43_1.apply(init_weights)
#         self.Translayer3_1.apply(init_weights)

#         self.fam32_2.apply(init_weights)
#         self.Translayer2_2.apply(init_weights)
#         self.fam43_2.apply(init_weights)
#         self.Translayer3_2.apply(init_weights)

#         self.final.apply(init_weights)
#         self.final2.apply(init_weights)
#         # self.final_edge.apply(init_weights)
#         # self.final2_edge.apply(init_weights)
#         self.final_2.apply(init_weights)
#         self.final2_2.apply(init_weights)


# class DMINet_level(nn.Module):
#     def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
#         super(DMINet_level, self).__init__()

#         self.show_Feature_Maps = show_Feature_Maps
        
#         self.resnet = resnet18()
#         # if pretrained:
#         self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
#         # # self.resnet.fc = nn.Identity()
#         self.resnet.layer4 = nn.Identity()

#         # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
#         self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
#         self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
#         self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

#         # self.Translayer2_1 = BasicConv2d(256,128,1)
#         # self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
#         self.Translayer3_1 = BasicConv2d(128,64,1)
#         self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

#         # # self.Translayer2_2 = BasicConv2d(256,128,1)
#         # # self.fam32_2 = decode(128,128,128)
#         self.Translayer3_2 = BasicConv2d(128,64,1)
#         self.fam43_2 = decode(64,64,64)

#         self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

#         self.final = nn.Sequential(
#             Conv(64, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         self.final2 = nn.Sequential(
#             Conv(64, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )

#         self.final_2 = nn.Sequential(
#             Conv(128, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         self.final2_2 = nn.Sequential(
#             Conv(128, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         if normal_init:
#             self.init_weights()

#     def forward(self, imgs1, imgs2, labels=None):

#         c0 = self.resnet.conv1(imgs1)
#         c0 = self.resnet.bn1(c0)
#         c0 = self.resnet.relu(c0)
#         c1 = self.resnet.maxpool(c0)
#         c1 = self.resnet.layer1(c1)
#         c2 = self.resnet.layer2(c1)
#         c3 = self.resnet.layer3(c2)

#         c0_img2 = self.resnet.conv1(imgs2)
#         c0_img2 = self.resnet.bn1(c0_img2)
#         c0_img2 = self.resnet.relu(c0_img2)
#         c1_img2 = self.resnet.maxpool(c0_img2)
#         c1_img2 = self.resnet.layer1(c1_img2)
#         c2_img2 = self.resnet.layer2(c1_img2)
#         c3_img2 = self.resnet.layer3(c2_img2)

#         # cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
#         cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
#         cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

#         # out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
#         out3 = cross_result3
#         out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))
#         # out4 = cross_result4

#         # out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
#         out3_2 = torch.abs(cur1_3-cur2_3)
#         out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))
#         # out4_2 = torch.abs(cur1_4-cur2_4)

#         # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
#         # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

#         out4_up = self.upsamplex4(out4)
#         out4_2_up = self.upsamplex4(out4_2)

#         # out_1_edge = self.final_edge(out4_up)
#         # out_2_edge = self.final2_edge(out4_2_up)

#         # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
#         # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
#         out_1 = self.final(out4_up)
#         out_2 = self.final2(out4_2_up)

#         # out_1_2 = self.final_2(self.upsamplex8(out3))
#         # out_2_2 = self.final2_2(self.upsamplex8(out3_2))
#         return out_1, out_2  #, out_1_2, out_2_2 #, out_1_edge, out_2_edge

#     def init_weights(self):
#         # self.cross1.apply(init_weights)
#         self.cross2.apply(init_weights)
#         self.cross3.apply(init_weights)        
#         self.cross4.apply(init_weights)

#         # # self.fam32_1.apply(init_weights)
#         # # self.Translayer2_1.apply(init_weights)
#         self.fam43_1.apply(init_weights)
#         self.Translayer3_1.apply(init_weights)

#         # # self.fam32_2.apply(init_weights)
#         # # self.Translayer2_2.apply(init_weights)
#         self.fam43_2.apply(init_weights)
#         self.Translayer3_2.apply(init_weights)

#         self.final.apply(init_weights)
#         self.final2.apply(init_weights)
#         # self.final_edge.apply(init_weights)
#         # self.final2_edge.apply(init_weights)
#         self.final_2.apply(init_weights)
#         self.final2_2.apply(init_weights)

class DMINet_level(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_level, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        # self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        # self.Translayer3_1 = BasicConv2d(128,64,1)
        # self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        # self.Translayer3_2 = BasicConv2d(128,64,1)
        # self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        # self.final = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final2 = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        # cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        # out3 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        # out3 = cross_result3
        # out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))
        # out4 = cross_result4

        # out3_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        # out3_2 = torch.abs(cur1_3-cur2_3)
        # out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))
        # out4_2 = torch.abs(cur1_4-cur2_4)

        # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        # out4_up = self.upsamplex4(out4)
        # out4_2_up = self.upsamplex4(out4_2)

        # out_1_edge = self.final_edge(out4_up)
        # out_2_edge = self.final2_edge(out4_2_up)

        # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
        # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
        # out_1 = self.final(out4_up)
        # out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1_2, out_2_2  #, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        # self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        # self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        # self.fam43_1.apply(init_weights)
        # self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        # self.fam43_2.apply(init_weights)
        # self.Translayer3_2.apply(init_weights)

        # self.final.apply(init_weights)
        # self.final2.apply(init_weights)
        # self.final_edge.apply(init_weights)
        # self.final2_edge.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class DMINet_level2(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_level2, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        # self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        # self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        # self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        # self.Translayer3_1 = BasicConv2d(128,64,1)
        # self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        # # self.fam32_2 = decode(128,128,128)
        # self.Translayer3_2 = BasicConv2d(128,64,1)
        # self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsamplex16 = nn.Upsample(scale_factor=16, mode='bilinear')

        # self.final = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final2 = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        # cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        # cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.Translayer2_1(cross_result2)
        # out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        # out3 = cross_result3
        # out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))
        # out4 = cross_result4

        out3_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        # out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        # out3_2 = torch.abs(cur1_3-cur2_3)
        # out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))
        # out4_2 = torch.abs(cur1_4-cur2_4)

        # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        # out4_up = self.upsamplex4(out4)
        # out4_2_up = self.upsamplex4(out4_2)

        # out_1_edge = self.final_edge(out4_up)
        # out_2_edge = self.final2_edge(out4_2_up)

        # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
        # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
        # out_1 = self.final(out4_up)
        # out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex16(out3))
        out_2_2 = self.final2_2(self.upsamplex16(out3_2))
        return out_1_2, out_2_2  #, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        # self.cross3.apply(init_weights)        
        # self.cross4.apply(init_weights)

        # self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        # self.fam43_1.apply(init_weights)
        # self.Translayer3_1.apply(init_weights)

        # # self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        # self.fam43_2.apply(init_weights)
        # self.Translayer3_2.apply(init_weights)

        # self.final.apply(init_weights)
        # self.final2.apply(init_weights)
        # self.final_edge.apply(init_weights)
        # self.final2_edge.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class DMINet_solo(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_solo, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer2_1 = BasicConv2d(256,128,1)
        # self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        # self.Translayer3_1 = BasicConv2d(128,64,1)
        # self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        # self.final = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        # self.final_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final2_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

        # self.final_2 = nn.Sequential(
        #     Conv(128, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        # out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        # out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        # out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        # out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        # out_1_edge = self.final_edge(out4_up)
        # out_2_edge = self.final2_edge(out4_2_up)

        # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
        # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
        # out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        # out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_2, out_2_2 #out_1, out_2, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        # self.fam32_1.apply(init_weights)
        # self.Translayer2_1.apply(init_weights)
        # self.fam43_1.apply(init_weights)
        # self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        # self.final.apply(init_weights)
        self.final2.apply(init_weights)
        # self.final_edge.apply(init_weights)
        # self.final2_edge.apply(init_weights)
        # self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class DMINet_double(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_double, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        # out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class Concat(nn.Module):
    def __init__(self, in_channel_x1, in_channel_x2, out_channel, norm_layer=nn.BatchNorm2d):
        super(Concat, self).__init__()
        self.conv3 = nn.Conv2d(in_channel_x1+in_channel_x2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, x1, x2):
        if x2.size()[2:] != x1.size()[2:]:
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        out = torch.cat((x1, x2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class DMINet_double2(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet_double2, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = CrossAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = CrossAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = Concat(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = Concat(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = Concat(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = Concat(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class WithoutAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        # input1 = self.conv1(input1)
        # input2 = self.conv1(input2)
        batch_size, channels, height, width = input1.shape
        
        out1 = input1
        out2 = input2

        feat_sum = self.conv_cat(torch.cat([input1,input2],1))
        return feat_sum, out1, out2


class DIMNet_woa(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DIMNet_woa, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = WithoutAtt(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = WithoutAtt(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = WithoutAtt(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        # self.final_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final2_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        # out_1_edge = self.final_edge(out4_up)
        # out_2_edge = self.final2_edge(out4_2_up)

        # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
        # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        # self.final_edge.apply(init_weights)
        # self.final2_edge.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class CrossAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU()) # conv5_s
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU()) # conv5_s

        self.query1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        # input1 = self.conv1(input1)
        # input2 = self.conv1(input2)
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) #.view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        q1 = q1.view(batch_size, -1, height * width).permute(0, 2, 1)
        q2 = q2.view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q2, k1)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix1 = self.softmax(attn_matrix1)#经过一个softmax进行缩放权重大小.
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q1, k2)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix2 = self.softmax(attn_matrix2)#经过一个softmax进行缩放权重大小.
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2

class DIMNet_cross(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DIMNet_cross, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = CrossAttBlock(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = CrossAttBlock(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = CrossAttBlock(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        # self.final_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final2_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        # out_1_edge = self.final_edge(out4_up)
        # out_2_edge = self.final2_edge(out4_2_up)

        # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
        # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        # self.final_edge.apply(init_weights)
        # self.final2_edge.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)

class SelfAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU()) # conv5_s
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU()) # conv5_s

        self.query1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        # input1 = self.conv1(input1)
        # input2 = self.conv1(input2)
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) #.view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        q1 = q1.view(batch_size, -1, height * width).permute(0, 2, 1)
        q2 = q2.view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q1, k1)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix1 = self.softmax(attn_matrix1)#经过一个softmax进行缩放权重大小.
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q2, k2)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix2 = self.softmax(attn_matrix2)#经过一个softmax进行缩放权重大小.
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2

class DIMNet_self(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DIMNet_self, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = SelfAttBlock(256, 256) # MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = SelfAttBlock(128, 128) # MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = SelfAttBlock(64, 64) # MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        # self.final_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final2_edge = nn.Sequential(
        #     Conv(32, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        # out3_2 = self.fam32_2(torch.abs(cur1_3*cur2_3), self.Translayer2_2(torch.abs(cur1_2*cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4*cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        # out_1_edge = self.final_edge(out4_up)
        # out_2_edge = self.final2_edge(out4_2_up)

        # out_1 = self.final(torch.cat([out4_up, out_1_edge],1))
        # out_2 = self.final2(torch.cat([out4_2_up, out_2_edge],1))
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        # self.final_edge.apply(init_weights)
        # self.final2_edge.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)


# class CrossAtt(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU()) # conv5_s
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU()) # conv5_s

#         self.query1 = nn.Conv2d(out_channels, out_channels // 8, kernel_size = 1, stride = 1)
#         self.key1   = nn.Conv2d(out_channels, out_channels // 4, kernel_size = 1, stride = 1)
#         self.value1 = nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1)

#         self.query2 = nn.Conv2d(out_channels, out_channels // 8, kernel_size = 1, stride = 1)
#         self.key2   = nn.Conv2d(out_channels, out_channels // 4, kernel_size = 1, stride = 1)
#         self.value2 = nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1)

#         self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
#         self.softmax = nn.Softmax(dim = -1)

#         self.conv_cat = nn.Sequential(nn.Conv2d(out_channels*2, out_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(out_channels),
#                                    nn.ReLU()) # conv_f

#     def forward(self, input1, input2):
#         input1 = self.conv1(input1)
#         input2 = self.conv1(input2)
#         batch_size, channels, height, width = input1.shape
#         q1 = self.query1(input1)
#         k1 = self.key1(input1).view(batch_size, -1, height * width)
#         v1 = self.value1(input1).view(batch_size, -1, height * width)

#         q2 = self.query2(input2) #.view(batch_size, -1, height * width).permute(0, 2, 1)
#         k2 = self.key2(input2).view(batch_size, -1, height * width)
#         v2 = self.value2(input2).view(batch_size, -1, height * width)

#         #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
#         q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
#         attn_matrix1 = torch.bmm(q, k1)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix1 = self.softmax(attn_matrix1)#经过一个softmax进行缩放权重大小.
#         out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
#         out1 = out1.view(*input1.shape)
#         out1 = self.gamma * out1 + input1

#         attn_matrix2 = torch.bmm(q, k2)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix2 = self.softmax(attn_matrix2)#经过一个softmax进行缩放权重大小.
#         out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
#         out2 = out2.view(*input2.shape)
#         out2 = self.gamma * out2 + input2

#         feat_sum = self.conv_cat(torch.cat([out1,out2],1))
#         return feat_sum, out1, out2

class CTFINet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # self.resnet = resnet34()
        # if pretrained:
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = AlignBlock(128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = AlignBlock(64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)

        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
        return out_1, out_2, out_1_2, out_2_2 #, out_1_edge, out_2_edge

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)
        
class CTFINet2(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet2, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_3 = BasicConv2d(256,128,1)
        self.fam32_3 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_3 = BasicConv2d(128,64,1)
        self.fam43_3 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final3 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        # out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        # out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out3_3 = self.fam32_3(torch.abs(cur1_3*cur2_3), self.Translayer2_3(torch.abs(cur1_2*cur2_2)))
        out4_3 = self.fam43_3(torch.abs(cur1_4*cur2_4), self.Translayer3_3(out3_3))

        # out = self.final(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_3 = self.final3(self.upsamplex4(out4_3))
        return out_2, out_3 #out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        # self.fam21_1.apply(init_weights)
        # self.Translayer1_1.apply(init_weights)
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)
        self.final.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        self.final2.apply(init_weights)

        self.fam32_3.apply(init_weights)
        self.Translayer2_3.apply(init_weights)
        self.fam43_3.apply(init_weights)
        self.Translayer3_3.apply(init_weights)
        self.final3.apply(init_weights)

class CTFINet3(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet3, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_3 = BasicConv2d(256,128,1)
        self.fam32_3 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_3 = BasicConv2d(128,64,1)
        self.fam43_3 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final3 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        # out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out3_3 = self.fam32_3(torch.abs(cur1_3*cur2_3), self.Translayer2_3(torch.abs(cur1_2*cur2_2)))
        out4_3 = self.fam43_3(torch.abs(cur1_4*cur2_4), self.Translayer3_3(out3_3))

        out = self.final(self.upsamplex4(out4))
        # out_2 = self.final2(self.upsamplex4(out4_2))
        out_3 = self.final3(self.upsamplex4(out4_3))
        return out, out_3 #out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        # self.fam21_1.apply(init_weights)
        # self.Translayer1_1.apply(init_weights)
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)
        self.final.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        self.final2.apply(init_weights)

        self.fam32_3.apply(init_weights)
        self.Translayer2_3.apply(init_weights)
        self.fam43_3.apply(init_weights)
        self.Translayer3_3.apply(init_weights)
        self.final3.apply(init_weights)

class CTFINet4(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet4, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = AlignBlock(64) # decode(64,64,64)

        self.Translayer2_3 = BasicConv2d(256,128,1)
        self.fam32_3 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_3 = BasicConv2d(128,64,1)
        self.fam43_3 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final3 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out3_3 = self.fam32_3(torch.abs(cur1_3*cur2_3), self.Translayer2_3(torch.abs(cur1_2*cur2_2)))
        out4_3 = self.fam43_3(torch.abs(cur1_4*cur2_4), self.Translayer3_3(out3_3))

        out = self.final(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_3 = self.final3(self.upsamplex4(out4_3))
        return out, out_2, out_3 

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        # self.fam21_1.apply(init_weights)
        # self.Translayer1_1.apply(init_weights)
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)
        self.final.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        self.final2.apply(init_weights)

        self.fam32_3.apply(init_weights)
        self.Translayer2_3.apply(init_weights)
        self.fam43_3.apply(init_weights)
        self.Translayer3_3.apply(init_weights)
        self.final3.apply(init_weights)

class CTFINet5(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet5, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.Translayer2_3 = BasicConv2d(256,128,1)

        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.Translayer3_3 = BasicConv2d(128,64,1)

        self.Translayer4_1 = BasicConv2d(64,32,1)
        self.Translayer4_2 = BasicConv2d(64,32,1)
        self.Translayer4_3 = BasicConv2d(64,32,1)

        self.residual2 = Residual(128*3, 64)
        self.residual3 = Residual(64*3, 64)
        self.residual4 = Residual(32*3, 32)

        self.residual5 = Residual(64, 32)

        self.fam32_1 = AlignBlock(64) # decode(128,128,128)
        self.fam43_1 = AlignBlock(32) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        fuse_result2 = self.residual2(torch.cat([self.Translayer2_1(cross_result2), self.Translayer2_2(torch.abs(cur1_2-cur2_2)), self.Translayer2_3(torch.abs(cur1_2*cur2_2))], 1)) #256 -> 128 -> 64
        fuse_result3 = self.residual3(torch.cat([self.Translayer3_1(cross_result3), self.Translayer3_2(torch.abs(cur1_3-cur2_3)), self.Translayer3_3(torch.abs(cur1_3*cur2_3))], 1)) #128 -> 64  -> 64
        fuse_result4 = self.residual4(torch.cat([self.Translayer4_1(cross_result4), self.Translayer4_2(torch.abs(cur1_4-cur2_4)), self.Translayer4_3(torch.abs(cur1_4*cur2_4))], 1)) #64  -> 32

        out3 = self.fam32_1(fuse_result3, fuse_result2)
        out4 = self.fam43_1(fuse_result4, self.residual5(out3))

        out = self.final(self.upsamplex4(out4))
        return out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.Translayer2_1.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.Translayer2_3.apply(init_weights)

        self.Translayer3_1.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        self.Translayer3_3.apply(init_weights)

        self.Translayer4_1.apply(init_weights)
        self.Translayer4_2.apply(init_weights)
        self.Translayer4_3.apply(init_weights)

        self.residual2.apply(init_weights)
        self.residual3.apply(init_weights)
        self.residual4.apply(init_weights)
        self.residual5.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.final.apply(init_weights)

class CTFINet6(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet6, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)

        self.residual2 = Residual(256*3, 256)
        self.residual3 = Residual(128*3, 128)
        self.residual4 = Residual(64*3, 64)
        
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        fuse_result2 = self.residual2(torch.cat([cross_result2, torch.abs(cur1_2-cur2_2), cur1_2*cur2_2], 1)) #256 
        fuse_result3 = self.residual3(torch.cat([cross_result3, torch.abs(cur1_3-cur2_3), cur1_3*cur2_3], 1)) #128 
        fuse_result4 = self.residual4(torch.cat([cross_result4, torch.abs(cur1_4-cur2_4), cur1_4*cur2_4], 1)) #64  

        out3 = self.fam32_1(fuse_result3, self.Translayer2_1(fuse_result2))
        out4 = self.fam43_1(fuse_result4, self.Translayer3_1(out3))

        out = self.final(self.upsamplex4(out4))
        return out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.Translayer2_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.residual2.apply(init_weights)
        self.residual3.apply(init_weights)
        self.residual4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.final.apply(init_weights)

class CTFINet7(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet7, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)

        self.residual2 = Residual(256*2, 256)
        self.residual3 = Residual(128*2, 128)
        self.residual4 = Residual(64*2, 64)
        
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        fuse_result2 = self.residual2(torch.cat([cross_result2, torch.abs(cur1_2-cur2_2)], 1)) #256 
        fuse_result3 = self.residual3(torch.cat([cross_result3, torch.abs(cur1_3-cur2_3)], 1)) #128 
        fuse_result4 = self.residual4(torch.cat([cross_result4, torch.abs(cur1_4-cur2_4)], 1)) #64  

        out3 = self.fam32_1(fuse_result3, self.Translayer2_1(fuse_result2))
        out4 = self.fam43_1(fuse_result4, self.Translayer3_1(out3))

        out = self.final(self.upsamplex4(out4))
        return out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.Translayer2_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.residual2.apply(init_weights)
        self.residual3.apply(init_weights)
        self.residual4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.final.apply(init_weights)

class CTFINet8(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet8, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)

        self.residual2 = Residual(256*2, 256)
        self.residual3 = Residual(128*2, 128)
        self.residual4 = Residual(64*2, 64)
        
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        fuse_result2 = self.residual2(torch.cat([torch.abs(cur1_2-cur2_2), cur1_2*cur2_2], 1)) #256 
        fuse_result3 = self.residual3(torch.cat([torch.abs(cur1_3-cur2_3), cur1_3*cur2_3], 1)) #128 
        fuse_result4 = self.residual4(torch.cat([torch.abs(cur1_4-cur2_4), cur1_4*cur2_4], 1)) #64  

        out3 = self.fam32_1(fuse_result3, self.Translayer2_1(fuse_result2))
        out4 = self.fam43_1(fuse_result4, self.Translayer3_1(out3))

        out = self.final(self.upsamplex4(out4))
        return out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.Translayer2_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.residual2.apply(init_weights)
        self.residual3.apply(init_weights)
        self.residual4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.final.apply(init_weights)

class CTFINet8(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CTFINet8, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)

        self.residual2 = Residual(256*2, 256)
        self.residual3 = Residual(128*2, 128)
        self.residual4 = Residual(64*2, 64)
        
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2) 

        fuse_result2 = self.residual2(torch.cat([cross_result2, cur1_2*cur2_2], 1)) #256 
        fuse_result3 = self.residual3(torch.cat([cross_result3, cur1_3*cur2_3], 1)) #128 
        fuse_result4 = self.residual4(torch.cat([cross_result4, cur1_4*cur2_4], 1)) #64  

        out3 = self.fam32_1(fuse_result3, self.Translayer2_1(fuse_result2))
        out4 = self.fam43_1(fuse_result4, self.Translayer3_1(out3))

        out = self.final(self.upsamplex4(out4))
        return out

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.Translayer2_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.residual2.apply(init_weights)
        self.residual3.apply(init_weights)
        self.residual4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.final.apply(init_weights)