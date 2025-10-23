import torch
import torch.nn as nn
from .resnet import resnet18,resnet34
from .ViTAE_Window_NoShift.base_model import ViTAE_Window_NoShift_basic
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

        fuse = self.residual(torch.cat([cur1, cur2], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse),cur1,cur2
        else:
            return fuse,cur1,cur2

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
        high_stage = self.bilinear_interpolate_torch_gridsample2(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample2(low_stage, (h, w), delta2)

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

class auge(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(auge, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge): 
        # edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out
        
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )
    
class BiFusion_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFusion_block, self).__init__()

        self.ca = ChannelAttention(in_channels=in_channels)
        self.bn_ca = nn.BatchNorm2d(in_channels)
        self.conv1 = conv2d_bn(in_channels, out_channels)
        self.conv2 = conv2d_bn(out_channels, out_channels)
        self.sa = SpatialAttention()
        self.bn_sa = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        #optional to use channel attention module in the first combined feature
        x = self.ca(x) * x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sa(x) * x
        x = self.bn_sa(x)
        return x
            
class MixAttNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet, self).__init__()

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

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        
        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.fusion_early = BiFusion_block(in_channels=128, out_channels=64)
        self.fusion_later = conv2d_bn(in_channels=128, out_channels=64)
        self.auge = auge(num_in=64,plane_mid=16, mids=4)

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
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)) # 4, 64, 64, 64

        fusion_early = self.fusion_early(torch.cat([c0,c0_img2],1)) # 4, 64, 128, 128
        fusion_later = self.fusion_later(torch.cat([out4,out4_2],1))  

        feature = self.auge(self.upsamplex2(fusion_later), fusion_early)
        output = self.final(self.upsamplex2(feature))

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return output, out_1, out_2, out_middle_1, out_middle_2

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

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        self.final.apply(init_weights)
        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class MixAttNet2(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet2, self).__init__()

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
        self.Translayer2_1 = conv2d_bn(256,128)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128) # AlignBlock(128) # 
        self.Translayer3_1 = conv2d_bn(128,64)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64) # AlignBlock(64) # 

        self.Translayer2_2 = conv2d_bn(256,128)
        self.fam32_2 = AlignBlock(128) # AlignBlock(128) # 
        self.Translayer3_2 = conv2d_bn(128,64)
        self.fam43_2 = AlignBlock(64) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        # self.final_middle_1 = nn.Sequential(
        #     Conv(128, 64, 3, bn=True, relu=True),
        #     Conv(64, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final_middle_2 = nn.Sequential(
        #     Conv(128, 64, 3, bn=True, relu=True),
        #     Conv(64, num_classes, 3, bn=False, relu=False)
        #     )

        self.fusion_early = BiFusion_block(in_channels=128, out_channels=128)
        self.fusion_later = conv2d_bn(in_channels=128, out_channels=128)
        self.auge = auge(num_in=128,plane_mid=64, mids=8)

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
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)) # 4, 64, 64, 64

        fusion_early = self.fusion_early(torch.cat([c0,c0_img2],1)) # 4, 64, 128, 128
        fusion_later = self.fusion_later(torch.cat([out4,out4_2],1))  

        feature = self.auge(self.upsamplex2(fusion_later), fusion_early)
        output = self.final(self.upsamplex2(feature))

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        # out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        # out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return output, out_1, out_2 #, out_middle_1, out_middle_2

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

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        self.final.apply(init_weights)
        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        # self.final_middle_1.apply(init_weights)
        # self.final_middle_2.apply(init_weights)

class MixAttNet3(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet3, self).__init__()

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

        self.cat_fuse_1_1 = BasicConv2d(128, 64,1)
        self.cat_fuse_2_1 = BasicConv2d(256, 128,1)

        self.cat_fuse_1_2 = BasicConv2d(128, 64,1)
        self.cat_fuse_2_2 = BasicConv2d(256, 128,1)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)
        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = AlignBlock(128) # decode(128,128,128) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = AlignBlock(64) # decode(64,64,64) # AlignBlock(64) # 

        # self.Translayer2_2 = BasicConv2d(256,128,1)
        # self.fam32_2 = AlignBlock(128) # AlignBlock(128) # 
        # self.Translayer3_2 = BasicConv2d(128,64,1)
        # self.fam43_2 = AlignBlock(64) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        # self.final_middle_1 = nn.Sequential(
        #     Conv(128, 64, 3, bn=True, relu=True),
        #     Conv(64, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final_middle_2 = nn.Sequential(
        #     Conv(128, 64, 3, bn=True, relu=True),
        #     Conv(64, num_classes, 3, bn=False, relu=False)
        #     )

        self.fusion_early = BiFusion_block(in_channels=128, out_channels=64)
        self.auge = auge(num_in=64,plane_mid=64, mids=8) # (num_in=64,plane_mid=16, mids=4)

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2)  # 64

        c2 = self.resnet.layer2(self.cat_fuse_1_1(torch.cat([c1,cur2_4],1)))
        c2_img2 = self.resnet.layer2(self.cat_fuse_1_2(torch.cat([c1_img2,cur1_4],1)))
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2) # 128
        
        c3 = self.resnet.layer3(self.cat_fuse_2_1(torch.cat([c2,cur2_3],1)))
        c3_img2 = self.resnet.layer3(self.cat_fuse_2_2(torch.cat([c2_img2,cur1_3],1)))
        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2) # 256

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        # out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        # out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)) # 4, 64, 64, 64

        fusion_early = self.fusion_early(torch.cat([c0,c0_img2],1)) # 4, 64, 128, 128
        # # fusion_later = self.fusion_later(torch.cat([out4,out4_2],1))  

        feature = self.auge(self.upsamplex2(out4), fusion_early)

        output = self.final(self.upsamplex2(feature))
        out_1 = self.final1(self.upsamplex4(out4))
        edge = self.final2(self.upsamplex4(fusion_early))
        # out_2 = self.final2(self.upsamplex4(out4_2))
        # out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        # out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return output, out_1, edge #, out_2, out_middle_1, out_middle_2

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

        # self.fam32_2.apply(init_weights)
        # self.Translayer2_2.apply(init_weights)
        # self.fam43_2.apply(init_weights)
        # self.Translayer3_2.apply(init_weights)
        self.final.apply(init_weights)
        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        # self.final_middle_1.apply(init_weights)
        # self.final_middle_2.apply(init_weights)

        self.cat_fuse_1_1.apply(init_weights)
        self.cat_fuse_2_1.apply(init_weights)
        self.cat_fuse_1_2.apply(init_weights)
        self.cat_fuse_2_2.apply(init_weights)

        self.fusion_early.apply(init_weights)
        self.auge.apply(init_weights)


class CPAMEnc(nn.Module):
    """
    CPAM encoding module
    """
    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

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

    def forward(self, x):
        b, c, h, w = x.size()
        
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4), 2)

class CPAMDec(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value = nn.Linear(in_channels, in_channels) # value2
    def forward(self, x,y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize,C,width ,height = x.size()
        m_batchsize,K,M = y.size()

        proj_query  = self.conv_query(x).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key =  self.conv_key(y).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        energy =  torch.bmm(proj_query,proj_key)#BxNxK
        attention = self.softmax(energy) #BxNxk

        proj_value = self.conv_value(y).permute(0,2,1) #BxCxK
        out = torch.bmm(proj_value,attention.permute(0,2,1))#BxCxN
        out = out.view(m_batchsize,C,width,height)
        out = self.scale*out + x
        return out

class CCAMDec(nn.Module):
    """
    CCAM decoding module
    """
    def __init__(self):
        super(CCAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x,y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize,C,width ,height = x.size()
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape #BXC1XN
        proj_key  = y_reshape.permute(0,2,1) #BX(N)XC
        energy =  torch.bmm(proj_query,proj_key) #BXC1XC
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) #BCN
        
        out = torch.bmm(attention,proj_value) #BC1N
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out

class CPAMDec_Mix(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2


    def forward(self,x1,y1,x2,y2):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        proj_value1 = self.conv_value1(y1).permute(0,2,1) #BxCxK

        proj_query2  = self.conv_query2(x2).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        proj_value2 = self.conv_value2(y2).permute(0,2,1) #BxCxK

        energy1 =  torch.bmm(proj_query1,proj_key1)#BxNxK
        energy2 =  torch.bmm(proj_query2,proj_key2)#BxNxK

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) #BxNxk

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))#BxCxN
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 # self.scale*

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))#BxCxN
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 # self.scale*
        return out1, out2

class ContrastiveAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt, self).__init__()

        inter_channels = in_channels // 2

        ## Convs or modules for CPAM 
        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix(inter_channels) # de_s

        ## Fusion conv
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv_f
        
    def forward(self, x, y):
        ## Compact Spatial Attention Module(CPAM)
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)#BKD

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)#BKD

        cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        ## Fuse two modules
        # cpam_feat1 = self.conv_cpam_e1(cpam_feat1)
        # cpam_feat2 = self.conv_cpam_e2(cpam_feat2)

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2

class DRAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DRAtt, self).__init__()

        inter_channels = in_channels // 2

        ## Convs or modules for CPAM 
        self.conv_cpam_b = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = CPAMEnc(out_channels, norm_layer) # en_s
        self.cpam_dec = CPAMDec(out_channels) # de_s
        # self.conv_cpam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                    norm_layer(inter_channels),
        #                    nn.ReLU()) # conv52

        ## Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) # conv51_c
        self.ccam_dec = CCAMDec() # de_c
        # self.conv_ccam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) # conv51

        ## Fusion conv
        # self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
        #                            norm_layer(out_channels),
        #                            nn.ReLU()) # conv_f
        
    def forward(self, x):
        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        ## Compact Spatial Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(ccam_feat)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)

        ## Fuse two modules
        # ccam_feat = self.conv_ccam_e(ccam_feat)
        # cpam_feat = self.conv_cpam_e(cpam_feat)
        # feat_sum = self.conv_cat(torch.cat([cpam_feat,ccam_feat],1))
        return cpam_feat

class MixAttNet4(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet4, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ContrastiveAtt(256,128)
        self.consrative3 = ContrastiveAtt(128,64)
        self.consrative4 = ContrastiveAtt(64,32)
        # self.cross1 = MultiHeadCrossAttention(512, 512, ch_out=512, drop_rate=drop_rate/2, qkv_bias=True)
        # self.cross2 = MultiHeadCrossAttention(256, 256, ch_out=256, drop_rate=drop_rate/2, qkv_bias=True)
        # self.cross3 = MultiHeadCrossAttention(128, 128, ch_out=128, drop_rate=drop_rate/2, qkv_bias=True)
        # self.cross4 = MultiHeadCrossAttention(64, 64, ch_out=64, drop_rate=drop_rate/2, qkv_bias=True)

        # self.cat_fuse_1_1 = BasicConv2d(96,64)
        # self.cat_fuse_2_1 = BasicConv2d(192,128)

        # self.cat_fuse_1_2 = BasicConv2d(96,64)
        # self.cat_fuse_2_2 = BasicConv2d(192,128)

        # self.Translayer1_g = BasicConv2d(512,256, 1)
        # self.fam21 = AlignBlock(256) # decode(320,320,320)
        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = decode(32,32,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        # self.final = nn.Sequential(
        #     Conv(64, 64, 3, bn=True, relu=True),
        #     Conv(64, num_classes, 3, bn=False, relu=False)
        #     )

        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        # self.fusion_early = DRAtt(in_channels=128, out_channels=32) # BiFusion_block
        # self.auge = auge(num_in=64, plane_mid=16, mids=4) # (num_in=64,plane_mid=16, mids=4)

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)) # 4, 64, 64, 64

        # fusion_early = self.fusion_early(torch.cat([c0,c0_img2],1)) # 4, 64, 128, 128
        # # # fusion_later = self.fusion_later(torch.cat([out4,out4_2],1))  
        # feature = self.auge(self.upsamplex2(out4), fusion_early)

        # output = self.final(self.upsamplex2(feature))
        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        # edge = self.final2(self.upsamplex4(fusion_early))
        # out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2 #, edge #, out_2, out_middle_1, out_middle_2

    def init_weights(self):
        # self.cross1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        # self.fam21_1.apply(init_weights)
        # self.Translayer1_1.apply(init_weights)
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)
        # self.final.apply(init_weights)
        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

        # self.cat_fuse_1_1.apply(init_weights)
        # self.cat_fuse_2_1.apply(init_weights)
        # self.cat_fuse_1_2.apply(init_weights)
        # self.cat_fuse_2_2.apply(init_weights)

        # self.fusion_early.apply(init_weights)
        # self.auge.apply(init_weights)

import matplotlib.pyplot as plt
import cv2
def draw_features(width,height,x,savename):
    #tic=time.time()
    fig = plt.figure(figsize=(60,60))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    #print("time:{}".format(time.time()-tic))

class MixAttNet5(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet5, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ContrastiveAtt(256,128)
        self.consrative3 = ContrastiveAtt(128,64)
        self.consrative4 = ContrastiveAtt(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt(128,64) #decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt(64,32) #decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt(128,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt(64,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        # if self.show_Feature_Maps:
        #     savepath=r'temp'
            
            # draw_features(4,8,(F.interpolate(out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/Out_fuse_2_img1.png".format(savepath))
            # draw_features(4,8,(F.interpolate(out2_2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/Out_fuse_2_img2.png".format(savepath))
            
            # draw_features(1,2,(out_1).cpu().detach().numpy(),"{}/Final_out_1-2.png".format(savepath))
            # draw_features(1,2,(out_2).cpu().detach().numpy(),"{}/Final_out_2-2.png".format(savepath))
            # draw_features(1,2,(out_middle_1).cpu().detach().numpy(),"{}/Final_out_middle_1-2.png.png".format(savepath))
            # draw_features(1,2,(out_middle_2).cpu().detach().numpy(),"{}/Final_out_middle_2-2.png.png".format(savepath))
        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class SelfAttention(nn.Module):

    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()
        
        f = self.f(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height) # B * C * (W * H)
        
        attention = torch.bmm(f.permute(0, 2, 1), g) # B * (W * H) * (W * H)
        attention = self.softmax(attention)
        
        self_attetion = torch.bmm(h, attention) # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height) # B * C * W * H
        
        out = self.gamma * self_attetion + x
        return out

class SelfA_Module(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(SelfA_Module, self).__init__()

        inter_channels = in_channels // 2

        ## Convs or modules for CPAM 
        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.SelfA1 = SelfAttention(inter_channels)
        self.SelfA2 = SelfAttention(inter_channels)
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv_f
        
    def forward(self, x, y):
        ## Compact Spatial Attention Module(CPAM)
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_feat1 = self.SelfA1(cpam_b_x)
        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_feat2 = self.SelfA2(cpam_b_y)

        ## Fuse two modules
        # cpam_feat1 = self.conv_cpam_e1(cpam_feat1)
        # cpam_feat2 = self.conv_cpam_e2(cpam_feat2)

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2

class CICNet_Self(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CICNet_Self, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = SelfA_Module(256,128)
        self.consrative3 = SelfA_Module(128,64)
        self.consrative4 = SelfA_Module(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt(128,64) #decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt(64,32) #decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt(128,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt(64,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        # if self.show_Feature_Maps:
        #     savepath=r'temp'
            
            # draw_features(4,8,(F.interpolate(out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/Out_fuse_2_img1.png".format(savepath))
            # draw_features(4,8,(F.interpolate(out2_2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/Out_fuse_2_img2.png".format(savepath))
            
            # draw_features(1,2,(out_1).cpu().detach().numpy(),"{}/Final_out_1-2.png".format(savepath))
            # draw_features(1,2,(out_2).cpu().detach().numpy(),"{}/Final_out_2-2.png".format(savepath))
            # draw_features(1,2,(out_middle_1).cpu().detach().numpy(),"{}/Final_out_middle_1-2.png.png".format(savepath))
            # draw_features(1,2,(out_middle_2).cpu().detach().numpy(),"{}/Final_out_middle_2-2.png.png".format(savepath))
        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CrossAttention(nn.Module):

    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim, activation=F.relu):
        super(CrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)

        self.f_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        init_conv(self.f_x)
        init_conv(self.g_x)
        init_conv(self.h_x)

        init_conv(self.f_y)
        init_conv(self.g_y)
        init_conv(self.h_y)
        
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()
        
        f_x = self.f_x(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        g_x = self.g_x(x).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h_x = self.h_x(x).view(m_batchsize, -1, width * height) # B * C * (W * H)

        f_y = self.f_y(y).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        g_y = self.g_y(y).view(m_batchsize, -1, width * height) # B * (C//8) * (W * H)
        h_y = self.h_y(y).view(m_batchsize, -1, width * height) # B * C * (W * H)
        

        attention_x = torch.bmm(f_y.permute(0, 2, 1), g_x) # B * (W * H) * (W * H)
        attention_x = self.softmax(attention_x)

        attention_y = torch.bmm(f_x.permute(0, 2, 1), g_y) # B * (W * H) * (W * H)
        attention_y = self.softmax(attention_y)
        
        self_attetion_x = torch.bmm(h_x, attention_x) # B * C * (W * H)
        self_attetion_x = self_attetion_x.view(m_batchsize, C, width, height) # B * C * W * H
        
        out_x = self.gamma * self_attetion_x + x

        self_attetion_y = torch.bmm(h_y, attention_y) # B * C * (W * H)
        self_attetion_y = self_attetion_y.view(m_batchsize, C, width, height) # B * C * W * H
        
        out_y = self.gamma * self_attetion_y + y
        return out_x, out_y

class CrossA_Module(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(CrossA_Module, self).__init__()

        inter_channels = in_channels // 2

        ## Convs or modules for CPAM 
        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
                                   
        self.CrossA = CrossAttention(inter_channels)
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv_f
        
    def forward(self, x, y):
        ## Compact Spatial Attention Module(CPAM)
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_feat1, cpam_feat2 = self.CrossA(cpam_b_x, cpam_b_y)

        ## Fuse two modules
        # cpam_feat1 = self.conv_cpam_e1(cpam_feat1)
        # cpam_feat2 = self.conv_cpam_e2(cpam_feat2)

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2

class CICNet_Cross(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CICNet_Cross, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = CrossA_Module(256,128)
        self.consrative3 = CrossA_Module(128,64)
        self.consrative4 = CrossA_Module(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt(128,64) #decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt(64,32) #decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt(128,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt(64,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        # if self.show_Feature_Maps:
        #     savepath=r'temp'
            
            # draw_features(4,8,(F.interpolate(out2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/Out_fuse_2_img1.png".format(savepath))
            # draw_features(4,8,(F.interpolate(out2_2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),"{}/Out_fuse_2_img2.png".format(savepath))
            
            # draw_features(1,2,(out_1).cpu().detach().numpy(),"{}/Final_out_1-2.png".format(savepath))
            # draw_features(1,2,(out_2).cpu().detach().numpy(),"{}/Final_out_2-2.png".format(savepath))
            # draw_features(1,2,(out_middle_1).cpu().detach().numpy(),"{}/Final_out_middle_1-2.png.png".format(savepath))
            # draw_features(1,2,(out_middle_2).cpu().detach().numpy(),"{}/Final_out_middle_2-2.png.png".format(savepath))
        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class MixAttNet34(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet34, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet34()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ContrastiveAtt(256,128)
        self.consrative3 = ContrastiveAtt(128,64)
        self.consrative4 = ContrastiveAtt(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt(128,64) #decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt(64,32) #decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt(128,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt(64,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class DRAtt6(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DRAtt6, self).__init__()

        inter_channels = in_channels // 2

        ## Convs or modules for CPAM 
        # self.conv_cpam_b = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
        #                            norm_layer(out_channels),
        #                            nn.ReLU()) # conv5_s
        # self.cpam_enc = CPAMEnc(out_channels, norm_layer) # en_s
        # self.cpam_dec = CPAMDec(out_channels) # de_s
        # self.conv_cpam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                    norm_layer(inter_channels),
        #                    nn.ReLU()) # conv52

        ## Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(out_channels, out_channels//16, 1, bias=False),
                                   norm_layer(out_channels//16),
                                   nn.ReLU()) # conv51_c
        self.ccam_dec = CCAMDec() # de_c
        # self.conv_ccam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) # conv51

        ## Fusion conv
        # self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
        #                            norm_layer(out_channels),
        #                            nn.ReLU()) # conv_f
        
    def forward(self, x):
        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        ## Compact Spatial Attention Module(CPAM)
        # cpam_b = self.conv_cpam_b(ccam_feat)
        # cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        # cpam_feat = self.cpam_dec(cpam_b,cpam_f)

        ## Fuse two modules
        # ccam_feat = self.conv_ccam_e(ccam_feat)
        # cpam_feat = self.conv_cpam_e(cpam_feat)
        # feat_sum = self.conv_cat(torch.cat([cpam_feat,ccam_feat],1))
        return ccam_feat

class MixAttNet8(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(MixAttNet8, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ContrastiveAtt(256,128)
        self.consrative3 = ContrastiveAtt(128,64)
        self.consrative4 = ContrastiveAtt(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt6(128,64) #decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt6(64,32) #decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt6(128,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt6(64,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)

        c1 = self.resnet.layer1(c1)
        c1_img2 = self.resnet.layer1(c1_img2)
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CICNet_VITAE2(nn.Module):
    def __init__(self, args, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(CICNet_VITAE2, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.backbone = ViTAE_Window_NoShift_basic(args,
                        RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
                        NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
                        stages=4, 
                        embed_dims=[64, 64, 128, 256], 
                        token_dims=[64, 128, 256, 512], 
                        downsample_ratios=[4, 2, 2, 2],
                        NC_depth=[2, 2, 8, 2], 
                        NC_heads=[1, 2, 4, 8], 
                        RC_heads=[1, 1, 2, 4], 
                        mlp_ratio=4., 
                        NC_group=[1, 32, 64, 128], 
                        RC_group=[1, 16, 32, 64],
                        img_size=1024, 
                        window_size=7,
                        drop_path_rate=0.3,
                        frozen_stages=-1,
                        norm_eval=False
                        )

        filters0 = [64, 128, 256, 512]

        self.consrative2 = ContrastiveAtt(256,128)
        self.consrative3 = ContrastiveAtt(128,64)
        self.consrative4 = ContrastiveAtt(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt(128,64) #decode(64,64,64) # AlignBlock(128) # 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt(64,32) #decode(32,32,32) # AlignBlock(64) # 

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt(128,64) # AlignBlock(128) # 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt(64,32) # AlignBlock(64) # 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        xA = self.backbone(imgs1)
        xB = self.backbone(imgs2)

        c1, c2, c3, c4 = xA
        c1_img2, c2_img2, c3_img2, c4_img2 = xB

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(self.Translayer3_1(out3))],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.upsamplex2(self.Translayer3_2(out3_2))],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex8(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex8(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)