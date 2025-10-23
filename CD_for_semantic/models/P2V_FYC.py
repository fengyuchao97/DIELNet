import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .resnet import resnet18
torch.backends.cudnn.enabled = False

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

def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

class MSGAP(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(MSGAP, self).__init__()
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
    
class Interaction(nn.Module):
    def __init__(self, in_channels, out_channels, num_tokens=1, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        # head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        # self.scale = qk_scale or head_dim ** -0.5
        self.q_ratio = 4
        self.k_ratio = self.q_ratio // 2

        inter_channels = in_channels // 2

        self.init_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.init_conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU()) # conv5_
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.conv_query1 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value1 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2

        self.conv_query2 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value2 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2
        self.msgap = MSGAP(inter_channels, norm_layer=nn.BatchNorm2d)

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        
    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

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

        q_cat = torch.cat([q1,q2],3)

        attn1 = (q_cat @ k1.transpose(-2, -1)) #* self.scale
        attn2 = (q_cat @ k2.transpose(-2, -1)) #* self.scale

        # pos_bias = self._get_relative_positional_bias()
        # attn = (torch.abs(attn1 - attn2) + pos_bias).softmax(dim=-1)
        attn = torch.abs(attn1 - attn2).softmax(dim=-1)

        attn = self.attn_drop(attn)
        x1 = (attn @ v1).transpose(1, 2).reshape(v1.shape[0], ws*ws, -1)
        x2 = (attn @ v2).transpose(1, 2).reshape(v2.shape[0], ws*ws, -1)

        # reverse
        x1 = window_reverse(x1, (H, W), (ws, ws)).permute(0, 2, 1).reshape(B, -1, H, W)
        x2 = window_reverse(x2, (H, W), (ws, ws)).permute(0, 2, 1).reshape(B, -1, H, W)
        x_cat = self.conv_cat(torch.cat([x1,x2],1))
        return x_cat, x1, x2
    
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
        x_cat, x1_interact, x2_interact = self.forward_interaction_local(q1, k1, v1, q2, k2, v2, H, W)
        return x_cat, x1_interact, x2_interact

class Interaction2(nn.Module):
    def __init__(self, in_channels, out_channels, num_tokens=1, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        # head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.q_ratio = 8
        self.k_ratio = self.q_ratio // 2
        # self.scale = qk_scale or head_dim ** -0.5

        inter_channels = in_channels

        self.init_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.init_conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU()) # conv5_
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.conv_query1 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value1 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2

        self.conv_query2 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value2 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2
        self.msgap = MSGAP(inter_channels, norm_layer=nn.BatchNorm2d)

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        
    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

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

        q_cat = torch.cat([q1,q2],3)

        attn1 = (q_cat @ k1.transpose(-2, -1)) #* self.scale
        attn2 = (q_cat @ k2.transpose(-2, -1)) #* self.scale

        # pos_bias = self._get_relative_positional_bias()
        attn = (torch.abs(attn1 - attn2) ).softmax(dim=-1) #+ pos_bias

        attn = self.attn_drop(attn)
        x1 = (attn @ v1).transpose(1, 2).reshape(v1.shape[0], ws*ws, -1)
        x2 = (attn @ v2).transpose(1, 2).reshape(v2.shape[0], ws*ws, -1)

        # reverse
        x1 = window_reverse(x1, (H, W), (ws, ws)).permute(0, 2, 1).reshape(B, -1, H, W)
        x2 = window_reverse(x2, (H, W), (ws, ws)).permute(0, 2, 1).reshape(B, -1, H, W)
        x_cat = self.conv_cat(torch.cat([x1,x2],1))
        return x_cat, x1, x2
    
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
        x_cat, x1_interact, x2_interact = self.forward_interaction_local(q1, k1, v1, q2, k2, v2, H, W)
        return x_cat, x1_interact, x2_interact
    
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
    
class DRAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DRAtt, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = MSGAP(out_channels, norm_layer) # en_s
        self.cpam_dec = CPAMDec(out_channels) # de_s

        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) # conv51_c
        self.ccam_dec = CCAMDec() # de_c

        
    def forward(self, x):
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        cpam_b = self.conv_cpam_b(ccam_feat)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)
        return cpam_feat
    
class ECICNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ECICNet, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = Interaction(256,128,window_size=8)
        self.consrative3 = Interaction(128,64,window_size=16)
        self.consrative4 = Interaction(64,32,window_size=32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.fam32_1 = DRAtt(128,64)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = DRAtt(64,32)

        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.fam32_2 = DRAtt(128,64)
        self.Translayer3_2 = BasicConv2d(64,32,1)
        self.fam43_2 = DRAtt(64,32)

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
        c2 = self.resnet.layer2(c1) 
        c2_img2 = self.resnet.layer2(c1_img2) 
        c3 = self.resnet.layer3(c2) 
        c3_img2 = self.resnet.layer3(c2_img2) 

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

class EDMINet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(EDMINet, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        self.resnet.layer4 = nn.Identity()

        self.cross2 = Interaction2(256, 256 , window_size=8) 
        self.cross3 = Interaction2(128, 128 , window_size=16) 
        self.cross4 = Interaction2(64, 64 , window_size=32) 

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

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))
    
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
        y = F.relu(y+res)
        return y
    
class VideoEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64,128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = 2
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
        # print('VE-x:',x.shape)
        feats = [x]

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            feats.append(x)
            # print('VE-loop-x:',x.shape)
        return feats

class Interaction3(nn.Module):
    def __init__(self, in_channels, num_tokens=1, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        # head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        # self.scale = qk_scale or head_dim ** -0.5
        self.q_ratio = 2
        self.k_ratio = self.q_ratio // 2

        inter_channels = in_channels // 2

        self.init_conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.init_conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU()) # conv5_
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.conv_query1 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value1 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2

        self.conv_query2 = nn.Conv2d(in_channels = inter_channels , out_channels = inter_channels//self.q_ratio, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Conv2d(inter_channels, inter_channels//self.k_ratio, kernel_size= 1) # key_conv2
        self.conv_value2 = nn.Conv2d(inter_channels, inter_channels, kernel_size= 1) # value2

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        
    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

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

        q_cat = torch.cat([q1,q2],3)

        attn1 = (q_cat @ k1.transpose(-2, -1)) #* self.scale
        attn2 = (q_cat @ k2.transpose(-2, -1)) #* self.scale

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
        return x1_interact, x2_interact

class MaxPool2x2(nn.MaxPool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)

# class PairEncoder(nn.Module):
#     def __init__(self, in_ch, enc_chs=(16,32,64), add_chs=(0,0)):
#         super().__init__()

#         self.n_layers = 3
#         self.conv1 = Conv3x3(in_ch, enc_chs[0], norm=True, act=True)
#         self.conv2 = Conv3x3(in_ch, enc_chs[0], norm=True, act=True)

#         self.interact1 = Interaction3(enc_chs[0], enc_chs[0], window_size=16)
#         self.pool1 = MaxPool2x2()

#         self.interact2 = Interaction3(enc_chs[0], enc_chs[1], window_size=16)
#         self.conv_fuse2 = Conv3x3(enc_chs[0]+add_chs[0], enc_chs[1])
#         self.pool2 = MaxPool2x2()

#         self.interact3 = Interaction3(enc_chs[1], enc_chs[2], window_size=16)
#         self.conv_fuse3 = Conv3x3(enc_chs[1]+add_chs[1], enc_chs[2])
#         self.pool3 = MaxPool2x2()

#     def forward(self, x1, x2, add_feats=None):
#         x = torch.cat([x1,x2], dim=1)
#         feats = [x]

#         x1 = self.conv1(x1)
#         x2 = self.conv1(x2)
#         for i in range(self.n_layers):
#             interact = getattr(self, f'interact{i+1}')
#             if i > 0 and add_feats is not None:
#                 add_feat = F.interpolate(add_feats[i-1], size=x.shape[2:])
#                 x = torch.cat([x, add_feat], dim=1)
#                 conv_fuse = getattr(self, f'conv_fuse{i+1}')
#                 x = conv_fuse(x)
#             x, x1, x2 = interact(x1,x2)
#             pool = getattr(self, f'pool{i+1}')
#             x1 = pool(x1)
#             x2 = pool(x2)
#             feats.append(x)

#         return feats

class PairEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(16,32,64), add_chs=(0,0)):
        super().__init__()

        self.n_layers = 3

        self.conv1 = SimpleResBlock(2*in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = SimpleResBlock(enc_chs[0]+add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock(enc_chs[1]+add_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()

    def forward(self, x1, x2, add_feats=None):
        x = torch.cat([x1,x2], dim=1)
        feats = [x]

        for i in range(self.n_layers):
            conv = getattr(self, f'conv{i+1}')
            if i > 0 and add_feats is not None:
                add_feat = F.interpolate(add_feats[i-1], size=x.shape[2:])
                x = torch.cat([x,add_feat], dim=1)
            x = conv(x)
            pool = getattr(self, f'pool{i+1}')
            x = pool(x)
            feats.append(x)

        return feats
    
class PairEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(16,32,64), add_chs=(0,0)):
        super().__init__()

        self.n_layers = 2

        self.solo_conv1 = Conv3x3(in_ch, enc_chs[0], norm=True, act=True)
        self.solo_conv2 = Conv3x3(in_ch, enc_chs[0], norm=True, act=True)

        self.interact = Interaction3(enc_chs[0], window_size=8)
        self.pool = MaxPool2x2()

        self.conv1 = SimpleResBlock(enc_chs[0]+add_chs[0], enc_chs[1])
        self.pool1 = MaxPool2x2()

        self.conv2 = ResBlock(enc_chs[1]+add_chs[1], enc_chs[2])
        self.pool2 = MaxPool2x2()

    def forward(self, x1, x2, add_feats=None):

        x = torch.cat([x1,x2], dim=1)
        feats = [x]

        x1 = self.solo_conv1(x1)
        x2 = self.solo_conv1(x2)
        x1, x2 = self.interact(x1,x2)
        x = torch.cat([x1,x2], dim=1)
        x = self.pool(x)
        feats.append(x)

        for i in range(self.n_layers):
            conv = getattr(self, f'conv{i+1}')
            if add_feats is not None:
                add_feat = F.interpolate(add_feats[i-2], size=x.shape[2:])
                x = torch.cat([x,add_feat], dim=1)
            x = conv(x)
            pool = getattr(self, f'pool{i+1}')
            x = pool(x)
            feats.append(x)

        return feats
    
class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))
    
class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1+in_ch2, out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        return self.conv_fuse(x)
    
class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):
        super().__init__()
        
        enc_chs = enc_chs[::-1]
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs, (itm_ch,)+dec_chs[:-1], dec_chs)
        ])
        self.conv_out = Conv1x1(dec_chs[-1], 1)
    
    def forward(self, x, feats):
        feats = feats[::-1]
        
        x = self.conv_bottom(x)
        
        for feat, blk in zip(feats, self.blocks):
            x = blk(feat, x)

        y = self.conv_out(x)

        return y
    
class P2V_FYC(nn.Module):
    def __init__(self, in_ch=3, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,128), dec_chs=(256,128,64,32)):
        super(P2V_FYC, self).__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.encoder_v = VideoEncoder(in_ch, enc_chs=enc_chs_v)
        enc_chs_v = tuple(ch*self.encoder_v.expansion for ch in enc_chs_v)
        self.encoder_p = PairEncoder(in_ch, enc_chs=enc_chs_p, add_chs=enc_chs_v)
        # self.encoder_p = PairEncoder(in_ch, enc_chs=enc_chs_p)
        self.conv_out_v = Conv1x1(enc_chs_v[-1], 1)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in enc_chs_v
            ]
        )
        self.decoder = SimpleDecoder(enc_chs_p[-1], (2*in_ch,)+enc_chs_p, dec_chs)
    
    def forward(self, t1, t2, return_aux=True):
        frames = self.pair_to_video(t1, t2) # 8, 8, 3, 256, 256

        feats_v = self.encoder_v(frames.transpose(1,2))
        # print('feats_v',feats_v[0].shape) # 8, 3, 8, 256, 256
        feats_v.pop(0)
        
        for i, feat in enumerate(feats_v):
            # print('feat：',feat.shape) # 8， 256， 8， 64， 64    # 8， 512，4，32，32
            # print('self.tem_aggr(feat)',self.tem_aggr(feat).shape)
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        feats_p = self.encoder_p(t1, t2, feats_v)
        # feats_p = self.encoder_p(t1, t2)

        pred = self.decoder(feats_p[-1], feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            pred = torch.sigmoid(pred)
            pred_v = torch.sigmoid(pred_v)
            return pred, pred_v
        else:
            pred = torch.sigmoid(pred)
            return pred

    def pair_to_video(self, im1, im2, rate_map=None):
        # print('imag1:',im1.shape) # 8, 3, 256, 256
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            # print('delta_map:',delta_map,delta_map.shape) # 8, 1, 256, 256
            steps = torch.arange(len+1, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            # print('im1.unsqueeze(1):',im1.unsqueeze(1).shape) # 8, 1, 3, 256, 256
            # print('steps:',steps.shape) # 1, 8, 1, 1, 1
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
            # print('rate_map:',rate_map.shape) # 8, 1, 256, 256
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        # print('frames:',frames.shape) # 8, 8, 3, 256, 256
        return frames

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)
    
if __name__ == '__main__':
    input1 = torch.randn(1, 3, 256, 256).cuda()
    input2 = torch.randn(1, 3, 256, 256).cuda()
    model = P2V_FYC().cuda()
    output = model(input1, input2)