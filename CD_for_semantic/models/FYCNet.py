import torch
import torch.nn as nn
from .resnet import resnet18,resnet34,resnet50,resnet101
import torch.nn.functional as F
import math

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
    
import matplotlib.pyplot as plt
import numpy as np
import cv2
def draw_features(x,savename):
    #tic=time.time()
    fig = plt.figure(figsize=(60,60))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # for i in range(width*height):
    # plt.subplot(height,width, i + 1)
    plt.axis('off')
    # img = x[0, 1, :, :]
    # img =  np.sum(x, axis=1)[1, :, :]
    img =  np.average(x,axis=1)[1, :, :]
    # print(img.shape)
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale

class FEM0222(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0222, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))
        
        self.common_vq = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.common_vs = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)

        self.key = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_

    def forward(self, q, s):

        s = self.Trans_s(s)
        q = self.Trans_q(q)

        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication
        # common feature learning
        v_q = self.common_vq(q).view(batch_size, self.inter_channels, -1)
        v_q = v_q.permute(0, 2, 1)

        v_s = self.common_vs(s).view(batch_size, self.inter_channels, -1)
        v_s = v_s.permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1)
        k_x = k_x.permute(0, 2, 1)

        q_x = self.query(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        # Intra-image channel attention
        E_s = self.ChannelGate(s) * p_s
        E_s = E_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        # Intra-image channel attention
        E_q = self.ChannelGate(q) * q_s
        E_q = E_q + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s

class FYCNet0222(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet0222, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0222(256)
        self.consrative3 = FEM0222(128)
        self.consrative4 = FEM0222(64)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
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

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.Translayer3_1(out3)],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)],1))

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

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

class FYCNet0223(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet0223, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0222(256)
        self.consrative3 = FEM0222(128)
        self.consrative4 = FEM0222(64)
        # self.consrative2 = ContrastiveAtt(256,128)
        # self.consrative3 = ContrastiveAtt(128,64)
        # self.consrative4 = ContrastiveAtt(64,32)

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

class FYCNet0224(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet0224, self).__init__()

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
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
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

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.Translayer3_1(out3)],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)],1))

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

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

class FEM0225(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0225, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        self.scale = nn.Parameter(torch.zeros(1))

        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))
        
        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.ChannelGate1 = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)
        self.ChannelGate2 = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_

    def forward(self, q, s):

        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication
        # common feature learning
        v_q1 = self.common_v(q)
        v_q = v_q1.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        v_s1 = self.common_v(s)
        v_s = v_s1.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1)
        k_x = k_x.permute(0, 2, 1)

        q_x = self.query(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        s = self.Trans_s(s)
        q = self.Trans_q(q)
        # Intra-image channel attention
        E_s = self.ChannelGate1(v_s1) * p_s
        # E_s = self.scale*p_s
        E_s = E_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        # Intra-image channel attention
        E_q = self.ChannelGate2(v_q1) * q_s
        # E_q = self.scale*q_s
        E_q = E_q + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s
    
class FYCNet0226(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet0226, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0225(256)
        self.consrative3 = FEM0225(128)
        self.consrative4 = FEM0225(64)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
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

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.Translayer3_1(out3)],1)) #

        out2_2 = self.Translayer2_2(torch.abs(cur1_2+cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3+cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4+cur2_4), self.Translayer3_2(out3_2)],1)) #

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

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


class FEM0227(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0227, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        self.scale = nn.Parameter(torch.zeros(1))

        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))
        
        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.key1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                             padding=0)
        self.query1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                           padding=0)
        
        self.key2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                             padding=0)
        self.query2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                           padding=0)

        # self.ChannelGate = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_

    def forward(self, q, s):

        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication
        # common feature learning
        v_q = self.common_v(q).view(batch_size, self.inter_channels, -1)
        v_q = v_q.permute(0, 2, 1)

        v_s = self.common_v(s).view(batch_size, self.inter_channels, -1)
        v_s = v_s.permute(0, 2, 1)

        k_x = torch.cat((self.key1(q), self.key2(s)),dim=1).view(batch_size, self.inter_channels, -1) # self.key1(s)
        k_x = k_x.permute(0, 2, 1)

        q_x = torch.cat((self.query1(q), self.query2(s)),dim=1).view(batch_size, self.inter_channels, -1) # self.query1(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        s = self.Trans_s(s)
        q = self.Trans_q(q)
        # Intra-image channel attention
        # E_s = self.ChannelGate(s) * p_s
        E_s = self.scale*p_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        # Intra-image channel attention
        # E_q = self.ChannelGate(q) * q_s
        E_q = self.scale*q_s + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class CrossAtt0227(nn.Module):
    def __init__(self, in_channels, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CrossAtt0227,self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels //2
        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))
        
        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_
        
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1half1 = nn.Linear(d_model, d_model//2)
        self.linear1half2 = nn.Linear(d_model, d_model//2)

        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)
    
    def forward(self, src1, src2):

        batch_size, channels, height_q, width_q = src1.shape
        batch_size, channels, height_s, width_s = src2.shape
        q1 = src1
        k1 = src1
        src12 = self.self_attn1(q1, k1, value=src1)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = src2
        k2 = src2
        src22 = self.self_attn2(q2, k2, value=src2)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)

        cross_q1 = self.linear1half1(src1)
        cross_q2 = self.linear1half2(src2)
        cross_q = torch.cat((cross_q1, cross_q2),dim=1)
        src12 = self.multihead_attn1(query=cross_q,
                                   key=src2,
                                   value=src2)[0]
        src22 = self.multihead_attn2(query=cross_q,
                                   key=src1,
                                   value=src1)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1).view(batch_size, channels, height_q, width_q)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2).view(batch_size, channels, height_s, width_s)

        src = self.conv_cat(torch.cat((src1,src2),dim=1))
    
        return src, src1, src2

class FEM0227_2(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0227_2, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        self.scale = nn.Parameter(torch.zeros(1))

        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))
        
        self.common_v = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.key1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                             padding=0)
        self.query1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                           padding=0)
        
        self.key2 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                             padding=0)
        self.query2 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1,
                           padding=0)

        # self.ChannelGate = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_
    
    def forward(self, q, s):

        s = self.Trans_s(s)
        q = self.Trans_q(q)
        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication
        # common feature learning
        v_q = self.common_v(q).view(batch_size, self.inter_channels, -1)
        v_q = v_q.permute(0, 2, 1)

        v_s = self.common_v(s).view(batch_size, self.inter_channels, -1)
        v_s = v_s.permute(0, 2, 1)

        k_x = torch.cat((self.key1(q), self.key2(s)),dim=1).view(batch_size, self.inter_channels, -1) # self.key1(s) torch.cat((key1, key2, key1, key2),dim=1)
        k_x = k_x.permute(0, 2, 1)

        q_x = torch.cat((self.query1(q), self.query2(s)),dim=1).view(batch_size, self.inter_channels, -1) # self.query1(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        # Intra-image channel attention
        # E_s = self.ChannelGate(s) * p_s
        E_s = self.scale*p_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        # Intra-image channel attention
        # E_q = self.ChannelGate(q) * q_s
        E_q = self.scale*q_s + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s
    
class FYCNet0227(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet0227, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0227_2(256)
        self.consrative3 = FEM0227_2(128)
        self.consrative4 = FEM0227_2(64)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
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

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.Translayer3_1(out3)],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2+cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3+cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4+cur2_4), self.Translayer3_2(out3_2)],1))

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

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

class ScaledDotProductAttention(nn.Module):
   def __init__(self, n_heads, d_model):
       super().__init__()
       self.n_heads = n_heads
       self.d_model = d_model
       self.d_k = d_model // 8
       self.d_v = d_model // 8
    #    self.pos_k = nn.Embedding(self.n_heads * self.len_k, self.d_k // 4)
    #    # self.pos_k = nn.Embedding(opt.n_heads * opt.len_k, opt.d_k)
    #    self.pos_v = nn.Embedding(self.n_heads * self.len_k, self.d_v)
    #    self.pos_ids = torch.LongTensor(list(range(self.n_heads * self.len_k))).view(1, self.n_heads, self.len_k)

   def forward(self, Q, K, V):
    #    K_pos = self.pos_k(self.pos_ids.cuda())
    # #    print(K_pos.shape)
    #    V_pos = self.pos_v(self.pos_ids.cuda())
       scores = torch.matmul(Q, (K).transpose(-1, -2)) / np.sqrt(self.d_k) #+ K_pos
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V) #+ V_pos
       return context, attn

class CrossAtt0228(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.d_k = self.in_channels // 8
        self.d_v = self.in_channels // 8
        BN_MOMENTUM = 0.1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels))
        
        self.query1 = nn.Conv2d(self.inter_channels, self.d_k // 8 * self.n_heads, kernel_size = 1, stride = 1)
        self.key1 = nn.Conv2d(self.inter_channels, self.d_k // 4 * self.n_heads, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(self.inter_channels, self.d_v * self.n_heads, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(self.inter_channels, self.d_k // 8 * self.n_heads, kernel_size = 1, stride = 1)
        self.key2 = nn.Conv2d(self.inter_channels, self.d_k // 4 * self.n_heads, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(self.inter_channels, self.d_v * self.n_heads, kernel_size = 1, stride = 1)

        self.att1 = ScaledDotProductAttention(n_heads=self.n_heads, d_model = self.inter_channels)
        self.att2 = ScaledDotProductAttention(n_heads=self.n_heads, d_model = self.inter_channels)

        self.gamma1 = nn.Parameter(torch.zeros(1)) #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.gamma2 = nn.Parameter(torch.zeros(1)) #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.out_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.out_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU()) # conv_f
        self.W_O_1 = nn.Linear(self.n_heads * self.d_v, self.out_channels)
        self.W_O_2 = nn.Linear(self.n_heads * self.d_v, self.out_channels)
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, input1, input2):
        input1 = self.Trans_s(input1)
        input2 = self.Trans_q(input2)
        batch_size, channels, height, width = input1.shape

        q1 = self.query1(input1).view(batch_size, self.n_heads, -1, height, width)
        k1 = self.key1(input1).view(batch_size, self.n_heads, -1, height * width).permute(0, 1, 3, 2)
        v1 = self.value1(input1).view(batch_size, self.n_heads,-1, height * width).permute(0, 1, 3, 2)

        q2 = self.query2(input2).view(batch_size, self.n_heads, -1, height, width)
        k2 = self.key2(input2).view(batch_size, self.n_heads, -1, height * width).permute(0, 1, 3, 2)
        v2 = self.value2(input2).view(batch_size, self.n_heads, -1, height * width).permute(0, 1, 3, 2)

        q = torch.cat([q1, q2], 2).view(batch_size, self.n_heads, -1, height * width).permute(0, 1, 3, 2)

        context1, attn1 = self.att1(q, k1, v1)
        out1 = context1.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        out1 = self.W_O_1(out1)
        out1 = out1.view(batch_size, self.out_channels, height, width)
        out1 = self.gamma1 * out1 + input1
        out1 = self.norm(out1.view(batch_size, height, width, self.out_channels)).view(batch_size, self.out_channels, height, width)

        context2, attn2 = self.att2(q, k2, v2)
        out2 = context2.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        out2 = self.W_O_2(out2).view(batch_size, self.out_channels, height, width)
        out2 = self.gamma2 * out2 + input2
        out2 = self.norm(out2.view(batch_size, height, width, self.out_channels)).view(batch_size, self.out_channels, height, width)

        feat_sum = self.conv_cat(torch.cat([out1, out2], 1))
        return feat_sum, out1, out2
    
class FYCNet0228(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet0228, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = CrossAtt0228(256,128)
        self.consrative3 = CrossAtt0228(128,64)
        self.consrative4 = CrossAtt0228(64,32) # 64,64,64

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
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

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, self.Translayer3_1(out3)],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2+cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3+cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4+cur2_4), self.Translayer3_2(out3_2)],1))

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

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

class FEM_best(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM_best, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU()
        )
        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU()
        )
        
        self.common_v = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.key1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1, padding=0)
        self.query1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1, padding=0)

        self.key2 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1, padding=0)
        self.query2 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_

    def forward(self, q, s):

        s = self.Trans_s(s)
        q = self.Trans_q(q)

        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication
        # common feature learning
        v_q1 = self.common_v(q)
        v_q = v_q1.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        v_s1 = self.common_v(s)
        v_s = v_s1.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        k_x1 = self.key1(s).view(batch_size, self.inter_channels//2, -1)
        k_x2 = self.key2(q).view(batch_size, self.inter_channels//2, -1)
        key_cat = torch.cat([k_x1, k_x2*(-1)], 1).permute(0, 2, 1)

        q_x1 = self.query1(s).view(batch_size, self.inter_channels//2, -1)
        q_x2 = self.query2(q).view(batch_size, self.inter_channels//2, -1)
        query_cat = torch.cat([q_x1, q_x2], 1)

        A_s = torch.abs(torch.matmul(key_cat, query_cat))
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        # Intra-image channel attention
        E_s = self.ChannelGate(v_s1) * p_s
        E_s = E_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        # Intra-image channel attention
        E_q = self.ChannelGate(v_q1) * q_s
        E_q = E_q + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s
    
class FYCNet_best(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(FYCNet_best, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM_best(256)
        self.consrative3 = FEM_best(128)
        self.consrative4 = FEM_best(64)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        # self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        # self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
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

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, out3],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2+cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3+cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4+cur2_4), out3_2],1))

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        # self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        # self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)