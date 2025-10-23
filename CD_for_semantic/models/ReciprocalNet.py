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



        proj_query = torch.cat([proj_query1,proj_query2],1) #Bx2Nxd
        proj_key = torch.cat([proj_key1,proj_key2],2) #Bxdx2K
        proj_value = torch.cat([proj_value1,proj_value2],2) #BxCx2K

        energy =  torch.bmm(proj_query,proj_key)#Bx2Nx2K
        attention = self.softmax(energy)

        out = torch.bmm(proj_value,attention.permute(0,2,1)).view(m_batchsize,2*C,width,height) #Bx2CxN

        return out

class ReciprocalAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ReciprocalAtt, self).__init__()

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
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)#BKD

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)#BKD

        cpam_feat = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        cpam_feat = self.conv_cat(cpam_feat)
        return cpam_feat

class CPAMDec_Mix0212(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix0212,self).__init__()
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

        attention_Q1K1 = self.softmax(torch.bmm(proj_query1,proj_key1))
        attention_Q1K2 = self.softmax(torch.bmm(proj_query1,proj_key2))
        attention_Q2K1 = self.softmax(torch.bmm(proj_query2,proj_key1))
        attention_Q2K2 = self.softmax(torch.bmm(proj_query2,proj_key2))

        attention1 = torch.abs(attention_Q1K1-attention_Q2K1)
        attention2 = torch.abs(attention_Q2K2-attention_Q1K2)
        out1 = torch.bmm(proj_value1,attention1.permute(0,2,1)).view(m_batchsize,C,width,height)#BxCxN
        out2 = torch.bmm(proj_value2,attention2.permute(0,2,1)).view(m_batchsize,C,width,height)
        out_sub = torch.abs(out1+out2)
        out = torch.cat((out_sub,out1,out2), dim=1)
        return out

class ReciprocalAtt0212(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ReciprocalAtt0212, self).__init__()

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

        self.cpam_dec_mix = CPAMDec_Mix0212(inter_channels) # de_s

        ## Fusion conv
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*3, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv_f
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)#BKD

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)#BKD

        cpam_feat = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        cpam_feat = self.conv_cat(cpam_feat)
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

class ReciprocalNet0212(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet0212, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        self.resnet.layer4 = nn.Identity()

        self.cross2 = ReciprocalAtt0212(256, 128) 
        self.cross3 = ReciprocalAtt0212(128, 64) 
        self.cross4 = ReciprocalAtt0212(64, 32) 

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)

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
        
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
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

        cross_result2 = self.cross2(c3, c3_img2)
        cross_result3 = self.cross3(c2, c2_img2)
        cross_result4 = self.cross4(c1, c1_img2) 

        out2 = self.upsamplex2(self.Translayer2_1(cross_result2))
        out3 = self.fam32_1(torch.cat((cross_result3, out2), dim=1))
        out4 = self.fam43_1(torch.cat((cross_result4, self.Translayer3_1(out3)), dim=1)) 

        out4_up = self.upsamplex2(out4)
        out_1 = self.final(out4_up)
        out_1_2 = self.final_2(self.upsamplex4(out3))
        return out_1, out_1_2

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.final.apply(init_weights)
        self.final_2.apply(init_weights)


class CPAMDec_Mix0213(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix0213,self).__init__()
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

        attention_Q1K1 = self.softmax(torch.bmm(proj_query1,proj_key1))
        attention_Q1K2 = self.softmax(torch.bmm(proj_query1,proj_key2))
        attention_Q2K1 = self.softmax(torch.bmm(proj_query2,proj_key1))
        attention_Q2K2 = self.softmax(torch.bmm(proj_query2,proj_key2))

        attention1 = torch.abs(attention_Q1K1-attention_Q2K1)
        attention2 = torch.abs(attention_Q2K2-attention_Q1K2)
        out1 = torch.bmm(proj_value1,attention1.permute(0,2,1)).view(m_batchsize,C,width,height)#BxCxN
        out2 = torch.bmm(proj_value2,attention2.permute(0,2,1)).view(m_batchsize,C,width,height)
        out_mul = torch.abs(out1*out2)
        out = torch.cat((out_mul,out1,out2), dim=1)
        return out

class ReciprocalNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

        # self.resnet = resnet34()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
        self.resnet.layer4 = nn.Identity()

        self.AFB2 = ReciprocalAtt(256, 256) 
        self.AFB3 = ReciprocalAtt(128, 128) 
        self.AFB4 = ReciprocalAtt(64, 64) 

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.Translayer3_1 = BasicConv2d(64,64,1)

        self.MLFB32 = nn.Sequential(
            BAB_Decoder(128*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.MLFB43 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
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

        cross_result2 = self.AFB2(c3, c3_img2)
        cross_result3 = self.AFB3(c2, c2_img2)
        cross_result4 = self.AFB4(c1, c1_img2) 

        out2 = self.upsamplex2(self.Translayer2_1(cross_result2))
        out3 = self.MLFB32(torch.cat((cross_result3, out2), dim=1))
        out4 = self.MLFB43(torch.cat((cross_result4, self.Translayer3_1(out3)), dim=1))

        out4 = self.final(self.upsamplex2(out4))
        out3 = self.final2(self.upsamplex4(out3))
        
        return out4, out3

    def init_weights(self):
        self.AFB2.apply(init_weights)
        self.AFB3.apply(init_weights)

        self.MLFB32.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.MLFB43.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)

# class ReciprocalNet2(nn.Module):
#     def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
#         super(ReciprocalNet2, self).__init__()

#         self.show_Feature_Maps = show_Feature_Maps
        
#         self.resnet = resnet18()
#         self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

#         # self.resnet = resnet34()
#         # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
#         self.resnet.layer4 = nn.Identity()

#         self.AFB2 = ReciprocalAtt(256, 128) 
#         self.AFB3 = ReciprocalAtt(128, 64) 
#         self.AFB4 = ReciprocalAtt(64, 32) 

#         self.Translayer2_1 = BasicConv2d(128,64,1)
#         self.Translayer3_1 = BasicConv2d(32,32,1)

#         self.MLFB32 = nn.Sequential(
#             BAB_Decoder(64*2, 32, 32, 3, 2),
#             nn.Dropout(0.2),
#             TransBasicConv2d(32, 32, kernel_size=2, stride=2,
#                             padding=0, dilation=1, bias=False)
#         )
#         self.MLFB43 = nn.Sequential(
#             BAB_Decoder(32*2, 32, 32, 3, 2),
#             nn.Dropout(0.2),
#             TransBasicConv2d(32, 32, kernel_size=2, stride=2,
#                             padding=0, dilation=1, bias=False)
#         )

#         self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

#         self.final = nn.Sequential(
#             Conv(32, 32, 3, bn=True, relu=True),
#             Conv(32, num_classes, 3, bn=False, relu=False)
#             )
#         self.final2 = nn.Sequential(
#             Conv(32, 32, 3, bn=True, relu=True),
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

#         cross_result2 = self.AFB2(c3, c3_img2)
#         cross_result3 = self.AFB3(c2, c2_img2)
#         cross_result4 = self.AFB4(c1, c1_img2) 

#         out2 = self.upsamplex2(self.Translayer2_1(cross_result2))
#         out3 = self.MLFB32(torch.cat((cross_result3, out2), dim=1))
#         out4 = self.MLFB43(torch.cat((cross_result4, self.Translayer3_1(out3)), dim=1))

#         out4 = self.final(self.upsamplex2(out4))
#         out3 = self.final2(self.upsamplex4(out3))
        
#         return out4, out3

#     def init_weights(self):
#         self.AFB2.apply(init_weights)
#         self.AFB3.apply(init_weights)

#         self.MLFB32.apply(init_weights)
#         self.Translayer2_1.apply(init_weights)
#         self.MLFB43.apply(init_weights)
#         self.Translayer3_1.apply(init_weights)

#         self.final.apply(init_weights)
#         self.final2.apply(init_weights)

class ReciprocalNet2(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet2, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

        # self.resnet = resnet34()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
        self.resnet.layer4 = nn.Identity()

        self.AFB2 = ReciprocalAtt(256, 128) 
        self.AFB3 = ReciprocalAtt(128, 64) 
        self.AFB4 = ReciprocalAtt(64, 32) 

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)

        self.MLFB32 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.MLFB43 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.pixelshuffle = nn.PixelShuffle(2)

        self.final = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final3 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
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

        cross_result2 = self.AFB2(c3, c3_img2)
        cross_result3 = self.AFB3(c2, c2_img2)
        cross_result4 = self.AFB4(c1, c1_img2) 

        out2 = self.upsamplex2(self.Translayer2_1(cross_result2))
        out3 = self.MLFB32(torch.cat((cross_result3, out2), dim=1))
        out4 = self.MLFB43(torch.cat((cross_result4, self.Translayer3_1(out3)), dim=1))

        out4 = self.final(self.upsamplex2(out4))
        out3 = self.final2(self.upsamplex2(self.pixelshuffle(out3)))
        out2 = self.final3(self.upsamplex4(self.pixelshuffle(out2)))
        
        return out4, out3, out2

    def init_weights(self):
        self.AFB2.apply(init_weights)
        self.AFB3.apply(init_weights)

        self.MLFB32.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.MLFB43.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final3.apply(init_weights)

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

class FEM(nn.Module):
    def __init__(self, inplanes, out_channel, channel_rate=2, reduction_ratio=16):
        super(FEM, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        self.out_channel = out_channel
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channel)
        )

        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channel)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        #
        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.out_channel, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.out_channel*2, self.out_channel, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.out_channel),
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
        # Intra-image channel attention
        E_s = self.ChannelGate(s) * p_s
        E_s = E_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        q = self.Trans_q(q)
        # Intra-image channel attention
        E_q = self.ChannelGate(q) * q_s
        E_q = E_q + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s

class CPAMDec_Share(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Share,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.ChannelGate1 = ChannelGate(in_channels, pool_types=['avg'], reduction_ratio=16)
        self.ChannelGate2 = ChannelGate(in_channels, pool_types=['avg'], reduction_ratio=16)

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels, kernel_size= 1) # query_conv2
        # self.conv_key1 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        # self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels) # key_conv2
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
        # proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        proj_value1 = self.conv_value1(y1).permute(0,2,1) #BxCxK

        # proj_query2  = self.conv_query2(x2).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        proj_value2 = self.conv_value2(y2).permute(0,2,1) #BxCxK

        energy =  torch.bmm(proj_query1,proj_key2)#BxNx2
        attention = self.softmax(energy)

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1)).view(m_batchsize,C,width,height) #BxCxN
        out1 = self.ChannelGate1(x1)*out1 + x1
        out2 = torch.bmm(proj_value2,attention.permute(0,2,1)).view(m_batchsize,C,width,height) #BxCxN
        out2 = self.ChannelGate1(x2)*out2 + x2

        return out1, out2

class ShareAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ShareAtt, self).__init__()

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

        self.cpam_dec_mix = CPAMDec_Share(inter_channels) # de_s

        ## Fusion conv
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv_f
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)#BKD

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)#BKD

        cpam_share1,cpam_share2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        cpam_feat = self.conv_cat(torch.cat((cpam_share1,cpam_share2),dim=1))
        return cpam_feat, cpam_share1, cpam_share2

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

        ## Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) # conv51_c
        self.ccam_dec = CCAMDec() # de_c

        
    def forward(self, x):
        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        ## Compact Spatial Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(ccam_feat)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)

        return cpam_feat

class ReciprocalNet0213(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet0213, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ShareAtt(256,128)
        self.consrative3 = ShareAtt(128,64)
        self.consrative4 = ShareAtt(64,32)

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

class ReciprocalNet0214(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet0214, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM(256,128)
        self.consrative3 = FEM(128,64)
        self.consrative4 = FEM(64,32)

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
        out4 = self.fam43_1(torch.cat([cross_result4, self.Translayer3_1(out3)],1))

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2)],1)) # 4, 64, 64, 64

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


class FEM0215(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0215, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU()
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU()
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)
        
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

class ReciprocalNet0215(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet0215, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0215(256)
        self.consrative3 = FEM0215(128)
        self.consrative4 = FEM0215(64)

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
        
        c2 = self.resnet.layer2(c1) # self.cat_fuse_1_1(torch.cat([c1,cur1_4],1)))
        c2_img2 = self.resnet.layer2(c1_img2) # self.cat_fuse_1_2(torch.cat([c1_img2,cur2_4],1)))

        c3 = self.resnet.layer3(c2) # self.cat_fuse_2_1(torch.cat([c2,cur1_3],1)))
        c3_img2 = self.resnet.layer3(c2_img2) # self.cat_fuse_2_2(torch.cat([c2_img2,cur2_3],1)))

        cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  # 64 32
        cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) # 128 64
        cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) # 256 128

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))
        out4 = self.fam43_1(torch.cat([cross_result4, out3],1)) #self.Translayer3_1(out3)

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), out3_2],1)) # 4, 64, 64, 64 self.Translayer3_2(out3_2)

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


class FEM0216(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0216, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU()
        )

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU()
        )
        
        # self.common_vq = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
        #                  padding=0)
        # self.common_vs = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
        #                  padding=0)

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
        v_q = q.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        v_s = s.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
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

class ReciprocalNet0216(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet0216, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0215(256)
        self.consrative3 = FEM0215(128)
        self.consrative4 = FEM0215(64)

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

class FEM0217(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0217, self).__init__()

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
        
        self.common_vq = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.common_vs = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)

        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_

    def forward(self, q, s):

        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication
        # common feature learning
        v_q = self.common_vq(q).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        v_s = self.common_vs(s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        q_x = self.query(q).view(batch_size, self.inter_channels, -1)

        s = self.Trans_s(s)
        q = self.Trans_q(q)

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

class ReciprocalNet0217(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(ReciprocalNet0217, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0215(256)
        self.consrative3 = FEM0215(128)
        self.consrative4 = FEM0215(64)

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
        # self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        # self.Translayer3_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)
    
class OneNet0221(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(OneNet0221, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0215(256)
        self.consrative3 = FEM0215(128)
        self.consrative4 = FEM0215(64)

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
        # self.final_middle_1 = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )
        # self.final_middle_2 = nn.Sequential(
        #     Conv(64, 32, 3, bn=True, relu=True),
        #     Conv(32, num_classes, 3, bn=False, relu=False)
        #     )

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
        # out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        # out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

        return out_1, out_2 #, out_middle_1, out_middle_2 

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
        # self.final_middle_1.apply(init_weights)
        # self.final_middle_2.apply(init_weights)

class CollectiveAtt0222(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(CollectiveAtt0222, self).__init__()

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
        
        self.common_v = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1, padding=0)
        self.query = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels//2, kernel_size=1, stride=1, padding=0)

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
        v_q = self.common_v(q).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        v_s = self.common_v(s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels//2, -1).permute(0, 2, 1)
        q_x = self.query(q).view(batch_size, self.inter_channels//2, -1)

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
    
class OneNet0222(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(OneNet0222, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = CollectiveAtt0222(256)
        self.consrative3 = CollectiveAtt0222(128)
        self.consrative4 = CollectiveAtt0222(64)

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

class CollectiveAtt0223(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(CollectiveAtt0223, self).__init__()

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
        
        self.common_vq = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.common_vs = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.query = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

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
        v_q = self.common_vq(q).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        v_s = self.common_vs(s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
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
    
class OneNet0223(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(OneNet0223, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = CollectiveAtt0223(256)
        self.consrative3 = CollectiveAtt0223(128)
        self.consrative4 = CollectiveAtt0223(64)

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
        out4 = self.fam43_1(torch.cat([cross_result4, out3],1)) # self.Translayer3_1(

        out2_2 = self.Translayer2_2(torch.abs(cur1_2+cur2_2))
        out3_2 = self.fam32_2(torch.cat([torch.abs(cur1_3+cur2_3), self.upsamplex2(out2_2)],1))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4+cur2_4), out3_2],1)) # self.Translayer3_2(

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