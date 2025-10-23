import torch
import torch.nn as nn
from .resnet import resnet18,resnet34
import torch.nn.functional as F
import numpy as np
import math
from torch import nn, einsum
from einops import rearrange

def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

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

class CPAMEnc(nn.Module):
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
    def __init__(self,in_channels):
        super(CPAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) 
        self.conv_key = nn.Linear(in_channels, in_channels//4) 
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
    
class CCAM(nn.Module):
    def __init__(self, in_channels=128, norm_layer=nn.BatchNorm2d):
        super(CCAM,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.ccam_enc_x = MGAP(in_channels, norm_layer)
        self.ccam_enc_y = MGAP(in_channels//16, norm_layer)
        # self.dw_conv = DWConv(in_channels)

    def forward(self, x, y):
        m_batchsize,C,width ,height = x.size()
        x_AVGPooling = self.ccam_enc_x(x)
        x_reshape =x_AVGPooling.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_AVGPooling = self.ccam_enc_y(y)
        y_reshape =y_AVGPooling.view(B,K,-1)
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
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_query2  = self.conv_query2(x2).view(m_batchsize,-1,width*height).permute(0,2,1)
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
        return out1, out2

class ContrastiveAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2

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

class CPAMDec_Mix2(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix2,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2
        self.CE = torch.nn.BCEWithLogitsLoss() #size_average=True, reduce=True
    
    def forward(self,x1,y1,x2,y2):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1)
        proj_query2  = self.conv_query2(x2)
        proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)
        
        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query,proj_key1)#.view(-1, width*height*K)
        energy2 =  torch.bmm(proj_query,proj_key2)#.view(-1, width*height*K)

        # loss = torch.abs(cos_sim(energy1, energy2)).sum()
        # energy1_norm = self.softmax(energy1)
        # energy2_norm = self.softmax(energy2)
        # energy1 = energy1.view(-1, width*height*K)
        # energy2 = energy2.view(-1, width*height*K)
        loss = self.CE(self.softmax(energy1),self.softmax(energy2))
        # bi_di_kld = torch.mean(self.kl_divergence(energy1_tanh, energy2_tanh)) + torch.mean(
        #     self.kl_divergence(energy2_tanh, energy1_tanh))

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1)) 
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2
        return out1, out2, loss
    
class ContrastiveAtt2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt2, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix2(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2, loss = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2, loss

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
    
class TCSVTNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TCSVTNet, self).__init__()
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # self.resnet = resnet34()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))

        self.consrative2 = ContrastiveAtt2(256,128)
        self.consrative3 = ContrastiveAtt2(128,64)
        self.consrative4 = ContrastiveAtt2(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            # DRAtt(64*2,64),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            # DRAtt(32*2,32),
            nn.Dropout(0.2),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            # DRAtt(64*2,64),
            nn.Dropout(0.2),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            # DRAtt(32*2,32),
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
        c2 = self.resnet.layer2(c1) 
        c2_img2 = self.resnet.layer2(c1_img2) 
        c3 = self.resnet.layer3(c2) 
        c3_img2 = self.resnet.layer3(c2_img2) 

        cross_result4, cur1_4, cur2_4, loss4 = self.consrative4(c1, c1_img2)  
        cross_result3, cur1_3, cur2_3, loss3 = self.consrative3(c2, c2_img2) 
        cross_result2, cur1_2, cur2_2, loss2 = self.consrative2(c3, c3_img2) 

        attention_loss_all = loss2 + loss3 + loss4

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

        return out_1, out_2, out_middle_1, out_middle_2, attention_loss_all

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

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

from .HolisticAttention import HA
from .batchrenorm import BatchRenorm2d
cos_sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)

class Saliency_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 64)
        self.conv2 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 128)
        self.conv3 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 256)
        self.conv4 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 512)

        self.spatial_axes = [2, 3]

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2*channel)
        self.conv432 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 3*channel)
        self.conv4321 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 4*channel)

        self.br1 = BatchRenorm2d(channel)
        self.br2 = BatchRenorm2d(channel)
        self.br3 = BatchRenorm2d(channel)
        self.br4 = BatchRenorm2d(channel)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1,x2,x3,x4):
        conv1_feat = self.br1(self.conv1(x1))
        conv2_feat = self.br2(self.conv2(x2))
        conv3_feat = self.br3(self.conv3(x3))
        conv4_feat = self.br4(self.conv4(x4))

        conv4_feat = self.upsample2(conv4_feat)
        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)
        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)
        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)
        conv4321 = self.conv4321(conv4321)
        sal_init = self.layer6(conv4321)
        return sal_init
    
class TCSVTNet0302(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TCSVTNet0302, self).__init__()
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # self.resnet = resnet34()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ContrastiveAtt2(256,128)
        self.consrative3 = ContrastiveAtt2(128,64)
        self.consrative4 = ContrastiveAtt2(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        # self.fam32_1 = DRAtt(128,64) 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        # self.fam43_1 = DRAtt(64,32)

        self.Translayer2_2 = BasicConv2d(128,64,1)
        # self.fam32_2 = DRAtt(128,64) 
        self.Translayer3_2 = BasicConv2d(64,32,1)
        # self.fam43_2 = DRAtt(64,32) 

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

        self.sal_decoder1 = Saliency_feat_decoder(2)
        self.sal_decoder2 = Saliency_feat_decoder(2)

        self.HA = HA()
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
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

        x1_input1 = self.resnet.layer1(c1) # 64, 64, 64
        x1_input2 = self.resnet.layer1(c1_img2)
        
        x2_input1 = self.resnet.layer2(x1_input1) # 128, 32, 32
        x2_input2 = self.resnet.layer2(x1_input2) 

        x3_input1 = self.resnet.layer3(x2_input1) # 256, 16, 16
        x3_input2 = self.resnet.layer3(x2_input2) 

        x4_input1 = self.resnet.layer4(x3_input1) # 512, 8, 8
        x4_input2 = self.resnet.layer4(x3_input2) 

        sal_init_input1 = self.sal_decoder1(x1_input1, x2_input1, x3_input1, x4_input1)
        x2_2_input1 = self.HA(self.upsample05(sal_init_input1).sigmoid(), x2_input1)
        x3_2_input1 = self.HA(sal_init_input1.sigmoid(), x3_input1)
        # x4_2_input1 = self.resnet.layer4_2(x3_2_input1)  
        # sal_ref_input1 = self.sal_decoder2(x1_input1, x2_2_input1, x3_2_input1, x4_2_input1)

        sal_init_input2 = self.sal_decoder2(x1_input2, x2_input2, x3_input2, x4_input2)
        x2_2_input2 = self.HA(self.upsample05(sal_init_input2).sigmoid(), x2_input2)
        x3_2_input2 = self.HA(sal_init_input2.sigmoid(), x3_input2)

        cross_result4, cur1_4, cur2_4, loss4 = self.consrative4(c1, c1_img2)  
        cross_result3, cur1_3, cur2_3, loss3 = self.consrative3(x2_2_input1, x2_2_input2) 
        cross_result2, cur1_2, cur2_2, loss2 = self.consrative2(x3_2_input1, x3_2_input2) 

        attention_loss_all = loss2 + loss3 + loss4

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

        return out_1, out_2, out_middle_1, out_middle_2, attention_loss_all

    def init_weights(self):
        self.sal_decoder1.apply(init_weights)
        self.sal_decoder2.apply(init_weights)

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

class MultiFusion(nn.Module):
    def __init__(self, in_channels_Q, in_channels_K, in_channels_V, channel_rate=2):
        super(MultiFusion, self).__init__()

        self.in_channels_Q = in_channels_Q
        self.in_channels_K = in_channels_K
        self.in_channels_V = in_channels_V
        self.inter_channels = in_channels_V

        if self.inter_channels == 0:
            self.inter_channels = 1
        
        self.query = nn.Conv2d(in_channels=self.in_channels_Q, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels=self.in_channels_K, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=self.in_channels_V, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*3, self.inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(self.inter_channels), nn.ReLU()) 

    def forward(self, Q, K, V):

        batch_size, channels, height, width = V.shape

        query1 = self.query(Q)
        query = query1.view(batch_size, self.inter_channels, -1)

        key1 = self.key(K)
        key = key1.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        value1 = self.value(V)
        value = value1.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        attention = torch.matmul(key, query)
        attention_s = F.softmax(attention, dim=-1)

        output1 = torch.matmul(attention_s, value)
        output1 = output1.permute(0, 2, 1).contiguous()
        output1 = output1.view(batch_size, self.inter_channels, height, width)  
        output2 = self.conv_cat(torch.cat((query1,key1,value1),dim=1))
        output = output1 + output2

        return output, output2
    
class TCSVTNet0303(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TCSVTNet0303, self).__init__()
        
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # self.resnet = resnet34()
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-b627a593.pth'))

        self.consrative2 = ContrastiveAtt2(256,128)
        self.consrative3 = ContrastiveAtt2(128,64)
        self.consrative4 = ContrastiveAtt2(64,32)

        self.multiFusion = MultiFusion(128,64,32)
        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.fam32_1 = DRAtt(128,64)
        self.fam43_1 = DRAtt(64,32)

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

        cross_result4, cur1_4, cur2_4, loss4 = self.consrative4(c1, c1_img2)  
        cross_result3, cur1_3, cur2_3, loss3 = self.consrative3(c2, c2_img2) 
        cross_result2, cur1_2, cur2_2, loss2 = self.consrative2(c3, c3_img2) 

        # cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2)  
        # cross_result3, cur1_3, cur2_3 = self.consrative3(c2, c2_img2) 
        # cross_result2, cur1_2, cur2_2 = self.consrative2(c3, c3_img2) 

        # attention_loss_all = loss2 + loss3 + loss4

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.Translayer3_1(self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1)))
        out4 = self.fam43_1(torch.cat([cross_result4, self.upsamplex2(out3)],1))

        query = self.upsamplex4(torch.abs(cur1_2-cur2_2))
        key = self.upsamplex2(torch.abs(cur1_3-cur2_3))
        value = torch.abs(cur1_4-cur2_4)
        out4_2,output2 = self.multiFusion(query,key,value)

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(output2))

        return out_1, out_2, out_middle_1, out_middle_2 #, attention_loss_all

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.multiFusion.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

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
    
class FEM0306(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM0306, self).__init__()

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

        self.conv_cat = nn.Sequential(nn.Conv2d(self.inter_channels*2, self.inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inter_channels),
                                   nn.ReLU()) # conv_
        
        self.ChannelGate1 = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)
        self.ChannelGate2 = ChannelGate(self.inter_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

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
        E_s = self.ChannelGate1(p_s) * p_s + s
        # E_s = self.scale*p_s+s
        # E_s = E_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        # Intra-image channel attention
        E_q = self.ChannelGate2(q_s) * q_s + q
        # E_q = self.scale*q_s+q
        # E_q = E_q + q
        cpam_feat = self.conv_cat(torch.cat((E_q,E_s),dim=1))

        return cpam_feat, E_q, E_s
    
class TCSVTNet0306(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0306, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = FEM0306(256)
        self.consrative3 = FEM0306(128)
        self.consrative4 = FEM0306(64)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.1),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_1 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.1),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam32_2 = nn.Sequential(
            BAB_Decoder(64*2, 64, 64, 3, 2),
            nn.Dropout(0.1),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                            padding=0, dilation=1, bias=False)
        )
        self.fam43_2 = nn.Sequential(
            BAB_Decoder(32*2, 32, 32, 3, 2),
            nn.Dropout(0.1),
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
        out3 = self.Translayer3_1(self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1)))
        out4 = self.fam43_1(torch.cat([cross_result4, out3],1)) #

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.Translayer3_2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1)))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), out3_2],1)) #

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

class TCSVTNet0307(nn.Module): # CICNet
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0307, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        
        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.consrative2 = ContrastiveAtt2(256,128)
        self.consrative3 = ContrastiveAtt2(128,64)
        self.consrative4 = ContrastiveAtt2(64,32)

        self.Translayer2_1 = BasicConv2d(128,64,1)
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.Translayer2_2 = BasicConv2d(128,64,1)
        self.Translayer3_2 = BasicConv2d(64,32,1)

        self.fam32_1 = DRAtt(64*2,64)
        self.fam43_1 = DRAtt(32*2,32)
        self.fam32_2 = DRAtt(64*2,64)
        self.fam43_2 = DRAtt(32*2,32)
        # self.fam32_1 = nn.Sequential(
        #     BAB_Decoder(64*2, 64, 64, 3, 2),
        #     nn.Dropout(0.2),
        #     TransBasicConv2d(64, 64, kernel_size=2, stride=2,
        #                     padding=0, dilation=1, bias=False)
        # )
        # self.fam43_1 = nn.Sequential(
        #     BAB_Decoder(32*2, 32, 32, 3, 2),
        #     nn.Dropout(0.2),
        #     TransBasicConv2d(32, 32, kernel_size=2, stride=2,
        #                     padding=0, dilation=1, bias=False)
        # )
        # self.fam32_2 = nn.Sequential(
        #     BAB_Decoder(64*2, 64, 64, 3, 2),
        #     nn.Dropout(0.2),
        #     TransBasicConv2d(64, 64, kernel_size=2, stride=2,
        #                     padding=0, dilation=1, bias=False)
        # )
        # self.fam43_2 = nn.Sequential(
        #     BAB_Decoder(32*2, 32, 32, 3, 2),
        #     nn.Dropout(0.2),
        #     TransBasicConv2d(32, 32, kernel_size=2, stride=2,
        #                     padding=0, dilation=1, bias=False)
        # )

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

        cross_result4, cur1_4, cur2_4, loss4 = self.consrative4(c1, c1_img2)
        cross_result3, cur1_3, cur2_3, loss3 = self.consrative3(c2, c2_img2)
        cross_result2, cur1_2, cur2_2, loss2 = self.consrative2(c3, c3_img2)

        out2 = self.Translayer2_1(cross_result2)
        out3 = self.upsamplex2(self.Translayer3_1(self.fam32_1(torch.cat([cross_result3, self.upsamplex2(out2)],1))))
        out4 = self.fam43_1(torch.cat([cross_result4, out3],1)) #

        out2_2 = self.Translayer2_2(torch.abs(cur1_2-cur2_2))
        out3_2 = self.upsamplex2(self.Translayer3_2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), self.upsamplex2(out2_2)],1))))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), out3_2],1)) #

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
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

from .GhostNetv2 import ghostnetv2
class TCSVTNet0308(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0308, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        # self.model = ghostnetv2(deploy=True)
        self.model = ghostnetv2()
        params=self.model.state_dict() 

        # print(model.parameters)
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        # model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        # for k,v in save_model.items():
        #     if k in params.keys():
        #         print(k)
        self.model.load_state_dict(state_dict)

        # self.consrative1 = ContrastiveAtt(112,64)
        # self.consrative2 = ContrastiveAtt(40,32)
        # self.consrative3 = ContrastiveAtt(24,16)
        # self.consrative4 = ContrastiveAtt(16,16)

        # self.fam21_1 = DRAtt(96,64)
        # self.fam32_1 = DRAtt(80,32)
        # self.fam43_1 = DRAtt(48,32)

        # self.fam21_2 = DRAtt(76,32)
        # self.fam32_2 = DRAtt(44,32)
        # self.fam43_2 = DRAtt(40,32)

        self.consrative1 = ContrastiveAtt(192,64)
        self.consrative2 = ContrastiveAtt(80,64)
        self.consrative3 = ContrastiveAtt(48,32)

        # self.consrative4 = ContrastiveAtt(16,16)

        self.fam21_1 = DRAtt(128,64)
        self.fam32_1 = DRAtt(96,32)
        # self.fam43_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(136,64)
        self.fam32_2 = DRAtt(88,32)
        # self.fam43_2 = DRAtt(40,16)

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
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128

        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64

        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32

        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        # cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2) # 16, 128, 128 -> 16
        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        # cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2) # 16, 128, 128 -> 16
        # cross_result3, cur1_3, cur2_3 = self.consrative3(c3, c3_img2) # 48, 64 , 64 -> 32
        # cross_result2, cur1_2, cur2_2 = self.consrative2(c5, c5_img2) # 80, 32 , 32 -> 64
        # cross_result1, cur1_1, cur2_1 = self.consrative1(c7, c7_img2) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))
        # out4 = self.fam43_1(torch.cat([cross_result4, out3],1))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))
        # out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), out3_2],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        # self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        # self.fam43_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        # self.fam43_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CPAMDec_Mix3(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix3,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2
        self.CE = torch.nn.BCEWithLogitsLoss() #size_average=True, reduce=True
        # self.loss_generator = nn.MSELoss()
        self.loss_generator = nn.L1Loss()
    
    def forward(self,x1,y1,x2,y2):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1)
        proj_query2  = self.conv_query2(x2)
        proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)
        
        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query,proj_key1)
        energy2 =  torch.bmm(proj_query,proj_key2)

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1)) 
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2

        out_res1 = torch.bmm(proj_value1,(1-attention).permute(0,2,1))
        out_res2 = torch.bmm(proj_value2,(1-attention).permute(0,2,1))

        loss_res = self.loss_generator(out_res1,out_res2)
        loss_att = 0
        for i in range(m_batchsize):
            loss_att += torch.mean(abs(attention[i,:,:]))

        return out1, out2, loss_res, loss_att
    
class ContrastiveAtt3(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt3, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix3(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2, loss_res, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2, loss_res, loss_att
    
class TCSVTNet0309_OLD(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0309_OLD, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        # self.model = ghostnetv2(deploy=True)
        self.model = ghostnetv2()
        params=self.model.state_dict() 

        # print(model.parameters)
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        # model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        # for k,v in save_model.items():
        #     if k in params.keys():
        #         print(k)
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt3(192,128)
        self.consrative2 = ContrastiveAtt3(80,64)
        self.consrative3 = ContrastiveAtt3(48,32)
        self.consrative4 = ContrastiveAtt3(16,16)

        self.fam21_1 = DRAtt(192,128)
        self.fam32_1 = DRAtt(160,64)
        self.fam43_1 = DRAtt(80,32)

        self.fam21_2 = DRAtt(136,128)
        self.fam32_2 = DRAtt(152,64)
        self.fam43_2 = DRAtt(72,32)

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
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128

        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64

        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32

        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result4, cur1_4, cur2_4, loss_res4, loss_att4 = self.consrative4(c1, c1_img2) # 16, 128, 128 -> 16
        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        # cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2) # 16, 128, 128 -> 16
        # cross_result3, cur1_3, cur2_3 = self.consrative3(c3, c3_img2) # 48, 64 , 64 -> 32
        # cross_result2, cur1_2, cur2_2 = self.consrative2(c5, c5_img2) # 80, 32 , 32 -> 64
        # cross_result1, cur1_1, cur2_1 = self.consrative1(c7, c7_img2) # 192, 16 , 16 -> 128
        loss_res = loss_res1 + loss_res2 + loss_res3 + loss_res4
        loss_att = loss_att1 + loss_att2 + loss_att3 + loss_att4

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))
        out4 = self.fam43_1(torch.cat([cross_result4, out3],1))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), out3_2],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex2(out4))
        out_2 = self.final2(self.upsamplex2(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex2(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex2(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att
        # return out_2, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.fam43_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.fam43_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CPAMDec_Mix4(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix4,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2
        self.loss_generator = nn.L1Loss()
        # torch.nn.KLDivLoss(reduction='mean')
        # self.loss_generator = nn.KLDivLoss(reduction='mean') # nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()


    def forward(self,x1,y1,x2,y2):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_query2  = self.conv_query2(x2).view(m_batchsize,-1,width*height).permute(0,2,1)
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

        out_res1 = torch.bmm(proj_value1,(1-attention).permute(0,2,1))
        out_res2 = torch.bmm(proj_value2,(1-attention).permute(0,2,1))

        loss_res = self.loss_generator(out_res1,out_res2)
        loss_att = torch.mean(abs(energy))
        # for i in range(m_batchsize):
        #     loss_att += torch.mean(abs(attention[i,:,:]))
        return out1, out2, loss_res, loss_att
        # return out1, out2

class ContrastiveAtt4(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt4, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix4(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2, loss_res, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        # return feat_sum, cpam_feat1, cpam_feat2
        return feat_sum, cpam_feat1, cpam_feat2, loss_res, loss_att
    
class TCSVTNet0309(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0309, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        # self.model = ghostnetv2(deploy=True)
        self.model = ghostnetv2()
        params=self.model.state_dict() 

        # print(model.parameters)
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        # model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        # for k,v in save_model.items():
        #     if k in params.keys():
        #         print(k)
        self.model.load_state_dict(state_dict)

        # self.consrative1 = ContrastiveAtt(112,64)
        # self.consrative2 = ContrastiveAtt(40,32)
        # self.consrative3 = ContrastiveAtt(24,16)
        # self.consrative4 = ContrastiveAtt(16,16)

        # self.fam21_1 = DRAtt(96,64)
        # self.fam32_1 = DRAtt(80,32)
        # self.fam43_1 = DRAtt(48,32)

        # self.fam21_2 = DRAtt(76,32)
        # self.fam32_2 = DRAtt(44,32)
        # self.fam43_2 = DRAtt(40,32)

        self.consrative1 = ContrastiveAtt4(192,64)
        self.consrative2 = ContrastiveAtt4(80,64)
        self.consrative3 = ContrastiveAtt4(48,32)

        # self.consrative4 = ContrastiveAtt(16,16)

        self.fam21_1 = DRAtt(128,64)
        self.fam32_1 = DRAtt(96,32)
        # self.fam43_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(136,64)
        self.fam32_2 = DRAtt(88,32)
        # self.fam43_2 = DRAtt(40,16)

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
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128

        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64

        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32

        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        # cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2) # 16, 128, 128 -> 16
        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        loss_res = loss_res1+loss_res2+loss_res3
        loss_att = loss_att1+loss_att2+loss_att3
        # cross_result4, cur1_4, cur2_4 = self.consrative4(c1, c1_img2) # 16, 128, 128 -> 16
        # cross_result3, cur1_3, cur2_3 = self.consrative3(c3, c3_img2) # 48, 64 , 64 -> 32
        # cross_result2, cur1_2, cur2_2 = self.consrative2(c5, c5_img2) # 80, 32 , 32 -> 64
        # cross_result1, cur1_1, cur2_1 = self.consrative1(c7, c7_img2) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))
        # out4 = self.fam43_1(torch.cat([cross_result4, out3],1))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))
        # out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_4-cur2_4), out3_2],1)) # 4, 64, 64, 64

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        
        # self.consrative4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        # self.fam43_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        # self.fam43_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)


class CPAMDec_Mix5(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix5,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

        self.loss_generator = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,y1,x2,y2):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1)
        proj_query2  = self.conv_query2(x2)
        proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)

        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query,proj_key1)
        energy2 =  torch.bmm(proj_query,proj_key2)

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 

        return out1, out2
    
class ContrastiveAtt5(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt5, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix5(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2
    
class TCSVTNet0310(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0310, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,64)
        self.consrative3 = ContrastiveAtt5(48,32)

        self.fam21_1 = DRAtt(128,64)
        self.fam32_1 = DRAtt(96,32)

        self.fam21_2 = DRAtt(136,64)
        self.fam32_2 = DRAtt(88,32)

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
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

from torch.autograd import Variable, Function
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1)

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DeformConv2D_v2(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None, modulation=True):
        super(DeformConv2D_v2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1)
        nn.init.constant_(self.p_conv.weight, 0)
        # self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1)
            nn.init.constant_(self.m_conv.weight, 0)
            # self.m_conv.register_backward_hook(self._set_lr)

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()

        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset
    
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
    
class TCSVTNet0311(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0311, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt(192,64)
        self.consrative2 = ContrastiveAtt(80,32)
        self.consrative3 = ContrastiveAtt(48,16)

        self.fam21_1 = DRAtt(96,32)
        self.fam32_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(136,32)
        self.fam32_2 = DRAtt(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class TCSVTNet0312(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0312, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,32)
        self.consrative3 = ContrastiveAtt5(48,16)

        self.fam21_1 = DRAtt(96,32)
        self.fam32_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(136,32)
        self.fam32_2 = DRAtt(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class TCSVTNet0313(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0313, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,32)
        self.consrative3 = ContrastiveAtt5(48,16)

        self.fam21_1 = DeformConv2D(96,32)
        self.fam32_1 = DeformConv2D(48,16)

        self.fam21_2 = DeformConv2D(136,32)
        self.fam32_2 = DeformConv2D(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        # cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        # cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        # cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        loss_res = loss_res3 + loss_res2 + loss_res1
        loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CADeform(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(CADeform, self).__init__()

        inter_channels = in_channels // 2

        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) 
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) 
        self.ccam_dec = CCAMDec()

        self.spatial = DeformConv2D(inter_channels,out_channels)
        
    def forward(self, x):
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        cpam_feat = self.spatial(ccam_feat)
        return cpam_feat
    
class TCSVTNet0314(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0314, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,32)
        self.consrative3 = ContrastiveAtt5(48,16)

        self.fam21_1 = CADeform(96,32)
        self.fam32_1 = CADeform(48,16)

        self.fam21_2 = CADeform(136,32)
        self.fam32_2 = CADeform(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CADeform2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(CADeform2, self).__init__()

        inter_channels = in_channels

        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) 
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) 
        self.ccam_dec = CCAMDec()

        self.spatial = DeformConv2D(inter_channels,out_channels)
        
    def forward(self, x):
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        cpam_feat = self.spatial(ccam_feat)
        return cpam_feat
    
class TCSVTNet0315(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0315, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,32)
        self.consrative3 = ContrastiveAtt5(48,16)

        self.fam21_1 = CADeform2(96,32)
        self.fam32_1 = CADeform2(48,16)

        self.fam21_2 = CADeform2(136,32)
        self.fam32_2 = CADeform2(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

from .ssim import SSIM,MS_SSIM
class CPAMDec_Mix6(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix6,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

        # self.loss_generator = nn.MSELoss()
        self.loss_generator = nn.L1Loss()
        # self.ssim = SSIM(data_range=1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,y1,x2,y2,label=None):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1)
        proj_query2  = self.conv_query2(x2)
        proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)

        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query,proj_key1)
        energy2 =  torch.bmm(proj_query,proj_key2)
        energy = torch.abs(energy1-energy2)

        # proj_key1 =  self.conv_key1(y1)
        # proj_key2 =  self.conv_key2(y2)
        # proj_key = torch.abs(proj_key1-proj_key2).view(m_batchsize,K,-1).permute(0,2,1)
        # proj_value1 = self.conv_value1(y1).permute(0,2,1)
        # proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        # energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 

        # # cmask = (torch.sign(attention - 0.5) + 1) / 2
        # # out_res1 = torch.bmm(proj_value1,((1-cmask)*(1-attention)).permute(0,2,1))
        # # out_res2 = torch.bmm(proj_value2,((1-cmask)*(1-attention)).permute(0,2,1))

        # # # loss_res = self.loss_generator(out_res1,out_res2)

        # # res_loss = 0
        # # loss_att = 0
        # # for i in range(m_batchsize):
        # #     res_loss += self.loss_generator(out_res1[i], out_res2[i]) * M * K / torch.sum(1-cmask[i])
        # #     loss_att += torch.sum(abs(cmask[i]))
        # # res_loss = res_loss / m_batchsize
        # # # loss_att = loss_att / m_batchsize

        # # norm = nn.functional.normalize(energy, p=1, dim=1)
        # norm = (attention-attention.min())/(attention.max()-attention.min())
        # cmask = (norm-norm.min())/(norm.max()-norm.min())

        # # # cmask = (torch.sign(norm - 0.3) + 1) / 2

        # out_res1 = torch.bmm(proj_value1,(1-cmask).permute(0,2,1))
        # out_res2 = torch.bmm(proj_value2,(1-cmask).permute(0,2,1))

        # # loss_res = self.loss_generator(out_res1,out_res2)

        # # print('label:',label.shape)
        # res_loss = 0
        # loss_att = 0 
        # if label is not None:
        #     for i in range(m_batchsize):
        #         res_loss += self.loss_generator(out_res1[i], out_res2[i]) #* M * K / (torch.sum(cmask[i]<0.5)+1)
        #         # loss_att += torch.mean(abs(cmask[i]))
        #     att = cmask.permute(0,2,1).view(m_batchsize,K,width,height)

        #     label = F.interpolate(label, size=(width,height))
        #     loss_att = self.loss_generator(torch.mean((att),dim=1),label)

        #     res_loss = res_loss / m_batchsize
        #     # loss_att = loss_att / m_batchsize
        #     # res_loss = self.loss_generator(out_res1, out_res2) 
        # else:
        #     res_loss = self.loss_generator(out_res1, out_res2)
        #     loss_att = torch.mean(abs(cmask),dim=1)

        if label is not None:
            label = F.interpolate(label, size=(width,height))
            norm = nn.functional.normalize(energy, p=1, dim=1)
            norm = (norm-norm.min())/(norm.max()-norm.min())
            cmask = (torch.sign(norm - 0.3) + 1) / 2
            out_res1 = torch.bmm(proj_value1,((1-cmask)).permute(0,2,1))
            out_res2 = torch.bmm(proj_value2,((1-cmask)).permute(0,2,1))

            # loss_res = self.loss_generator(out_res1,out_res2)

            res_loss = 0
            loss_att = 0 
            for i in range(m_batchsize):
                res_loss += self.loss_generator(out_res1[i], out_res2[i]) #* M * K / (torch.sum(1-cmask[i])+1)
            res_loss = res_loss / m_batchsize
            att = cmask.permute(0,2,1).view(m_batchsize,K,width,height)
            loss_att = self.loss_generator(torch.mean((att),dim=1),label)
            loss_att = loss_att / m_batchsize
            # attention2 = attention.permute(0,2,1).view(m_batchsize,K,width,height)*255
            return out1, out2, res_loss, loss_att
        else:
            return out1, out2
    
class ContrastiveAtt6(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt6, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix6(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y, label=None):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        if label is not None:
            cpam_feat1, cpam_feat2, loss_res, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y,label) 
        else: 
            cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y,label) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))

        if label is not None:
            return feat_sum, cpam_feat1, cpam_feat2, loss_res, loss_att
        else:
            return feat_sum, cpam_feat1, cpam_feat2

class TCSVTNet0316(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0316, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt6(192,64)
        self.consrative2 = ContrastiveAtt6(80,32)
        self.consrative3 = ContrastiveAtt6(48,16)

        self.fam21_1 = DRAtt(96,32)
        self.fam32_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(136,32)
        self.fam32_2 = DRAtt(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        if labels is not None:
            cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1),labels) # 48, 64 , 64 -> 32
            cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 64
            cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) # 192, 16 , 16 -> 128
            loss_res = loss_res3 + loss_res2 + loss_res1
            loss_att = loss_att3 + loss_att2 + loss_att1
        else: 
            cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1),labels) # 48, 64 , 64 -> 32
            cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 64
            cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels)

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        if labels is not None:
            return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att
        else:
            return out_1, out_2, out_middle_1, out_middle_2

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class TCSVTNet0316_large(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0316_large, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt6(192,128)
        self.consrative2 = ContrastiveAtt6(80,64)
        self.consrative3 = ContrastiveAtt6(48,32)

        self.fam21_1 = DRAtt(192,64)
        self.fam32_1 = DRAtt(96,32)

        self.fam21_2 = DRAtt(136,64)
        self.fam32_2 = DRAtt(88,32) # DRAtt DeformConv2d

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
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        loss_res = loss_res3 + loss_res2 + loss_res1
        loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class MGAP(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(MGAP, self).__init__()
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
        
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        feat5 = self.conv5(self.pool5(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4, feat5), 2)
    
class CCAM(nn.Module):
    def __init__(self, in_channels=128, norm_layer=nn.BatchNorm2d):
        super(CCAM,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.ccam_enc_x = MGAP(in_channels, norm_layer)
        self.ccam_enc_y = MGAP(in_channels//16, norm_layer)
        # self.dw_conv = DWConv(in_channels)

    def forward(self, x, y):
        m_batchsize,C,width ,height = x.size()
        x_AVGPooling = self.ccam_enc_x(x)
        x_reshape =x_AVGPooling.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_AVGPooling = self.ccam_enc_y(y)
        y_reshape =y_AVGPooling.view(B,K,-1)
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
    
class SingleAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(SingleAtt, self).__init__()

        # inter_channels = in_channels // 2
        inter_channels = out_channels

        self.conv_cpam_b = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = CPAMEnc(out_channels, norm_layer) # en_s
        self.cpam_dec = CPAMDec(out_channels) # de_s

        # self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) 
        # self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
        #                            norm_layer(inter_channels//16),
        #                            nn.ReLU()) 
        # self.ccam_dec = CCAM(inter_channels)
        
    def forward(self, x):
        # ccam_b = self.conv_ccam_b(x)
        # ccam_f = self.ccam_enc(ccam_b)
        # ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        cpam_b = self.conv_cpam_b(x)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)
        return cpam_feat
    
class LCANet_woDRA(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(LCANet_woDRA, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt6(192,64)
        self.consrative2 = ContrastiveAtt6(80,32)
        self.consrative3 = ContrastiveAtt6(48,16)

        self.fam21_1 = SingleAtt(96,32)
        self.fam32_1 = SingleAtt(48,16)

        self.fam21_2 = SingleAtt(136,32)
        self.fam32_2 = SingleAtt(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        loss_res = loss_res3 + loss_res2 + loss_res1
        loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

from . import MobileNetV2
class Mobile_LCA(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(Mobile_LCA, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = MobileNetV2.mobilenet_v2(pretrained=True)
        # params=self.model.state_dict() 
        # save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        # state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        # self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt6(96,64)
        self.consrative2 = ContrastiveAtt6(32,32)
        self.consrative3 = ContrastiveAtt6(24,16)

        self.fam21_1 = DRAtt(96,32)
        self.fam32_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(64,32)
        self.fam32_2 = DRAtt(44,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):

        x1_1, c2, c4, c6, x1_5 = self.model(imgs1)
        x2_1, c2_img2, c4_img2, c6_img2, x2_5 = self.model(imgs2)
        # 16, 24, 32, 96, 320

        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(c2, c2_img2) # 24, 64 , 64 -> 16
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(c4, c4_img2) # 32, 32 , 32 -> 32
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(c6, c6_img2) # 96, 16 , 16 -> 64

        loss_res = loss_res3 + loss_res2 + loss_res1
        loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class LGCA(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LGCA, self).__init__()

        inter_channels = out_channels

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
        return ccam_feat
    
class TCSVTNet0317(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0317, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,32)
        self.consrative3 = ContrastiveAtt5(48,16)

        self.SA21_1 = DeformConv2D(64,32)
        self.SA32_1 = DeformConv2D(32,16)
        self.CA21_1 = LGCA(64,32)
        self.CA32_1 = LGCA(32,16)
        # self.CA21_1 = DeformConv2D(64,32)
        # self.CA32_1 = DeformConv2D(32,16)

        self.SA21_2 = DeformConv2D(96,32)
        self.SA32_2 = DeformConv2D(32,16) 
        self.CA21_2 = LGCA(72,32)
        self.CA32_2 = LGCA(40,16)
        # self.CA21_2 = DeformConv2D(72,32)
        # self.CA32_2 = DeformConv2D(40,16)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 16
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64

        out2 = self.SA32_1(self.upsamplex2(self.CA21_1(torch.cat([cross_result2, self.SA21_1(self.upsamplex2(cross_result1))],1))))
        out3 = self.upsamplex2(self.CA32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.SA32_2(self.upsamplex2(self.CA21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.SA21_2(self.upsamplex2(torch.abs(cur1_1-cur2_1)))],1))))
        out3_2 = self.upsamplex2(self.CA32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.SA21_1.apply(init_weights)
        self.SA32_1.apply(init_weights)
        self.SA21_2.apply(init_weights)
        self.SA32_2.apply(init_weights)

        self.CA21_1.apply(init_weights)
        self.CA32_1.apply(init_weights)
        self.CA21_2.apply(init_weights)
        self.CA32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class TCSVTNet0318(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TCSVTNet0318, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt5(192,64)
        self.consrative2 = ContrastiveAtt5(80,32)
        self.consrative3 = ContrastiveAtt5(48,16)

        self.fam21_1 = DeformConv2D_v2(96,32)
        self.fam32_1 = DeformConv2D_v2(48,16)

        self.fam21_2 = DeformConv2D_v2(136,32)
        self.fam32_2 = DeformConv2D_v2(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)

class CPAMDec_Mix7(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix7,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.threshold = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

        # self.loss_generator = nn.MSELoss()
        self.loss_generator = nn.L1Loss()
        # self.ssim = SSIM(data_range=1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,y1,x2,y2):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1)
        proj_query2  = self.conv_query2(x2)
        proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)

        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query,proj_key1)
        energy2 =  torch.bmm(proj_query,proj_key2)

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 

        # cmask = (torch.sign(attention - 0.5) + 1) / 2
        # out_res1 = torch.bmm(proj_value1,((1-cmask)*(1-attention)).permute(0,2,1))
        # out_res2 = torch.bmm(proj_value2,((1-cmask)*(1-attention)).permute(0,2,1))

        # # loss_res = self.loss_generator(out_res1,out_res2)

        # res_loss = 0
        # loss_att = 0
        # for i in range(m_batchsize):
        #     res_loss += self.loss_generator(out_res1[i], out_res2[i]) * M * K / torch.sum(1-cmask[i])
        #     loss_att += torch.sum(abs(cmask[i]))
        # res_loss = res_loss / m_batchsize
        # # loss_att = loss_att / m_batchsize

        norm = nn.functional.normalize(energy, p=1, dim=1)
        norm = (norm-norm.min())/(norm.max()-norm.min())
        cmask = (torch.sign(norm - 0.3) + 1) / 2
        out_res1 = torch.bmm(proj_value1,((1-cmask)).permute(0,2,1))
        out_res2 = torch.bmm(proj_value2,((1-cmask)).permute(0,2,1))

        # loss_res = self.loss_generator(out_res1,out_res2)

        res_loss = 0
        loss_att = 0 
        for i in range(m_batchsize):
            res_loss += self.loss_generator(out_res1[i], out_res2[i]) * M * K / torch.sum(1-cmask[i])
            loss_att += torch.mean(abs(cmask[i]))
        res_loss = res_loss / m_batchsize

        # res_loss = self.loss_generator(out_res1, out_res2) 
        
        return out1, out2, res_loss, loss_att
    
class ContrastiveAtt7(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt7, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix7(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2, loss_res, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2, loss_res, loss_att
    
class NeurIPS0317(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(NeurIPS0317, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt6(192,64) # collective-attention distribution
        self.consrative2 = ContrastiveAtt6(80,32)
        self.consrative3 = ContrastiveAtt6(48,16)

        self.fam21_1 = DeformConv2D(96,32)
        self.fam32_1 = DeformConv2D(48,16)

        self.fam21_2 = DeformConv2D(136,32)
        self.fam32_2 = DeformConv2D(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        loss_res = loss_res3 + loss_res2 + loss_res1
        loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        # self.fam21_1.apply(init_weights)
        self.fam32_1.apply(init_weights)
        # self.fam21_2.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)


class CPAMDec_Mix0319(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix0319,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

        self.loss_generator = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,y1,x2,y2):
        m_batchsize,C,width ,height = x1.size()
        m_batchsize,K,M = y1.size()

        proj_query1  = self.conv_query1(x1)
        proj_query2  = self.conv_query2(x2)
        proj_query = torch.cat([proj_query1,proj_query2],1).view(m_batchsize,-1,width*height).permute(0,2,1)

        proj_key1 =  self.conv_key1(y1).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value1 = self.conv_value1(y1).permute(0,2,1)

        proj_key2 =  self.conv_key2(y2).view(m_batchsize,K,-1).permute(0,2,1)
        proj_value2 = self.conv_value2(y2).permute(0,2,1) 

        energy1 =  torch.bmm(proj_query,proj_key1)
        energy2 =  torch.bmm(proj_query,proj_key2)

        energy = torch.abs(energy1-energy2)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 

        norm = nn.functional.normalize(energy, p=1, dim=1)
        norm = (norm-norm.min())/(norm.max()-norm.min())
        cmask = (torch.sign(norm - 0.3) + 1) / 2
        out_res1 = torch.bmm(proj_value1,(1-cmask).permute(0,2,1)) # torch.mul((1-cmask),attention)
        out_res2 = torch.bmm(proj_value2,(1-cmask).permute(0,2,1))

        res_loss = 0
        loss_att = 0 
        for i in range(m_batchsize):
            res_loss += self.loss_generator(out_res1[i], out_res2[i]) * M * K / torch.sum(1-cmask[i])
            loss_att += torch.mean(abs(cmask[i]))
        res_loss = res_loss / m_batchsize
        # res_loss = self.loss_generator(out_res1, out_res2) 
        
        return out1, out2, res_loss, loss_att
    
class ContrastiveAtt0319(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt0319, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix0319(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2, res_loss, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2, res_loss, loss_att
    
class NeurIPS0319(nn.Module): 
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(NeurIPS0319, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt0319(192,64)
        self.consrative2 = ContrastiveAtt0319(80,32)
        self.consrative3 = ContrastiveAtt0319(48,16)

        self.fam21_1 = DRAtt(96,32)
        self.fam32_1 = DRAtt(48,16)

        self.fam21_2 = DRAtt(136,32)
        self.fam32_2 = DRAtt(56,16) # DRAtt DeformConv2d

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3, res_loss3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, res_loss2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, res_loss1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        res_loss = res_loss1 + res_loss2 + res_loss3
        loss_att = loss_att1 + loss_att2 + loss_att3

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        return out_1, out_2, out_middle_1, out_middle_2 ,res_loss, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)
    
class NeurIPS0322(nn.Module): 
    def __init__(self, num_classes=2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(NeurIPS0322, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt0319(192,64)
        self.consrative2 = ContrastiveAtt0319(80,32)
        self.consrative3 = ContrastiveAtt0319(48,16)

        self.Translayer2_1 = DeformConv2D(64,32)
        self.Translayer3_1 = DeformConv2D(32,16)
        self.Translayer2_2 = DeformConv2D(96,32)
        self.Translayer3_2 = DeformConv2D(32,16)

        self.fam32_1 = BAB_Decoder(64, 32, 32, 3, 2) # (64,32) # DeformConv2D(64,32) # BAB_Decoder(64*2, 64, 64, 3, 2) DeformConv2D
        self.fam43_1 = BAB_Decoder(32, 16, 16, 3, 2)# DeformConv2D(32,16)
        self.fam32_2 = BAB_Decoder(72, 32, 32, 3, 2)# DeformConv2D(72,32)
        self.fam43_2 = BAB_Decoder(40, 16, 16, 3, 2)# DeformConv2D(40,16)
    
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.final1 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_1 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.final_middle_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )
        self.sigmoid = nn.Sigmoid()
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, labels=None):
        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs1)))
        c1 = self.model.blocks[0](c0) # 16, 128 , 128
        c2 = self.model.blocks[1](c1) # 24, 64 , 64
        c3 = self.model.blocks[2](c2) # 24, 64 , 64
        c4 = self.model.blocks[3](c3) # 40, 32 , 32
        c5 = self.model.blocks[4](c4) # 40, 32 , 32
        c6 = self.model.blocks[5](c5) # 80, 16, 16
        c7 = self.model.blocks[6](c6) # 112, 16, 16

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(imgs2)))
        c1_img2 = self.model.blocks[0](c0_img2) # 16, 128 , 128
        c2_img2 = self.model.blocks[1](c1_img2) # 24, 64 , 64
        c3_img2 = self.model.blocks[2](c2_img2) # 24, 64 , 64
        c4_img2 = self.model.blocks[3](c3_img2) # 40, 32 , 32
        c5_img2 = self.model.blocks[4](c4_img2) # 40, 32 , 32
        c6_img2 = self.model.blocks[5](c5_img2) # 
        c7_img2 = self.model.blocks[6](c6_img2) # 

        cross_result3, cur1_3, cur2_3, res_loss3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, res_loss2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, res_loss1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        res_loss = res_loss1 + res_loss2 + res_loss3
        loss_att = loss_att1 + loss_att2 + loss_att3

        out2 = self.Translayer2_1(cross_result1) # 64 -> 32
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result2, self.upsamplex2(out2)],1))) # 32 + 32 -> 32
        out4 = self.fam43_1(torch.cat([cross_result3, self.Translayer3_1(out3)],1)) # 16 + 16 -> 16

        out2_2 = self.Translayer2_2(torch.abs(cur1_1-cur2_1)) # 96 -> 32
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(out2_2)],1))) # 40 + 32 -> 32
        out4_2 = self.fam43_2(torch.cat([torch.abs(cur1_3-cur2_3), self.Translayer3_2(out3_2)],1)) # 24 + 32 -> 16

        out_1 = self.final1(self.upsamplex4(out4))
        out_2 = self.final2(self.upsamplex4(out4_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out3))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out3_2))

        return out_1, out_2, out_middle_1, out_middle_2 ,res_loss, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.fam43_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)