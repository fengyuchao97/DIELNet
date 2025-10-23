import torch
import torch.nn as nn
import math
from .resnet import resnet18
from .GhostNetv2 import ghostnetv2
from .MobileNetV2 import mobilenet_v2
from einops import rearrange
import torch.nn.functional as F

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

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        # x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) #NHWC
        return x
    
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
        
class MGAP(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(MGAP, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(1)
        # self.pool2 = nn.AdaptiveAvgPool2d(2)
        # self.pool3 = nn.AdaptiveAvgPool2d(3)
        # self.pool4 = nn.AdaptiveAvgPool2d(6)

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        
        # feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        # feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        # feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        # feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return x.view(b,c,-1) #torch.cat((feat1, feat2, feat3, feat4), 2)
    
class ITCA(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(ITCA,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.ratio = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2
        # self.conv_value1_2 = DWConv(in_channels)

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2
        # self.conv_value2_2 = DWConv(in_channels)

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

        proj_key = torch.abs(proj_key1-proj_key2)

        # energy1 =  torch.bmm(proj_query,proj_key1)
        # energy2 =  torch.bmm(proj_query,proj_key2)

        # energy = torch.abs(energy1-energy2)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 # self.conv_value1_2(x1) 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 # self.conv_value2_2(x2) 

        # norm = nn.functional.normalize(energy, p=1, dim=1)
        # norm = (norm-norm.min())/(norm.max()-norm.min())
        # ratio = self.ratio
        # if ratio > 0.5: ratio = 0.5
        # elif ratio < 0: ratio = 0
        # cmask = (torch.sign(norm - ratio) + 1) / 2

        # norm = (attention-attention.min())/(attention.max()-attention.min())
        # cmask = (torch.sign(norm-0.3) + 1) / 2
        # out_res1 = torch.bmm(proj_value1,(1-cmask).permute(0,2,1))
        # out_res2 = torch.bmm(proj_value2,(1-cmask).permute(0,2,1))

        # res_loss = 0
        # loss_att = 0 
        # for i in range(m_batchsize):
        #     nc_sum = ((1-cmask[i]) == 1).sum()
        #     # nc_ratio = (nc_sum) / (width*height * K)
        #     # print("nc_ratio:")
        #     # print(nc_ratio)
        #     res_loss += self.loss_generator(out_res1[i], out_res2[i]) * (width*height * K) / (nc_sum+0.01)
        #     loss_att += torch.mean(abs(cmask[i]))
        # res_loss = res_loss / m_batchsize
        # loss_att = loss_att / m_batchsize

        return out1, out2 #, res_loss, loss_att

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
    
class ITFI(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ITFI, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = MGAP(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = MGAP(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = ITCA(inter_channels) # de_s

        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) 
        
    def forward(self, x, y):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y)  #, loss_res, loss_att

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2#, loss_res, loss_att

class CPAM(nn.Module):
    def __init__(self,in_channels):
        super(CPAM,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) 
        self.conv_key = nn.Linear(in_channels, in_channels//4) 
        self.conv_value = nn.Linear(in_channels, in_channels) 
        # self.dw_conv = DWConv(in_channels)

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
        out = self.scale*out + x # self.dw_conv(x)
        return out
    
class MDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(MDecoder, self).__init__()

        inter_channels = in_channels // 2

        self.conv_cpam_b = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = MGAP(out_channels, norm_layer) # en_s
        self.cpam_dec = CPAM(out_channels) # de_s

        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) 
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) 
        self.ccam_dec = CCAM(inter_channels)
        
    def forward(self, x):
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        cpam_b = self.conv_cpam_b(ccam_feat)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)
        return cpam_feat

class LCANet_N3C(nn.Module): 
    def __init__(self, num_classes=2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(LCANet_N3C, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ITFI(192,64)
        self.consrative2 = ITFI(80,32)
        self.consrative3 = ITFI(48,16)

        self.fam21_1 = MDecoder(96,32)
        self.fam32_1 = MDecoder(48,16)
        self.fam21_2 = MDecoder(136,32)
        self.fam32_2 = MDecoder(56,16) # DRAtt DeformConv2d

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

        # cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        # cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        # cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        # loss_res = loss_res3 + loss_res2 + loss_res1
        # loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = torch.sigmoid(self.final1(self.upsamplex2(out3)))
        out_2 = torch.sigmoid(self.final2(self.upsamplex2(out3_2)))
        out_middle_1 = torch.sigmoid(self.final_middle_1(self.upsamplex4(out2)))
        out_middle_2 = torch.sigmoid(self.final_middle_2(self.upsamplex4(out2_2)))

        return out_1, out_2, out_middle_1, out_middle_2#, loss_res, loss_att

    def init_weights(self):
        self.consrative1.apply(init_weights)
        self.consrative2.apply(init_weights)
        self.consrative3.apply(init_weights)        

        self.fam21_1.apply(init_weights)
        self.fam21_2.apply(init_weights)
        self.fam32_1.apply(init_weights)
        self.fam32_2.apply(init_weights)

        self.final1.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_middle_1.apply(init_weights)
        self.final_middle_2.apply(init_weights)



class CPAMEnc(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(1)
        # self.pool2 = nn.AdaptiveAvgPool2d(2)
        # self.pool3 = nn.AdaptiveAvgPool2d(3)
        # self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        
        # feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        # feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        # feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        # feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return self.conv1(x).view(b,c,-1) #torch.cat((feat1, feat2, feat3, feat4), 2)

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
    
class DRAtt(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DRAtt, self).__init__()

        inter_channels = in_channels // 2
        if inter_channels <= 16:
            inter_channels = in_channels

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
    
class CPAMDec_Mix(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.loss_generator = nn.L1Loss()

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
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
    
class ContrastiveAtt(nn.Module):
    def __init__(self, in_channels, out_channels=32, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt, self).__init__()

        inter_channels = in_channels//2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix(inter_channels) # de_s
        
    def forward(self, x, y, label=None):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        if label is not None:
            cpam_feat1, cpam_feat2, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y,label) 
            return cpam_feat1, cpam_feat2, loss_att
        else: 
            cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 
            return cpam_feat1, cpam_feat2

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
    def __init__(self, in_ch, enc_chs=(32,64), expansion=1):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = expansion
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
    
class densecat_cat_single_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_single_diff, self).__init__()

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

        return self.conv_out(torch.abs(x1-y1)+torch.abs(x2-y2)+torch.abs(x3-y3))

class densecat_cat(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat, self).__init__()

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

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn*2, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn*2, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn*2, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        fuse1 = self.conv4(torch.cat([x1,y1],1))
        fuse2 = self.conv5(torch.cat([x2,y2],1))
        fuse3 = self.conv6(torch.cat([x3,y3],1))

        return self.conv_out(fuse1+fuse2+fuse3)
    
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
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
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
    
class Difference(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(Difference, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None

        self.diff = densecat_cat_diff(dim_in, dim_out)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in*2, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(dim_out*2, dim_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim_out),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        diff = self.diff(x1,x2)
        cat = torch.cat([x1,x2],1)
        output = self.conv_out(torch.cat([diff,self.conv2(cat)],1))
        return output
    
class Difference2(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=False):
        super(Difference2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in*2, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x1, x2):
        diff = torch.abs(x1-x2)
        cat = torch.cat([x1,x2],1)
        output1 = self.conv1(diff) 
        output2 = self.conv2(cat)
        return output1, output2
    
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
        self.conv_out = Conv1x1(out_ch, num_classes)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x2 = F.interpolate(x2, size=x1.shape[2:])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        out = self.conv_fuse(x)
        output = self.conv_out(out)
        return out, output

class Decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_right, out_channel, num_classes=1, norm_layer=nn.BatchNorm2d):
        super(Decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_right, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)
        self.conv_out = Conv1x1(out_channel, num_classes)

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
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        output = self.conv_out(out)
        return out, output 
    
class ContrastiveAtt_chunk(nn.Module):
    def __init__(self, in_channels, out_channels=32, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt_chunk, self).__init__()

        inter_channels = in_channels // 2

        # self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) # conv5_s
        # self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) # conv5_s
        
        # self.conv_cpam_b_x_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) # conv5_s
        # self.conv_cpam_b_y_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU()) # conv5_s

        self.conv_x = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU()) # conv5_s
        self.conv_y = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                   norm_layer(in_channels),
                                   nn.ReLU()) # conv5_s
        
        self.dwconv_x = DWConv(inter_channels)
        self.dwconv_y = DWConv(inter_channels)  

        self.cpam_enc_x = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix(inter_channels) # de_s
        
    def forward(self, x, y, label=None):
        x1,x2 = self.conv_x(x).chunk(2,dim=1)
        y1,y2 = self.conv_y(y).chunk(2,dim=1)

        # cpam_b_x = self.conv_cpam_b_x(x1)
        # cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        # cpam_b_y = self.conv_cpam_b_y(y1)
        # cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        # x2 = self.conv_cpam_b_x_2(x2)
        # y2 = self.conv_cpam_b_y_2(y2)

        # if label is not None:
        #     cpam_feat1, cpam_feat2, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y,label) 
        #     cpam_feat1 = cpam_feat1 + x2
        #     cpam_feat2 = cpam_feat2 + y2
        #     return cpam_feat1, cpam_feat2, loss_att
        # else: 
        #     cpam_feat1, cpam_feat2 = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 
        #     cpam_feat1 = cpam_feat1 + x2
        #     cpam_feat2 = cpam_feat2 + y2
        #     return cpam_feat1, cpam_feat2
        cpam_f_x = self.cpam_enc_x(x1).permute(0,2,1)
        cpam_f_y = self.cpam_enc_y(y1).permute(0,2,1)

        x2 = self.dwconv_x(x2)
        y2 = self.dwconv_y(y2)

        if label is not None:
            cpam_feat1, cpam_feat2, loss_att = self.cpam_dec_mix(x1,cpam_f_x,y1,cpam_f_y,label) 
            out1 = cpam_feat1 + x2
            out2 = cpam_feat2 + y2
            return out1, out2, loss_att
        else: 
            cpam_feat1, cpam_feat2 = self.cpam_dec_mix(x1,cpam_f_x,y1,cpam_f_y) 
            out1 = cpam_feat1 + x2
            out2 = cpam_feat2 + y2
            return out1, out2
        
class Local_interaction_chunk(nn.Module):
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

        inter_channels = in_channels // 2

        self.init_conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU()) # conv5_s
        self.init_conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU()) # conv5_
        self.dwconv1 = DWConv(inter_channels)
        self.dwconv2 = DWConv(inter_channels)
        
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
        x1_1,x1_2 = self.init_conv1(x1).chunk(2,dim=1)
        x2_1,x2_2 = self.init_conv2(x2).chunk(2,dim=1)

        x1_2 = self.dwconv1(x1_2)
        x2_2 = self.dwconv2(x2_2)

        B, C, H, W = x1_1.shape
        q1,k1,v1, q2,k2,v2 = self.get_qkv(x1_1,x2_1)
        x1_interact, x2_interact = self.forward_interaction_local(q1, k1, v1, q2, k2, v2, H, W)

        out1 = x1_interact + x1_2 
        out2 = x2_interact + x2_2
        return out1, out2
    
class TDANet(nn.Module): 
    def __init__(self, num_classes=1, drop_rate=0., normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TDANet, self).__init__()
        self.video_len = 8
        self.show_Feature_Maps = show_Feature_Maps
        self.enc_chs = (16,32)

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.global_consrative2 = ContrastiveAtt(192)
        self.global_consrative1 = ContrastiveAtt(80)
        self.local_consrative2 = Local_interaction(48, window_size=16)
        self.local_consrative1 = Local_interaction(32, window_size=16)
        
        self.encoder_v = VideoEncoder(in_ch=3, enc_chs=self.enc_chs)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in self.enc_chs
            ]
        )
        
        self.fusion3 = Fusion(96,64)
        self.fusion2 = Fusion(40,32)
        self.fusion1 = Fusion(24,16)
        self.fusion0 = Fusion(16,8)

        # self.fusion3 = Difference2(96,64)
        # self.fusion2 = Difference2(40,32)
        # self.fusion1 = Difference2(24,16)
        # self.fusion0 = Difference2(16,8)

        self.decoder1 = DecBlock(64+64, 32, num_classes) 
        self.decoder2 = DecBlock(32+32, 16, num_classes)
        self.decoder3 = DecBlock(16+8, 8, num_classes)

        # self.decoder_sub1 = DecBlock(64+64,32,num_classes)
        # self.decoder_sub2 = DecBlock(32+32,16,num_classes)
        # self.decoder_sub3 = DecBlock(16+8,8,num_classes)
        # self.conv_out_v1 = Conv1x1(16, num_classes)
        self.conv_out_v = Conv1x1(32, num_classes)
        
        if normal_init:
            self.init_weights()

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
    
    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        frames = self.pair_to_video(imgs1, imgs2) # b, 9, 3, 256, 256

        feats_v = self.encoder_v(frames.transpose(1,2))
        # print('feats_v',feats_v[0].shape) # b, 3, 9, 256, 256
        feats_v.pop(0)
        for i, feat in enumerate(feats_v):
            # print('self.tem_aggr(feat)',self.tem_aggr(feat).shape)
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))
            # print('The ',i,"th :",feats_v[i].shape)
            # The  0 th : torch.Size([2, 32, 64, 64])
            # The  1 th : torch.Size([2, 64, 32, 32])

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
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        cur1_0, cur2_0 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
        cur1_1, cur2_1 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64

        if labels is not None:
            cur1_2, cur2_2, loss_att2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3, loss_att1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) # 192, 16 , 16 -> 64, 16 , 16
            loss_att = loss_att1 + loss_att2
        else:
            cur1_2, cur2_2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16

        # cur1_0, cur2_0 = self.local_consrative1(c1, c1_img2) # 16, 128 , 128 -> 8, 128 , 128
        # cur1_1, cur2_1 = self.local_consrative2(c3, c3_img2) # 48, 64 , 64 -> 24, 64 , 64

        # if labels is not None:
        #     cur1_2, cur2_2, loss_att2 = self.global_consrative1(c5, c5_img2, labels) # 80, 32 , 32 -> 32, 32 , 32
        #     cur1_3, cur2_3, loss_att1 = self.global_consrative2(c7, c7_img2, labels) # 192, 16 , 16 -> 64, 16 , 16
        #     loss_att = loss_att1 + loss_att2
        # else:
        #     cur1_2, cur2_2 = self.global_consrative1(c5, c5_img2) # 80, 32 , 32 -> 32, 32 , 32
        #     cur1_3, cur2_3 = self.global_consrative2(c7, c7_img2) # 192, 16 , 16 -> 64, 16 , 16

        fuse2 = self.fusion2(cur1_2,cur2_2) # [2, 32, 32, 32]
        fuse1 = self.fusion1(cur1_1,cur2_1) # [2, 16, 64, 64]

        fuse3 = self.fusion3(cur1_3,cur2_3) # [2, 64, 16, 16]
        cat2 = torch.cat([fuse2,feats_v[1]],1) # [2, 64, 32, 32]
        cat1 = torch.cat([fuse1,feats_v[0]],1) # [2, 32, 64, 64]
        fuse0 = self.fusion0(cur1_0,cur2_0) # [2, 8, 128, 128]

        dec1,output_middle2 = self.decoder1(cat2,fuse3)
        dec2,output_middle1 = self.decoder2(cat1,dec1)
        dec3,output = self.decoder3(fuse0,dec2)

        # return output,output_middle1,output_middle2
        if return_aux:
            # output_middle1 = F.interpolate(output_middle1, size=imgs1.shape[2:])
            # output_middle2 = F.interpolate(output_middle2, size=imgs1.shape[2:])
            output = F.interpolate(output, size=imgs1.shape[2:])
            
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)

            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=imgs1.shape[2:])
            pred_v = torch.sigmoid(pred_v)

            # pred_v2 = self.conv_out_v2(feats_v[1])
            # pred_v2 = F.interpolate(pred_v2, size=imgs1.shape[2:])
            # pred_v2 = torch.sigmoid(pred_v2)

            if labels is not None:
                # return output, output_middle1, output_middle2, pred_v1, pred_v2, loss_att
                return output, output_middle1, output_middle2, pred_v, loss_att
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att
            else:
                return output
            
        # sub2, fuse2 = self.fusion2(cur1_2,cur2_2) # [2, 32, 32, 32]
        # sub1, fuse1 = self.fusion1(cur1_1,cur2_1) # [2, 16, 64, 64]
        # sub3, fuse3 = self.fusion3(cur1_3,cur2_3) # [2, 64, 16, 16]
        # sub0, fuse0 = self.fusion0(cur1_0,cur2_0) # [2, 8, 128, 128]
        
        # cat2 = torch.cat([fuse2,feats_v[1]],1) # [2, 64, 32, 32]
        # cat1 = torch.cat([fuse1,feats_v[0]],1) # [2, 32, 64, 64]
        # cat2_sub = torch.cat([sub2,feats_v[1]],1) # [2, 64, 32, 32]
        # cat1_sub = torch.cat([sub1,feats_v[0]],1) # [2, 32, 64, 64]

        # dec_fuse1,output_middle_fuse2 = self.decoder1(cat2,fuse3)
        # dec_fuse2,output_middle_fuse1 = self.decoder2(cat1,dec_fuse1)
        # dec_fuse3,output_fuse = self.decoder3(fuse0,dec_fuse2)

        # dec_sub1,output_middle_sub2 = self.decoder_sub1(sub3,cat2_sub)
        # dec_sub2,output_middle_sub1 = self.decoder_sub2(cat1_sub,dec_sub1)
        # dec_sub3,output_sub = self.decoder_sub3(sub0,dec_sub2)

        # # return output,output_middle1,output_middle2
        # if return_aux:
        #     output_middle_fuse2 = F.interpolate(output_middle_fuse2, size=imgs1.shape[2:])
        #     output_middle_fuse1 = F.interpolate(output_middle_fuse1, size=imgs1.shape[2:])
        #     output_fuse = F.interpolate(output_fuse, size=imgs1.shape[2:])
        #     output_fuse = torch.sigmoid(output_fuse)
        #     output_middle_fuse1 = torch.sigmoid(output_middle_fuse1)
        #     output_middle_fuse2 = torch.sigmoid(output_middle_fuse2)

        #     output_middle_sub2 = F.interpolate(output_middle_sub2, size=imgs1.shape[2:])
        #     output_middle_sub1 = F.interpolate(output_middle_sub1, size=imgs1.shape[2:])
        #     output_sub = F.interpolate(output_sub, size=imgs1.shape[2:])
        #     output_sub = torch.sigmoid(output_sub)
        #     output_middle_sub1 = torch.sigmoid(output_middle_sub1)
        #     output_middle_sub2 = torch.sigmoid(output_middle_sub2)

        #     pred_v = self.conv_out_v(feats_v[-1])
        #     pred_v = F.interpolate(pred_v, size=imgs1.shape[2:])
        #     pred_v = torch.sigmoid(pred_v)

        #     # pred_v2 = self.conv_out_v2(feats_v[1])
        #     # pred_v2 = F.interpolate(pred_v2, size=imgs1.shape[2:])
        #     # pred_v2 = torch.sigmoid(pred_v2)

        #     if labels is not None:
        #         # return output, output_middle1, output_middle2, pred_v1, pred_v2, loss_att
        #         return output_fuse, output_sub, output_middle_fuse1, output_middle_sub1, output_middle_fuse2, output_middle_sub2, pred_v, loss_att
        #     else:
        #         return output_fuse, output_sub, output_middle_fuse1, output_middle_sub1, output_middle_fuse2, output_middle_sub2, pred_v
        # else:
        #     output = F.interpolate(output, size=imgs1.shape[2:])
        #     output = torch.sigmoid(output)
        #     if labels is not None:
        #         return output, loss_att 
        #     else:
        #         return output

    def init_weights(self):
        self.global_consrative2.apply(init_weights)
        self.global_consrative1.apply(init_weights)
        self.local_consrative2.apply(init_weights)        
        self.local_consrative1.apply(init_weights) 

        self.encoder_v.apply(init_weights) 
        self.convs_video.apply(init_weights) 
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 
        # self.conv_out_v2.apply(init_weights) 
        # self.decoder_sub1.apply(init_weights) 
        # self.decoder_sub2.apply(init_weights) 
        # self.decoder_sub3.apply(init_weights) 

class TDANet_Resnet(nn.Module): 
    def __init__(self, num_classes=1, drop_rate=0., normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TDANet_Resnet, self).__init__()
        self.video_len = 8
        self.show_Feature_Maps = show_Feature_Maps
        self.enc_chs = (16,32)

        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        # self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.global_consrative2 = ContrastiveAtt(256)
        self.global_consrative1 = ContrastiveAtt(128)
        self.local_consrative1 = Local_interaction(64, window_size=8)
        
        self.encoder_v = VideoEncoder(in_ch=3, enc_chs=self.enc_chs)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in self.enc_chs
            ]
        )
        
        self.fusion3 = Fusion(128,64)
        self.fusion2 = Fusion(64,32)
        self.fusion1 = Fusion(32,16)

        self.decoder1 = DecBlock(64+64,32,num_classes) 
        self.decoder2 = DecBlock(32+32,16,num_classes)
        self.conv_out_v = Conv1x1(32, num_classes)
        
        if normal_init:
            self.init_weights()

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
    
    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        frames = self.pair_to_video(imgs1, imgs2) # b, 9, 3, 256, 256

        feats_v = self.encoder_v(frames.transpose(1,2))
        # print('feats_v',feats_v[0].shape) # b, 3, 9, 256, 256
        feats_v.pop(0)
        for i, feat in enumerate(feats_v):
            # print('self.tem_aggr(feat)',self.tem_aggr(feat).shape)
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))
            # print('The ',i,"th :",feats_v[i].shape)
            # The  0 th : torch.Size([2, 32, 64, 64])
            # The  1 th : torch.Size([2, 64, 32, 32])

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

        cur1_1, cur2_1 = self.local_consrative1(c1, c1_img2)

        if labels is not None:
            cur1_2, cur2_2, loss_att2 = self.global_consrative1(c2, c2_img2, labels)
            cur1_3, cur2_3, loss_att1 = self.global_consrative2(c3, c3_img2, labels)
            loss_att = loss_att1 + loss_att2
        else:
            cur1_2, cur2_2 = self.global_consrative1(c2, c2_img2)
            cur1_3, cur2_3 = self.global_consrative2(c3, c3_img2)

        fuse2 = self.fusion2(cur1_2,cur2_2) # [2, 32, 32, 32]
        fuse1 = self.fusion1(cur1_1,cur2_1) # [2, 16, 64, 64]

        fuse3 = self.fusion3(cur1_3,cur2_3) # [2, 64, 16, 16]
        cat2 = torch.cat([fuse2,feats_v[1]],1) # [2, 64, 32, 32]
        cat1 = torch.cat([fuse1,feats_v[0]],1) # [2, 32, 64, 64]

        dec1,output_middle1 = self.decoder1(cat2,fuse3)
        dec2,output = self.decoder2(cat1,dec1)

        if return_aux:
            output_middle1 = F.interpolate(output_middle1, size=imgs1.shape[2:])
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)

            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=imgs1.shape[2:])
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                # return output, output_middle1, output_middle2, pred_v1, pred_v2, loss_att
                return output, output_middle1, pred_v, loss_att
            else:
                return output, output_middle1, pred_v
        else:
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att
            else:
                return output
            
    def init_weights(self):
        self.global_consrative2.apply(init_weights)
        self.global_consrative1.apply(init_weights)  
        self.local_consrative1.apply(init_weights) 

        self.encoder_v.apply(init_weights) 
        self.convs_video.apply(init_weights) 
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

class CPAMDec_Mix_large(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix_large,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.loss_generator = nn.L1Loss()

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
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
        
class CPAMEnc_large(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc_large, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(1)
        # self.pool2 = nn.AdaptiveAvgPool2d(2)
        # self.pool3 = nn.AdaptiveAvgPool2d(3)
        # self.pool4 = nn.AdaptiveAvgPool2d(6)
        # self.pool5 = nn.AdaptiveAvgPool2d(9)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))
        # self.conv5 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
        #                         norm_layer(in_channels),
        #                         nn.ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        
        # feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        # feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        # feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        # feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        # feat5 = self.conv5(self.pool5(x)).view(b,c,-1)
        
        return self.conv1(x).view(b,c,-1) # torch.cat((feat1, feat2, feat3, feat4, feat5), 2)

class ContrastiveAtt_large(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt_large, self).__init__()

        inter_channels = in_channels

        self.cpam_enc_x = CPAMEnc_large(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc_large(inter_channels, norm_layer) # en_s

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
    
class ContrastiveAtt_Block(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
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

        self.attn = ContrastiveAtt_large(dim)

        self.pre_norm = pre_norm
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv_T(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
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
                x1, x2, loss_att = self.attn(t1, t2, labels)
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
            return t1, t2, loss_att
        else:
            return t1, t2

class DecBlock_large(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes=1):
        super().__init__()
        self.conv_fuse = DRAtt(in_ch, out_ch) # DRAtt
        self.conv_out = Conv1x1(out_ch, num_classes)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x2 = F.interpolate(x2, size=x1.shape[2:])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        out = self.conv_fuse(x)
        output = self.conv_out(out)
        return out, output
          
class TDANet_large(nn.Module): 
    def __init__(self, num_classes=1, drop_rate=0., normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TDANet_large, self).__init__()
        self.video_len = 8
        self.show_Feature_Maps = show_Feature_Maps
        enc_chs = (16,32)

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.global_consrative2 = ContrastiveAtt_large(192)
        self.global_consrative1 = ContrastiveAtt_large(80)
        self.local_consrative2 = Local_interaction(48, window_size=16)
        self.local_consrative1 = Local_interaction(32, window_size=16)
        
        self.encoder_v = VideoEncoder(in_ch=3, enc_chs=enc_chs, expansion=2)
        enc_chs = tuple(ch*self.encoder_v.expansion for ch in enc_chs)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in enc_chs
            ]
        )
        
        self.fusion3 = Fusion(96,128, reduction=False) #Difference
        self.fusion2 = Fusion(40,64, reduction=False)
        self.fusion1 = Fusion(24,32, reduction=False)
        self.fusion0 = Fusion(16,16, reduction=False)

        self.decoder1 = DecBlock(128+128, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.conv_out_v = Conv1x1(64, num_classes)
        
        if normal_init:
            self.init_weights()

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
    
    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        frames = self.pair_to_video(imgs1, imgs2) # b, 9, 3, 256, 256

        feats_v = self.encoder_v(frames.transpose(1,2))
        feats_v.pop(0)
        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

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
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        cur1_0, cur2_0 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
        cur1_1, cur2_1 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64

        if labels is not None:
            cur1_2, cur2_2, loss_att2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3, loss_att1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) # 192, 16 , 16 -> 64, 16 , 16
            loss_att = loss_att1 + loss_att2
        else:
            cur1_2, cur2_2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16

        fuse2 = self.fusion2(cur1_2,cur2_2) 
        fuse1 = self.fusion1(cur1_1,cur2_1)

        fuse3 = self.fusion3(cur1_3,cur2_3) 
        cat2 = torch.cat([fuse2,feats_v[1]],1) 
        cat1 = torch.cat([fuse1,feats_v[0]],1) 
        fuse0 = self.fusion0(cur1_0,cur2_0) 

        dec1,output_middle2 = self.decoder1(cat2,fuse3)
        dec2,output_middle1 = self.decoder2(cat1,dec1)
        dec3,output = self.decoder3(fuse0,dec2)

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=imgs1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=imgs1.shape[2:])
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=imgs1.shape[2:])

            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)

            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att
            else:
                return output

    def init_weights(self):
        self.global_consrative2.apply(init_weights)
        self.global_consrative1.apply(init_weights)
        self.local_consrative2.apply(init_weights)        
        self.local_consrative1.apply(init_weights) 

        self.encoder_v.apply(init_weights) 
        self.convs_video.apply(init_weights) 
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

class TDANet_large2D(nn.Module): 
    def __init__(self, num_classes=1, drop_rate=0., normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TDANet_large2D, self).__init__()
        self.video_len = 8
        self.show_Feature_Maps = show_Feature_Maps
        enc_chs = (32,64)
        self.expansion = 1
        self.latent_dim = 6

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.global_consrative2 = ContrastiveAtt_large(192)
        self.global_consrative1 = ContrastiveAtt_large(80)
        self.local_consrative2 = Local_interaction(48, window_size=16)
        self.local_consrative1 = Local_interaction(32, window_size=16)
        
        # self.encoder_v = VideoEncoder(in_ch=3, enc_chs=enc_chs, expansion=2)
        self.encoder_v = TMEncoder(3*(self.video_len+1), enc_chs=enc_chs, expansion=self.expansion)
        enc_chs = tuple(ch*self.expansion for ch in enc_chs)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in enc_chs
            ]
        )
        
        self.fusion3 = Fusion(96,128, reduction=False) #Difference
        self.fusion2 = Fusion(40,64, reduction=False)
        self.fusion1 = Fusion(24,32, reduction=False)
        self.fusion0 = Fusion(16,16, reduction=False)

        self.decoder1 = DecBlock(128+128, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.conv_out_v = Conv1x1(64, num_classes)
        
        if normal_init:
            self.init_weights()

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
        frame = rearrange(frames, "n l c h w -> n (l c) h w")
        return frame

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)
    
    def channel_shuffle(self, x, groups=3):
        batch, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
        x = x.view(batch, groups, channels_per_group, height, width)
        # channel shuffle, 通道洗牌
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, -1, height, width)
        return x
    
    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        # frames = self.pair_to_video(imgs1, imgs2) # b, 9, 3, 256, 256

        # feats_v = self.encoder_v(frames.transpose(1,2))
        # feats_v.pop(0)
        # for i, feat in enumerate(feats_v):
        #     feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        frames = self.pair_to_video(imgs1, imgs2)  # [1, 24, 256, 256]
        frame = self.channel_shuffle(frames, groups=self.video_len+1)

        feats_v = self.encoder_v(frames)
        feats_v.pop(0)

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
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        cur1_0, cur2_0 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
        cur1_1, cur2_1 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64

        if labels is not None:
            cur1_2, cur2_2, loss_att2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3, loss_att1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) # 192, 16 , 16 -> 64, 16 , 16
            loss_att = loss_att1 + loss_att2
        else:
            cur1_2, cur2_2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16

        fuse2 = self.fusion2(cur1_2,cur2_2) 
        fuse1 = self.fusion1(cur1_1,cur2_1)

        fuse3 = self.fusion3(cur1_3,cur2_3) 
        cat2 = torch.cat([fuse2,feats_v[1]],1) 
        cat1 = torch.cat([fuse1,feats_v[0]],1) 
        fuse0 = self.fusion0(cur1_0,cur2_0) 

        dec1,output_middle2 = self.decoder1(cat2,fuse3)
        dec2,output_middle1 = self.decoder2(cat1,dec1)
        dec3,output = self.decoder3(fuse0,dec2)

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)

            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att
            else:
                return output

    def init_weights(self):
        self.global_consrative2.apply(init_weights)
        self.global_consrative1.apply(init_weights)
        self.local_consrative2.apply(init_weights)        
        self.local_consrative1.apply(init_weights) 

        self.encoder_v.apply(init_weights) 
        self.convs_video.apply(init_weights) 
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
CE = torch.nn.BCELoss(reduction='sum')
cos_sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        # self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        # self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        # self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        # self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.channel = channels

        self.fc1_image1 = nn.Linear(channels * 1 * 8 * 8, latent_size)
        self.fc2_image1 = nn.Linear(channels * 1 * 8 * 8, latent_size)
        self.fc1_image2_1 = nn.Linear(channels * 1 * 8 * 8, latent_size)
        self.fc2_image2_1 = nn.Linear(channels * 1 * 8 * 8, latent_size)

        self.fc1_image2 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_image2 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc1_image2_2 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_image2_2 = nn.Linear(channels * 1 * 16 * 16, latent_size)

        self.fc1_image3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc2_image3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc1_image2_3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc2_image2_3 = nn.Linear(channels * 1 * 4 * 4, latent_size)

        self.fc1_image4 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_image4 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc1_image2_4 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_image2_4 = nn.Linear(channels * 1 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, image_feat, image2__feat):
        image_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(image_feat))))
        image2__feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(image2__feat))))
        # print(image_feat.size())
        # print(image2__feat.size())
        if image_feat.shape[2] == 4:
            image_feat = image_feat.view(-1, self.channel * 1 * 4 * 4)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 4 * 4)
            mu_image = self.fc1_image3(image_feat)
            logvar_image = self.fc2_image3(image_feat)
            mu_image2_ = self.fc1_image2_3(image2__feat)
            logvar_image2_ = self.fc2_image2_3(image2__feat)
        elif image_feat.shape[2] == 8:
            image_feat = image_feat.view(-1, self.channel * 1 * 8 * 8)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 8 * 8)
            mu_image = self.fc1_image1(image_feat)
            logvar_image = self.fc2_image1(image_feat)
            mu_image2_ = self.fc1_image2_1(image2__feat)
            logvar_image2_ = self.fc2_image2_1(image2__feat)
        elif image_feat.shape[2] == 16:
            image_feat = image_feat.view(-1, self.channel * 1 * 16 * 16)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 16 * 16)
            mu_image = self.fc1_image2(image_feat)
            logvar_image = self.fc2_image2(image_feat)
            mu_image2_ = self.fc1_image2_2(image2__feat)
            logvar_image2_ = self.fc2_image2_2(image2__feat)
        elif image_feat.shape[2] == 32:
            image_feat = image_feat.view(-1, self.channel * 1 * 32 * 32)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 32 * 32)
            mu_image = self.fc1_image4(image_feat)
            logvar_image = self.fc2_image4(image_feat)
            mu_image2_ = self.fc1_image2_4(image2__feat)
            logvar_image2_ = self.fc2_image2_4(image2__feat)

        mu_image2_ = self.tanh(mu_image2_)
        mu_image = self.tanh(mu_image)
        logvar_image2_ = self.tanh(logvar_image2_)
        logvar_image = self.tanh(logvar_image)
        z_image = self.reparametrize(mu_image, logvar_image)
        dist_image = Independent(Normal(loc=mu_image, scale=torch.exp(logvar_image)), 1)
        z_image2_ = self.reparametrize(mu_image2_, logvar_image2_)
        dist_image2_ = Independent(Normal(loc=mu_image2_, scale=torch.exp(logvar_image2_)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_image, dist_image2_)) + torch.mean(
            self.kl_divergence(dist_image2_, dist_image))
        z_image_norm = torch.sigmoid(z_image)
        z_image2__norm = torch.sigmoid(z_image2_)
        ce_image_image2_ = CE(z_image_norm,z_image2__norm.detach())
        ce_image2__image = CE(z_image2__norm, z_image_norm.detach())
        latent_loss = ce_image_image2_+ce_image2__image-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_image,z_image2_)).sum()

        return latent_loss, z_image, z_image2_

class Mutual_info_reg2(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Mutual_info_reg2, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        self.fc1_image1 = nn.Linear(channels * 1 * 8 * 8, latent_size)
        self.fc2_image1 = nn.Linear(channels * 1 * 8 * 8, latent_size)
        self.fc1_image2_1 = nn.Linear(channels * 1 * 8 * 8, latent_size)
        self.fc2_image2_1 = nn.Linear(channels * 1 * 8 * 8, latent_size)

        self.fc1_image2 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_image2 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc1_image2_2 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_image2_2 = nn.Linear(channels * 1 * 16 * 16, latent_size)

        self.fc1_image3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc2_image3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc1_image2_3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc2_image2_3 = nn.Linear(channels * 1 * 4 * 4, latent_size)

        self.fc1_image4 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_image4 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc1_image2_4 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_image2_4 = nn.Linear(channels * 1 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, image_feat, image2__feat):
        image_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(image_feat))))
        image2__feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(image2__feat))))
        # print(image_feat.size())
        # print(image2__feat.size())
        if image_feat.shape[2] == 4:
            image_feat = image_feat.view(-1, self.channel * 1 * 4 * 4)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 4 * 4)
            mu_image = self.fc1_image3(image_feat)
            logvar_image = self.fc2_image3(image_feat)
            mu_image2_ = self.fc1_image2_3(image2__feat)
            logvar_image2_ = self.fc2_image2_3(image2__feat)
        elif image_feat.shape[2] == 8:
            image_feat = image_feat.view(-1, self.channel * 1 * 8 * 8)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 8 * 8)
            mu_image = self.fc1_image1(image_feat)
            logvar_image = self.fc2_image1(image_feat)
            mu_image2_ = self.fc1_image2_1(image2__feat)
            logvar_image2_ = self.fc2_image2_1(image2__feat)
        elif image_feat.shape[2] == 16:
            image_feat = image_feat.view(-1, self.channel * 1 * 16 * 16)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 16 * 16)
            mu_image = self.fc1_image2(image_feat)
            logvar_image = self.fc2_image2(image_feat)
            mu_image2_ = self.fc1_image2_2(image2__feat)
            logvar_image2_ = self.fc2_image2_2(image2__feat)
        elif image_feat.shape[2] == 32:
            image_feat = image_feat.view(-1, self.channel * 1 * 32 * 32)
            image2__feat = image2__feat.view(-1, self.channel * 1 * 32 * 32)
            mu_image = self.fc1_image4(image_feat)
            logvar_image = self.fc2_image4(image_feat)
            mu_image2_ = self.fc1_image2_4(image2__feat)
            logvar_image2_ = self.fc2_image2_4(image2__feat)

        mu_image2_ = self.tanh(mu_image2_)
        mu_image = self.tanh(mu_image)
        logvar_image2_ = self.tanh(logvar_image2_)
        logvar_image = self.tanh(logvar_image)
        z_image = self.reparametrize(mu_image, logvar_image)
        dist_image = Independent(Normal(loc=mu_image, scale=torch.exp(logvar_image)), 1)
        z_image2_ = self.reparametrize(mu_image2_, logvar_image2_)
        dist_image2_ = Independent(Normal(loc=mu_image2_, scale=torch.exp(logvar_image2_)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_image, dist_image2_)) + torch.mean(
            self.kl_divergence(dist_image2_, dist_image))
        z_image_norm = torch.sigmoid(z_image)
        z_image2__norm = torch.sigmoid(z_image2_)
        ce_image_image2_ = CE(z_image_norm,z_image2__norm.detach())
        ce_image2__image = CE(z_image2__norm, z_image_norm.detach())
        latent_loss = ce_image_image2_+ce_image2__image-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_image,z_image2_)).sum()

        return latent_loss, z_image, z_image2_

class CPAMDec_Mix_large_mul(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix_large_mul,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.loss_generator = nn.L1Loss()

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//2, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

        self.latent_dim = 6
        self.Mutual_info_reg = Mutual_info_reg(in_channels,in_channels,self.latent_dim)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)
    
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
         
        lat_loss, z1, z2 = self.Mutual_info_reg(out1,out2)

        # if z1!=None:
        #     z1 = torch.unsqueeze(z1, 2)
        #     z1 = self.tile(z1, 2, width)
        #     z1 = torch.unsqueeze(z1, 3)
        #     z1 = self.tile(z1, 3, height)

        #     z2 = torch.unsqueeze(z2, 2)
        #     z2 = self.tile(z2, 2, width)
        #     z2 = torch.unsqueeze(z2, 3)
        #     z2 = self.tile(z2, 3, height)

        #     out1 = torch.cat([out1,z1],1)
        #     out2 = torch.cat([out2,z2],1)

        if label is not None:
            label = F.interpolate(label, size=(width,height))
            att = attention.permute(0,2,1).view(m_batchsize,K,width,height)
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)
            return out1, out2, lat_loss, loss_att
        else:
            return out1, out2, lat_loss
        
class ContrastiveAtt_large_mul(nn.Module):
    def __init__(self, in_channels, out_channels=32, norm_layer=nn.BatchNorm2d):
        super(ContrastiveAtt_large_mul, self).__init__()

        inter_channels = in_channels//2

        self.conv_cpam_b_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.GELU()) # conv5_s
        self.conv_cpam_b_y = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.GELU()) # conv5_s

        self.cpam_enc_x = CPAMEnc_large(inter_channels, norm_layer) # en_s
        self.cpam_enc_y = CPAMEnc_large(inter_channels, norm_layer) # en_s

        self.cpam_dec_mix = CPAMDec_Mix_large_mul(inter_channels) # de_s
        
    def forward(self, x, y, label=None):
        cpam_b_x = self.conv_cpam_b_x(x)
        cpam_f_x = self.cpam_enc_x(cpam_b_x).permute(0,2,1)

        cpam_b_y = self.conv_cpam_b_y(y)
        cpam_f_y = self.cpam_enc_y(cpam_b_y).permute(0,2,1)

        if label is not None:
            cpam_feat1, cpam_feat2, loss_lat, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y,label) 
            return cpam_feat1, cpam_feat2, loss_lat, loss_att
        else: 
            cpam_feat1, cpam_feat2, loss_lat = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y) 
            return cpam_feat1, cpam_feat2, loss_lat

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
    
class Local_interaction_mul(nn.Module):
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

        inter_channels = in_channels//2

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

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        
        self.latent_dim = 6
        self.Mutual_info_reg2 = Mutual_info_reg2(inter_channels,inter_channels,self.latent_dim)
        
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
    
    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)
    
    def forward(self, x1, x2):
        x1 = self.init_conv1(x1)
        x2 = self.init_conv2(x2)

        B, C, H, W = x1.shape
        q1,k1,v1, q2,k2,v2 = self.get_qkv(x1,x2)
        x1_interact, x2_interact = self.forward_interaction_local(q1, k1, v1, q2, k2, v2, H, W)

        x1_interact = x1_interact + self.scale*x1 
        # add-self.scale*x1_interact + x1: TDANet-0523-GZ： 84.67
        # Without: TDANet-0525-GZ：0.85622
        # x1_interact + self.scale*x1: TDANet-0524-GZ：85.115
        x2_interact = x2_interact + self.scale*x2  

        lat_loss, z1, z2 = self.Mutual_info_reg2(x1_interact,x2_interact)

        if z1!=None:
            z1 = torch.unsqueeze(z1, 2)
            z1 = self.tile(z1, 2, W)
            z1 = torch.unsqueeze(z1, 3)
            z1 = self.tile(z1, 3, H)

            z2 = torch.unsqueeze(z2, 2)
            z2 = self.tile(z2, 2, W)
            z2 = torch.unsqueeze(z2, 3)
            z2 = self.tile(z2, 3, H)

            x1_interact = torch.cat([x1_interact,z1],1)
            x2_interact = torch.cat([x2_interact,z2],1)

        return x1_interact, x2_interact, lat_loss

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))

class TMEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64, 128), expansion=1):
        super(TMEncoder, self).__init__()

        self.n_layers = 2
        self.expansion = expansion
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            # nn.Conv2d(in_ch, enc_chs[0], kernel_size=(9, 9), stride=(4, 4), padding=(4, 4), bias=False),
            nn.Conv2d(in_ch, enc_chs[0], kernel_size=(9, 9), stride=(4, 4), padding=(4, 4), bias=False),
            nn.Conv2d(enc_chs[0], enc_chs[0], 3, 1, 1, bias=True, groups=enc_chs[0]),
            nn.BatchNorm2d(enc_chs[0]),
            nn.GELU()
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock(
                enc_chs[0],
                enc_chs[0]*exps,
            ),
            nn.Conv2d(enc_chs[0]*exps, enc_chs[0]*exps, 3, 1, 1, bias=True, groups=enc_chs[0]*exps),
            # LSKAtt(enc_chs[0]*exps),
            nn.BatchNorm2d(enc_chs[0]*exps),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            ResBlock(
                enc_chs[0]*exps,
                enc_chs[1]*exps,
            ),
            # nn.Conv2d(enc_chs[1] * exps, enc_chs[1] * exps, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.Conv2d(enc_chs[1] * exps, enc_chs[1] * exps, kernel_size=(2, 2), stride=(2, 2), bias=True, groups=enc_chs[1]*exps),
            # LSKAtt(enc_chs[1]*exps),
            nn.BatchNorm2d(enc_chs[1]*exps),
            nn.GELU()
        )

        # self.layer1 = nn.Sequential(
        #     ResBlock(
        #         enc_chs[0],
        #         enc_chs[0]*exps,
        #     ),
        #     LSKAtt(enc_chs[0]*exps),
        #     # nn.Conv2d(enc_chs[0]*exps, enc_chs[0]*exps, 3, 1, 1, bias=True, groups=enc_chs[0]*exps),
        #     nn.BatchNorm2d(enc_chs[0]*exps),
        #     nn.GELU()
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(enc_chs[0]*exps, enc_chs[0]*exps, kernel_size=(2, 2), stride=(2, 2), bias=True, groups=enc_chs[0]*exps),
        #     ResBlock(
        #         enc_chs[0]*exps,
        #         enc_chs[1]*exps,
        #     ),
        #     LSKAtt(enc_chs[1]*exps),
        #     nn.BatchNorm2d(enc_chs[1]*exps),
        #     nn.GELU()
        # )

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        # print(x.shape)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            # print(x.shape)
            feats.append(x)

        return feats
                 
class TDANet_large_mul(nn.Module): # 0625
    def __init__(self, num_classes=1, drop_rate=0., normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TDANet_large_mul, self).__init__()
        self.video_len = 8
        self.show_Feature_Maps = show_Feature_Maps
        enc_chs = (32,64)
        self.expansion = 1
        self.latent_dim = 6

        self.model = ghostnetv2()
        params=self.model.state_dict()
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.global_consrative2 = ContrastiveAtt_large_mul(192)
        self.global_consrative1 = ContrastiveAtt_large_mul(80)
        self.local_consrative2 = Local_interaction(48, window_size=16)
        self.local_consrative1 = Local_interaction(32, window_size=16)
        
        # self.encoder_v = VideoEncoder(in_ch=3, enc_chs=enc_chs, expansion=2)
        self.encoder_v = TMEncoder(3*(self.video_len+1), enc_chs=enc_chs, expansion=self.expansion)
        enc_chs = tuple(ch*self.expansion for ch in enc_chs)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in enc_chs
            ]
        )
        
        self.fusion3 = Fusion(96,128, reduction=False) #Difference +self.latent_dim
        self.fusion2 = Fusion(40,64, reduction=False)
        self.fusion1 = Fusion(24,32, reduction=False)
        self.fusion0 = Fusion(16,16, reduction=False)

        self.decoder1 = DecBlock(128+64+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+32+32, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.conv_out_v = Conv1x1(64, num_classes)
        
        if normal_init:
            self.init_weights()

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
        frame = rearrange(frames, "n l c h w -> n (l c) h w")
        return frame

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)
    
    def channel_shuffle(self, x, groups=3):
        batch, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
        x = x.view(batch, groups, channels_per_group, height, width)
        # channel shuffle, 通道洗牌
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, -1, height, width)
        return x
    
    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        # frames = self.pair_to_video(imgs1, imgs2) # b, 9, 3, 256, 256

        # feats_v = self.encoder_v(frames.transpose(1,2))
        # feats_v.pop(0)
        # for i, feat in enumerate(feats_v):
        #     feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        frames = self.pair_to_video(imgs1, imgs2)  # [1, 24, 256, 256]
        # frame = self.channel_shuffle(frames, groups=self.video_len+1)

        feats_v = self.encoder_v(frames)
        feats_v.pop(0)

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
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16

        cur1_0, cur2_0 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
        cur1_1, cur2_1 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64

        if labels is not None:
            cur1_2, cur2_2, lat_loss2, loss_att2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3, lat_loss1, loss_att1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) # 192, 16 , 16 -> 64, 16 , 16
            loss_lat = lat_loss1 + lat_loss2 #+ lat_loss_local0 + lat_loss_local1
            loss_att = loss_att1 + loss_att2
        else:
            cur1_2, cur2_2, lat_loss2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3, lat_loss1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16
            loss_lat = lat_loss1 + lat_loss2 #+ lat_loss_local0 + lat_loss_local1

        fuse2 = self.fusion2(cur1_2,cur2_2) 
        fuse1 = self.fusion1(cur1_1,cur2_1)

        fuse3 = self.fusion3(cur1_3,cur2_3) 
        cat2 = torch.cat([fuse2,feats_v[1]],1) 
        cat1 = torch.cat([fuse1,feats_v[0]],1) 
        fuse0 = self.fusion0(cur1_0,cur2_0) 

        dec1, output_middle2 = self.decoder1(cat2,fuse3)
        dec2, output_middle1 = self.decoder2(cat1,dec1)
        dec3, output = self.decoder3(fuse0,dec2)

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att, loss_lat
            else:
                return output, output_middle1, output_middle2, pred_v, loss_lat
        else:
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att
            else:
                return output

    def init_weights(self):
        self.global_consrative2.apply(init_weights)
        self.global_consrative1.apply(init_weights)
        self.local_consrative2.apply(init_weights)        
        self.local_consrative1.apply(init_weights) 

        self.encoder_v.apply(init_weights) 
        self.convs_video.apply(init_weights) 
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)
    
class Conv_interaction(nn.Module):
    def __init__(self, channel=16, kernerl_size=5):
        super(Conv_interaction, self).__init__()
        self.kernerl_size = kernerl_size
        self.fuse4 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True) 
        self.smooth4 = DSConv3x3(channel, channel, stride=1, dilation=1)  # 96channel-> 96channel

        self.fuse3 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)
        self.smooth3 = DSConv3x3(channel, channel, stride=1, dilation=1)  # 32channel-> 96channel

        self.kernel_img1 = DSConv3x3(channel, channel, stride=1)
        self.kernel_img2 = DSConv3x3(channel, channel, stride=1)

        self.pool = nn.AdaptiveAvgPool2d(kernerl_size)

        # self.ChannelCorrelation = CCorrM(channel)

    def forward(self, img1, img2):  # x4:96*18*18 k4:96*5*5; x3:32*36*36 k3:32*5*5
        kernel1 = self.pool(self.kernel_img1(img1))
        kernel2 = self.pool(self.kernel_img2(img2))

        B, C, H, W = kernel1.size()
        pad = (self.kernerl_size-1)//2

        x_B, x_C, x_H, x_W = img1.size()  # 8*96*18*18

        img1_new = img1.clone()
        img2_new = img2.clone()
        for i in range(1, B):

            kernel4 = kernel1[i, :, :, :]
            kernel3 = kernel2[i, :, :, :]
            kernel4 = kernel4.view(C, 1, H, W)
            kernel3 = kernel3.view(C, 1, H, W)

            # DDconv
            img1_r1 = F.conv2d(img1[i, :, :, :].view(1, C, x_H, x_W), kernel4, stride=1, padding=pad, dilation=1,
                             groups=C)
            img1_r2 = F.conv2d(img1[i, :, :, :].view(1, C, x_H, x_W), kernel4, stride=1, padding=pad*2, dilation=2,
                             groups=C)
            img1_r3 = F.conv2d(img1[i, :, :, :].view(1, C, x_H, x_W), kernel4, stride=1, padding=pad*3, dilation=3,
                             groups=C)
            img1_new[i, :, :, :] = img1_r1 + img1_r2 + img1_r3

            # DDconv
            img2_r1 = F.conv2d(img2[i, :, :, :].view(1, C, x_H, x_W), kernel3, stride=1, padding=pad, dilation=1,
                             groups=C)
            img2_r2 = F.conv2d(img2[i, :, :, :].view(1, C, x_H, x_W), kernel3, stride=1, padding=pad*2, dilation=2,
                             groups=C)
            img2_r3 = F.conv2d(img2[i, :, :, :].view(1, C, x_H, x_W), kernel3, stride=1, padding=pad*3, dilation=3,
                             groups=C)
            img2_new[i, :, :, :] = img2_r1 + img2_r2 + img2_r3

        # Pconv
        img1_all = self.fuse4(img1_new)
        img1_smooth = self.smooth4(img1_all)
        # Pconv
        img2_all = self.fuse3(img2_new)
        img2_smooth = self.smooth3(img2_all)

        return img1_smooth, img2_smooth 

# class Local_LSKAtt(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_img1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial_img1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1_img1 = nn.Conv2d(dim, dim//2, 1)
#         self.conv2_img1 = nn.Conv2d(dim, dim//2, 1)
#         self.conv_squeeze_img1 = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv_img1 = nn.Conv2d(dim//2, dim, 1)

#         self.conv0_img2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial_img2 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1_img2 = nn.Conv2d(dim, dim//2, 1)
#         self.conv2_img2 = nn.Conv2d(dim, dim//2, 1)
#         self.conv_squeeze_img2 = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv_img2 = nn.Conv2d(dim//2, dim, 1)

#         self.loss_generator = nn.L1Loss()
#         self.scale = nn.Parameter(torch.zeros(1))

#     def forward(self, x1, x2, labels = None):   
#         attn1_img1 = self.conv0_img1(x1)
#         attn2_img1 = self.conv_spatial_img1(attn1_img1)
#         attn1_img1 = self.conv1_img1(attn1_img1)
#         attn2_img1 = self.conv2_img1(attn2_img1)
#         attn_img1 = torch.cat([attn1_img1, attn2_img1], dim=1)
#         avg_attn_img1 = torch.mean(attn_img1, dim=1, keepdim=True)
#         max_attn_img1, _ = torch.max(attn_img1, dim=1, keepdim=True)
#         agg_img1 = torch.cat([avg_attn_img1, max_attn_img1], dim=1)
#         sig_img1 = self.conv_squeeze_img1(agg_img1).sigmoid()
#         attn_img1 = attn1_img1 * sig_img1[:,0,:,:].unsqueeze(1) + attn2_img1 * sig_img1[:,1,:,:].unsqueeze(1)
#         attn_img1 = self.conv_img1(attn_img1)

#         attn1_img2 = self.conv0_img2(x2)
#         attn2_img2 = self.conv_spatial_img2(attn1_img2)
#         attn1_img2 = self.conv1_img2(attn1_img2)
#         attn2_img2 = self.conv2_img2(attn2_img2)
#         attn_img2 = torch.cat([attn1_img2, attn2_img2], dim=1)
#         avg_attn_img2 = torch.mean(attn_img2, dim=1, keepdim=True)
#         max_attn_img2, _ = torch.max(attn_img2, dim=1, keepdim=True)
#         agg_img2 = torch.cat([avg_attn_img2, max_attn_img2], dim=1)
#         sig_img2 = self.conv_squeeze_img2(agg_img2).sigmoid()
#         attn_img2 = attn1_img2 * sig_img2[:,0,:,:].unsqueeze(1) + attn2_img2 * sig_img2[:,1,:,:].unsqueeze(1)
#         attn_img2 = self.conv_img2(attn_img2)

#         att = torch.abs(attn_img1-attn_img2).sigmoid()
#         x1 = x1 * att # (att + self.scale*attn_img1) 
#         x2 = x2 * att # (att + self.scale*attn_img2) #att # 

#         if labels is not None:
#             m_batchsize, C, width, height = x1.size()
#             label = F.interpolate(labels, size=(width,height))
#             loss_att = self.loss_generator(torch.mean(att,dim=1),label)
#             return x1, x2, loss_att
#         else:
#             return x1, x2
        
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

class LSKAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0_img1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_img1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_img1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2_img1 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze_img1 = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_img1 = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x1):   
        attn1_img1 = self.conv0_img1(x1)
        attn2_img1 = self.conv_spatial_img1(attn1_img1)
        attn1_img1 = self.conv1_img1(attn1_img1)
        attn2_img1 = self.conv2_img1(attn2_img1)
        attn_img1 = torch.cat([attn1_img1, attn2_img1], dim=1)
        avg_attn_img1 = torch.mean(attn_img1, dim=1, keepdim=True)
        max_attn_img1, _ = torch.max(attn_img1, dim=1, keepdim=True)
        agg_img1 = torch.cat([avg_attn_img1, max_attn_img1], dim=1)
        sig_img1 = self.conv_squeeze_img1(agg_img1).sigmoid()

        attn_img1 = attn1_img1 * sig_img1[:,0,:,:].unsqueeze(1) + attn2_img1 * sig_img1[:,1,:,:].unsqueeze(1)
        attn_img1 = self.conv_img1(attn_img1)
        x1 = x1 * attn_img1

        return x1
        
class Local_Block(nn.Module):
    def __init__(self, in_channels, window_size=8, drop_path=0.1, mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
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
        self.attn = Local_LSKAtt(dim)

        self.pre_norm = pre_norm
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv_T(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
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
                x1, x2, loss_att = self.attn(t1, t2, labels)
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
            return t1, t2, loss_att
        else:
            return t1, t2

class TDANet_3D_ALL_GLOBAL(nn.Module): 
    def __init__(self, num_classes=1, drop_rate=0., normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(TDANet_3D_ALL_GLOBAL, self).__init__()
        self.video_len = 8
        self.show_Feature_Maps = show_Feature_Maps
        enc_chs = (16,32)
        # enc_chs = (32,64)
        # self.expansion = 1
        # self.latent_dim = 6

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.global_consrative2 = ContrastiveAtt_Block(192) # Local_Block(192)# 
        self.global_consrative1 = ContrastiveAtt_Block(80) # Local_Block(80)# 
        self.local_consrative2 = ContrastiveAtt_Block(48)# Local_Block(48, window_size=16) # Local_Block(48, window_size=16)
        self.local_consrative1 = ContrastiveAtt_Block(32)# Local_Block(32, window_size=16)
        
        self.encoder_v = VideoEncoder(in_ch=3, enc_chs=enc_chs, expansion=2)
        # self.encoder_v = TMEncoder(3*(self.video_len+1), enc_chs=enc_chs, expansion=self.expansion)
        enc_chs = tuple(ch*self.encoder_v.expansion for ch in enc_chs)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2*ch, ch, norm=True, act=True)
                for ch in enc_chs
            ]
        )
        
        self.fusion3 = Fusion(192,128,reduction=False) #Difference
        self.fusion2 = Fusion(80,64,reduction=False)
        self.fusion1 = Fusion(48,32,reduction=False)
        self.fusion0 = Fusion(32,16,reduction=False)

        self.decoder1 = DecBlock(128+128, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.conv_out_v = Conv1x1(64, num_classes)
        
        if normal_init:
            self.init_weights()

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
        # frame = rearrange(frames, "n l c h w -> n (l c) h w")
        return frames

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)
    
    def channel_shuffle(self, x, groups=3):
        batch, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
        x = x.view(batch, groups, channels_per_group, height, width)
        # channel shuffle, 通道洗牌
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, -1, height, width)
        return x
    
    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        frames = self.pair_to_video(imgs1, imgs2) # b, 9, 3, 256, 256

        feats_v = self.encoder_v(frames.transpose(1,2))
        feats_v.pop(0)
        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        # frames = self.pair_to_video(imgs1, imgs2)  # [1, 24, 256, 256]
        # # frames = self.channel_shuffle(frames, groups=self.video_len+1)
        # feats_v = self.encoder_v(frames)
        # feats_v.pop(0)

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
        c6_img2 = self.model.blocks[5](c5_img2) # 80, 16, 16
        c7_img2 = self.model.blocks[6](c6_img2) # 112, 16, 16


        if labels is not None:
            cur1_0, cur2_0, loss_local_att1 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1),labels) # 16, 128 , 128 -> 8, 128 , 128
            cur1_1, cur2_1, loss_local_att2 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1),labels) # 48, 64 , 64 -> 24, 64 , 64
            cur1_2, cur2_2, loss_att2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3, loss_att1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) # 192, 16 , 16 -> 64, 16 , 16
            loss_att = loss_att1 + loss_att2 + 0.4*(loss_local_att1 + loss_local_att2)
        else:
            cur1_0, cur2_0 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) # 16, 128 , 128 -> 8, 128 , 128
            cur1_1, cur2_1 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 24, 64 , 64
            cur1_2, cur2_2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 32, 32 , 32
            cur1_3, cur2_3 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 64, 16 , 16

        fuse2 = self.fusion2(cur1_2,cur2_2) 
        fuse1 = self.fusion1(cur1_1,cur2_1)

        fuse3 = self.fusion3(cur1_3,cur2_3) 
        cat2 = torch.cat([fuse2,feats_v[1]],1) 
        cat1 = torch.cat([fuse1,feats_v[0]],1) 
        fuse0 = self.fusion0(cur1_0,cur2_0) 

        dec1,output_middle2 = self.decoder1(cat2,fuse3)
        dec2,output_middle1 = self.decoder2(cat1,dec1)
        dec3,output = self.decoder3(fuse0,dec2)

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=imgs1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att
            else:
                return output

    def init_weights(self):
        self.global_consrative2.apply(init_weights)
        self.global_consrative1.apply(init_weights)
        # self.local_consrative2.apply(init_weights)
        # self.local_consrative1.apply(init_weights)

        self.encoder_v.apply(init_weights) 
        self.convs_video.apply(init_weights) 
        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 
