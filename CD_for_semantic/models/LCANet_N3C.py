import torch
import torch.nn as nn
import math
from .GhostNetv2 import ghostnetv2
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
    
class ITCA(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(ITCA,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.ratio = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2
        # self.conv_value1_2 = DWConv(in_channels)

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//4) # key_conv2
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

        # proj_key = torch.abs(proj_key1-proj_key2)

        energy1 =  torch.bmm(proj_query,proj_key1)
        energy2 =  torch.bmm(proj_query,proj_key2)

        energy = torch.abs(energy1-energy2)
        # energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy) 

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1))
        out1 = out1.view(m_batchsize,C,width,height)
        out1 = self.scale*out1 + x1 # self.conv_value1_2(x1) 

        out2 = torch.bmm(proj_value2,attention.permute(0,2,1))
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + x2 # self.conv_value2_2(x2) 

        norm = nn.functional.normalize(energy, p=1, dim=1)
        norm = (norm-norm.min())/(norm.max()-norm.min())
        # # ratio = self.ratio
        # # if ratio > 0.5: ratio = 0.5
        # # elif ratio < 0: ratio = 0
        # cmask = (torch.sign(norm - 0.3) + 1) / 2

        # # norm = (attention-attention.min())/(attention.max()-attention.min())
        cmask = (torch.sign(norm-0.3) + 1) / 2
        out_res1 = torch.bmm(proj_value1,(1-cmask).permute(0,2,1))
        out_res2 = torch.bmm(proj_value2,(1-cmask).permute(0,2,1))

        res_loss = 0
        loss_att = 0 
        for i in range(m_batchsize):
            nc_sum = ((1-cmask[i]) == 1).sum()
            # nc_ratio = (nc_sum) / (width*height * K)
            # print("nc_ratio:")
            # print(nc_ratio)
            res_loss += self.loss_generator(out_res1[i], out_res2[i]) * (width*height * K) / (nc_sum+1)
            loss_att += torch.mean(abs(cmask[i]))
        res_loss = res_loss / m_batchsize
        loss_att = loss_att / m_batchsize

        return out1, out2, res_loss, loss_att

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

        cpam_feat1, cpam_feat2, loss_res, loss_att = self.cpam_dec_mix(cpam_b_x,cpam_f_x,cpam_b_y,cpam_f_y)  #, loss_res, loss_att

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2, loss_res, loss_att

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

        cross_result3, cur1_3, cur2_3, loss_res3, loss_att3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        cross_result2, cur1_2, cur2_2, loss_res2, loss_att2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        cross_result1, cur1_1, cur2_1, loss_res1, loss_att1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        # cross_result3, cur1_3, cur2_3 = self.consrative3(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) # 48, 64 , 64 -> 32
        # cross_result2, cur1_2, cur2_2 = self.consrative2(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) # 80, 32 , 32 -> 64
        # cross_result1, cur1_1, cur2_1 = self.consrative1(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) # 192, 16 , 16 -> 128

        loss_res = loss_res3 + loss_res2 + loss_res1
        loss_att = loss_att3 + loss_att2 + loss_att1

        out2 = self.upsamplex2(self.fam21_1(torch.cat([cross_result2, self.upsamplex2(cross_result1)],1)))
        out3 = self.upsamplex2(self.fam32_1(torch.cat([cross_result3, out2],1)))

        out2_2 = self.upsamplex2(self.fam21_2(torch.cat([torch.abs(cur1_2-cur2_2), self.upsamplex2(torch.abs(cur1_1-cur2_1))],1)))
        out3_2 = self.upsamplex2(self.fam32_2(torch.cat([torch.abs(cur1_3-cur2_3), out2_2],1)))

        out_1 = torch.sigmoid(self.final1(self.upsamplex2(out3)))
        out_2 = torch.sigmoid(self.final2(self.upsamplex2(out3_2)))
        out_middle_1 = torch.sigmoid(self.final_middle_1(self.upsamplex4(out2)))
        out_middle_2 = torch.sigmoid(self.final_middle_2(self.upsamplex4(out2_2)))

        return out_1, out_2, out_middle_1, out_middle_2, loss_res, loss_att

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
    
class CPAMDec_Mix(nn.Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec_Mix,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query1 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key1 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value1 = nn.Linear(in_channels, in_channels) # value2

        self.conv_query2 = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key2 = nn.Linear(in_channels, in_channels//2) # key_conv2
        self.conv_value2 = nn.Linear(in_channels, in_channels) # value2

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

class LCANet(nn.Module): 
    def __init__(self, num_classes=3, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(LCANet, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)

        self.consrative1 = ContrastiveAtt(192,64)
        self.consrative2 = ContrastiveAtt(80,32)
        self.consrative3 = ContrastiveAtt(48,16)

        # self.consrative1 = ConvBlock(192,64)
        # self.consrative2 = ConvBlock(80,32)
        # self.consrative3 = ConvBlock(48,16)

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
        
        self.final_binary = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, 1, 3, bn=False, relu=False)
            )
        self.final_binary2 = nn.Sequential(
            Conv(16, 16, 3, bn=True, relu=True),
            Conv(16, 1, 3, bn=False, relu=False)
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

        # out_1 = torch.sigmoid(self.final1(self.upsamplex2(out3)))
        # out_2 = torch.sigmoid(self.final2(self.upsamplex2(out3_2)))
        # out_middle_1 = torch.sigmoid(self.final_middle_1(self.upsamplex4(out2)))
        # out_middle_2 = torch.sigmoid(self.final_middle_2(self.upsamplex4(out2_2)))
        out_1 = self.final1(self.upsamplex2(out3))
        out_2 = self.final2(self.upsamplex2(out3_2))
        out_middle_1 = self.final_middle_1(self.upsamplex4(out2))
        out_middle_2 = self.final_middle_2(self.upsamplex4(out2_2))

        out1_binary = self.final_binary(self.upsamplex2(out3))
        out2_binary = self.final_binary2(self.upsamplex2(out3_2))

        out_binary = torch.sigmoid(out1_binary+out2_binary)

        return out_1, out_2, out_middle_1, out_middle_2, out_binary

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

        self.final_binary.apply(init_weights)
        self.final_binary2.apply(init_weights)

def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        
        inter_channels = in_channels // 2
        self.Conv1 = Conv(in_channels, inter_channels)
        self.Conv2 = Conv(in_channels, inter_channels)
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, out_channels, 3, padding=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU()) # conv_f
        
    def forward(self, x, y):
        ## Compact Spatial Attention Module(CPAM)
        # cpam_b_x = self.conv_cpam_b_x(x)
        cpam_feat1 = self.Conv1(x)
        # cpam_b_y = self.conv_cpam_b_y(y)
        cpam_feat2 = self.Conv2(y)

        feat_sum = self.conv_cat(torch.cat([cpam_feat1,cpam_feat2],1))
        return feat_sum, cpam_feat1, cpam_feat2
    
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
