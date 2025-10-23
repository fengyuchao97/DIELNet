import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.unet import Unet
from models.siamunet_conc import SiamUnet_conc
from models.siamunet_diff import SiamUnet_diff
from models.fresunet import FresUNet
from models.CrossNet import CrossNet 
from models.SNUNet_ECAM import SNUNet_ECAM,Siam_NestedUNet_Conc
from models.DSIFN import DSIFN
from models.DTCDSCN import CDNet34
from models.UNet_mtask import EGRCNN
from models.TransFuse_WSelf import TransFuse_WSelf
from models.CTFINet import CTFINet,DIMNet_woa,DIMNet_cross,DIMNet_self,DMINet_solo,DMINet_double,DMINet_double2,DMINet_level,DMINet_level2,DMINet_0916,DMINet_Final,DMINet50,DMINet101 #,CTFINet2,CTFINet3,CTFINet4,CTFINet5,CTFINet6
from models.DMINet import DMINet
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.MixAtt import MixAttNet34,MixAttNet5,CICNet_Self,CICNet_Cross, CICNet_VITAE2  
from models.MSPSNet import MSPSNet
from models.DARNet import DARNet
from models.FYCNet import FYCNet0222,FYCNet0223,FYCNet0224,FYCNet0227,FYCNet0226,FYCNet0228,FYCNet_best
from models.TFIGR import TFIGR
from models.TCSVT import TCSVTNet,TCSVTNet0303,TCSVTNet0306,TCSVTNet0307,TCSVTNet0308,TCSVTNet0309, TCSVTNet0310,TCSVTNet0311,TCSVTNet0312,TCSVTNet0313,TCSVTNet0314,TCSVTNet0315,TCSVTNet0316,TCSVTNet0317,TCSVTNet0318,NeurIPS0317,NeurIPS0319,NeurIPS0322,Mobile_LCA,LCANet_woDRA,TCSVTNet0316_large
from models.ReciprocalNet import ReciprocalNet,ReciprocalNet2,ReciprocalNet0212,ReciprocalNet0213,ReciprocalNet0214,ReciprocalNet0215,ReciprocalNet0216,ReciprocalNet0217,OneNet0221,OneNet0222,OneNet0223
from models.LCANet_N3C import LCANet_N3C,LCANet
from models.p2v import P2VNet
from models.P2V_FYC import ECICNet,EDMINet,P2V_FYC
from models.TDANet import TDANet_large_interact 
from models.A2Net import BaseNet
from models.MYNet3_best import MYNet3
from models.SARASNet import SARASNet
from models.ChangeFormer import ChangeFormerV6
from models.TDANet_3D_ALL_GLOBAL import TDANet_3D_ALL_GLOBAL
from models.TDANet import TDANet_3D_Mobile, TDANet_3D_ResNet, VideoNet
from models.CVPRNet import CVPRNet
from models.IJCAI import IJCAINet
from models.ASCNet import ASCNet,ASCNet_Four,ASCNet_swap,ASCNet_multi_granularity,ASCNet_full_swap
from models.GSINet import FresUNet_mixup
from models.DIELNet import DIELNet
from models.STENet import STENet

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'CosineAnnealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'BIT': #base_transformer_pos_s4_dd8
        net = BASE_Transformer(input_nc=3, output_nc=3, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    # elif args.net_G == 'cnn_trans_fuse':
    #     net = TransFuse_S(pretrained=True)
    # elif args.net_G == 'TransFuse_Trans':
    #     net = TransFuse_Trans(pretrained=True)
    # elif args.net_G == 'TransFuse_CNN':
    #     net = TransFuse_CNN(pretrained=True)
    # elif args.net_G == 'TransFuse_only_CNN':
    #     net = TransFuse_only_CNN(pretrained=True)
    # elif args.net_G == 'TransFuse_only_Trans':
    #     net = TransFuse_only_Trans(pretrained=True)
    # elif args.net_G == 'TransFuse_NCA':
    #     net = TransFuse_NCA(pretrained=True)
    # elif args.net_G == 'TransFuse_ALLC':
    #     net = TransFuse_ALLC(pretrained=True)
    # elif args.net_G == 'TransFuse_ALLD':
    #     net = TransFuse_ALLD(pretrained=True)
    # elif args.net_G == 'TransFuse_WOMF':
    #     net = TransFuse_WOMF(pretrained=True)
    elif args.net_G == 'TDANet_3D_Mobile':
        net = TDANet_3D_Mobile()
    elif args.net_G == 'TDANet_3D_ResNet':
        net = TDANet_3D_ResNet()
    elif args.net_G == 'TDANet_3D_ALL_GLOBAL':
        net = TDANet_3D_ALL_GLOBAL()
    elif args.net_G == 'FC-EF':
        net = Unet()
    elif args.net_G == 'ASCNet':
        net = ASCNet()
    elif args.net_G == 'ASCNet_multi_granularity':
        net = ASCNet_multi_granularity()
    elif args.net_G == 'MLLANet':
        net = MLLANet()
    elif args.net_G == 'MLLANet_SAM':
        net = MLLANet_SAM()
    elif args.net_G == 'ASCNet_Four':
        net = ASCNet_Four()
    elif args.net_G == 'FC-Siam-Diff':
        net = SiamUnet_diff()
    elif args.net_G == 'FC-Siam-Conc':
        net = SiamUnet_conc()  
    elif args.net_G == 'FresUNet':
        net = FresUNet()
    elif args.net_G == 'FresUNet_mixup':
        net = FresUNet_mixup()

    elif args.net_G == 'STENet':
        net = STENet()
    elif args.net_G == 'ICIFNet':
        net = TransFuse_WSelf(pretrained=True)
    elif args.net_G == 'CTFINet':
        net = CTFINet()
    elif args.net_G == 'DMINet':
        net = DMINet()
    elif args.net_G == 'VideoNet':
        net = VideoNet()
    elif args.net_G == 'MYNet3':
        net = MYNet3()
    elif args.net_G == 'DIMNet_woa':
        net = DIMNet_woa()
    elif args.net_G == 'DIMNet_cross':
        net = DIMNet_cross()
    elif args.net_G == 'DIMNet_self':
        net = DIMNet_self()
    elif args.net_G == 'DMINet_solo':
        net = DMINet_solo()
    elif args.net_G == 'DMINet_double':
        net = DMINet_double()
    elif args.net_G == 'DMINet_double2':
        net = DMINet_double2()
    elif args.net_G == 'DMINet_level':
        net = DMINet_level()
    elif args.net_G == 'DMINet_level2':
        net = DMINet_level2()
    elif args.net_G == 'DMINet_0916':
        net = DMINet_0916()
    elif args.net_G == 'DMINet_Final':
        net = DMINet_Final()
    elif args.net_G == 'ReciprocalNet':
        net = ReciprocalNet()
    elif args.net_G == 'ReciprocalNet2':
        net = ReciprocalNet2()
    elif args.net_G == 'ReciprocalNet0212':
        net = ReciprocalNet0212()
    elif args.net_G == 'ReciprocalNet0213':
        net = ReciprocalNet0213()
    elif args.net_G == 'ReciprocalNet0214':
        net = ReciprocalNet0214()
    elif args.net_G == 'ReciprocalNet0215':
        net = ReciprocalNet0215()
    elif args.net_G == 'ReciprocalNet0216':
        net = ReciprocalNet0216()
    elif args.net_G == 'ReciprocalNet0217':
        net = ReciprocalNet0217()
    elif args.net_G == 'OneNet0221':
        net = OneNet0221()
    elif args.net_G == 'OneNet0222':
        net = OneNet0222()
    elif args.net_G == 'OneNet0223':
        net = OneNet0223()
    elif args.net_G == 'FYCNet0222':
        net = FYCNet0222()    
    elif args.net_G == 'FYCNet0223':
        net = FYCNet0223()   
    elif args.net_G == 'FYCNet0224':
        net = FYCNet0224()   
    elif args.net_G == 'FYCNet0227':
        net = FYCNet0227()
    elif args.net_G == 'FYCNet0228':
        net = FYCNet0228()
    elif args.net_G == 'FYCNet0226':
        net = FYCNet0226()
    elif args.net_G == 'FYCNet_best':
        net = FYCNet_best()
    elif args.net_G == 'ECICNet':
        net = ECICNet()
    elif args.net_G == 'EDMINet':
        net = EDMINet() 
    elif args.net_G == 'TDANet_large_interact':
        net = TDANet_large_interact()    
    # elif args.net_G == 'TDANet_large':
    #     net = TDANet_large()    
    # elif args.net_G == 'TDANet_large_mul':
    #     net = TDANet_large_mul()  
    # elif args.net_G == 'TDANet_large2D':
    #     net = TDANet_large2D()    
    # elif args.net_G == 'TDANet_large_0607':
    #     net = TDANet_large_0607()    
    # elif args.net_G == 'TDANet_Resnet':
    #     net = TDANet_Resnet() 
    elif args.net_G == 'TFIGR':
        net = TFIGR()
    elif args.net_G == 'DMINet50':
        net = DMINet50()
    elif args.net_G == 'DMINet101':
        net = DMINet101()
    elif args.net_G == 'CICNet_CVPR':
        net = CICNet_CVPR()        
    elif args.net_G == 'CICNet_CVPR0228':
        net = CICNet_CVPR0228()  
    elif args.net_G == 'CICNet_CVPR0229':
        net = CICNet_CVPR0229()  
    elif args.net_G == 'CICNet_CVPR0230':
        net = CICNet_CVPR0230() 

    elif args.net_G == 'P2VNet':
        net = P2VNet()
    elif args.net_G == 'EP2V':
        net = P2V_FYC()
    elif args.net_G == 'A2Net':
        net = BaseNet()
    elif args.net_G == 'ASCNet_swap':
        net = ASCNet_swap()
    elif args.net_G == 'ASCNet_full_swap':
        net = ASCNet_full_swap()        
    elif args.net_G == 'CVPRNet':
        net = CVPRNet()
    elif args.net_G == 'IJCAINet':
        net = IJCAINet()
    elif args.net_G == 'Mobile_LCA':
        net = Mobile_LCA() 
    elif args.net_G == 'LCANet_woDRA':
        net = LCANet_woDRA() 
    elif args.net_G == 'TCSVTNet':
        net = TCSVTNet() 
    elif args.net_G == 'TCSVTNet0303':
        net = TCSVTNet0303() 
    elif args.net_G == 'TCSVTNet0306':
        net = TCSVTNet0306()
    elif args.net_G == 'TCSVTNet0307':
        net = TCSVTNet0307() 
    elif args.net_G == 'TCSVTNet0308':
        net = TCSVTNet0308() 
    elif args.net_G == 'TCSVTNet0309':
        net = TCSVTNet0309() 
    elif args.net_G == 'TCSVTNet0310':
        net = TCSVTNet0310() 
    elif args.net_G == 'TCSVTNet0311':
        net = TCSVTNet0311() 
    elif args.net_G == 'TCSVTNet0312':
        net = TCSVTNet0312() 
    elif args.net_G == 'TCSVTNet0313':
        net = TCSVTNet0313()
    elif args.net_G == 'TCSVTNet0314':
        net = TCSVTNet0314() 
    elif args.net_G == 'TCSVTNet0315':
        net = TCSVTNet0315()  
    elif args.net_G == 'TCSVTNet0316':
        net = TCSVTNet0316() 
    elif args.net_G == 'LCANet_large':
        net = TCSVTNet0316_large() 
    elif args.net_G == 'TCSVTNet0317':
        net = TCSVTNet0317() 
    elif args.net_G == 'TCSVTNet0318':
        net = TCSVTNet0318() 
    elif args.net_G == 'NeurIPS0317':
        net = NeurIPS0317() 
    elif args.net_G == 'NeurIPS0319':
        net = NeurIPS0319() 
    elif args.net_G == 'NeurIPS0322':
        net = NeurIPS0322() 
    
    elif args.net_G == 'LCANet_N3C':
        net = LCANet_N3C() 
    elif args.net_G == 'LCANet':
        net = LCANet() 
    # elif args.net_G == 'CTFINet2':
    #     net = CTFINet2()
    # elif args.net_G == 'CTFINet3':
    #     net = CTFINet3()
    # elif args.net_G == 'CTFINet4':
    #     net = CTFINet4()
    # elif args.net_G == 'CTFINet5':
    #     net = CTFINet5()
    # elif args.net_G == 'CTFINet6':
    #     net = CTFINet6()
    # elif args.net_G == 'TransFuse_TCA':
    #     net = TransFuse_TCA()
    elif args.net_G == 'CICNet':
        net = MixAttNet5()
    elif args.net_G == 'CICNet34':
        net = MixAttNet34()
    elif args.net_G == 'CICNet_Self':
        net = CICNet_Self()
    elif args.net_G == 'CICNet_Cross':
        net = CICNet_Cross()
    elif args.net_G == 'CICNet_VITAE2':
        net = CICNet_VITAE2(args)
    # elif args.net_G == 'CrossNet2':
    #     net = CrossNet2()
    # elif args.net_G == 'CrossNet3':
    #     net = CrossNet3()
    # elif args.net_G == 'CrossNet4':
    #     net = CrossNet4()
    # elif args.net_G == 'CrossNet5':
    #     net = CrossNet5()
    # elif args.net_G == 'CrossNet6':
    #     net = CrossNet6()
    # elif args.net_G == 'CrossNet7':
    #     net = CrossNet7()
    # elif args.net_G == 'CrossNet_Final':
    #     net = CrossNet_Final()
    elif args.net_G == 'SNUNet': # SNUNet_ECAM
        net = SNUNet_ECAM()
    elif args.net_G == 'Siam_NestedUNet_Conc':
        net = Siam_NestedUNet_Conc()
    elif args.net_G == 'IFNet': # IFNet
        net = DSIFN()
    elif args.net_G == 'DTCDSCN':
        net = CDNet34(3)
    elif args.net_G == 'EGRCNN':
        net = EGRCNN()
    # elif args.net_G == 'ClassSegNet':
    #     net = ClassSegNet()
    # elif args.net_G == 'MixAttNet':
    #     net = MixAttNet()
    # elif args.net_G == 'MixAttNet2':
    #     net = MixAttNet2()
    # elif args.net_G == 'MixAttNet3':
    #     net = MixAttNet3()
    # elif args.net_G == 'MixAttNet4':
    #     net = MixAttNet4()
    # elif args.net_G == 'MixAttNet5':
    #     net = MixAttNet5()   
    elif args.net_G == 'MSPSNet':
        net = MSPSNet()
    elif args.net_G == 'DARNet':
        net = DARNet()
    # elif args.net_G == 'MixAttNet6':
    #     net = MixAttNet6()   
    # elif args.net_G == 'MixAttNet7':
    #     net = MixAttNet7()  
    # elif args.net_G == 'MixAttNet8':
    #     net = MixAttNet8()  
    # elif args.net_G == 'MixAttNet9':
    #     net = MixAttNet9()  
    # elif args.net_G == 'MixAttNet10':
    #     net = MixAttNet10()  
    elif args.net_G == 'SARASNet':
        net = SARASNet()
    elif args.net_G == 'ChangeFormerV6':
        net = ChangeFormerV6()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=False,
                                          replace_stride_with_dilation=[False,True,True])
            self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.classifier2 = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x_binary = self.classifier2(x)
        x = self.classifier(x)

        x = self.sigmoid(x)
        x_binary = self.sigmoid(x_binary)
        return x, x_binary