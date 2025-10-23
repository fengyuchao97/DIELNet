import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def balanced_cross_entropy(input, target, weight=None,ignore_index=255):
    """
    类别均衡的交叉熵损失，暂时只支持2类
    TODO: 扩展到多类C>2
    """
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    # print('target.sum',target.sum())
    pos = (target==1).float()
    neg = (target==0).float()
    pos_num = torch.sum(pos) + 0.0000001
    neg_num = torch.sum(neg) + 0.0000001
    # print(pos_num)
    # print(neg_num)
    target_pos = target.float()
    target_pos[target_pos!=1] = ignore_index  # 忽略不为正样本的区域
    target_neg = target.float()
    target_neg[target_neg!=0] = ignore_index  # 忽略不为负样本的区域

    # print('target.sum',target.sum())

    loss_pos = cross_entropy(input, target_pos,weight=weight,reduction='sum',ignore_index=ignore_index)
    loss_neg = cross_entropy(input, target_neg,weight=weight,reduction='sum',ignore_index=ignore_index)
    # print(loss_neg, loss_pos)
    loss = 0.5 * loss_pos / pos_num + 0.5 * loss_neg / neg_num
    # loss = (loss_pos + loss_neg)/ (pos_num+neg_num)
    return loss

def softmax_focalloss(y_pred, y_true, ignore_index=255, gamma=2.0, normalize=False):
    """

    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar

    Returns:

    """
    y_true = y_true.long()
    if y_true.dim() == 4:
        y_true = torch.squeeze(y_true, dim=1)
    if y_pred.shape[-1] != y_true.shape[-1]:
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear',align_corners=True)

    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses


def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)


def poly_annealing(lower_bound, upper_bound, _t, _t_max):
    factor = (1 - _t / _t_max) ** 0.9
    return upper_bound + factor * (lower_bound - upper_bound)


def linear_annealing(lower_bound, upper_bound, _t, _t_max):
    factor = 1 - _t / _t_max
    return upper_bound + factor * (lower_bound - upper_bound)


def annealing_softmax_focalloss(y_pred, y_true, t=0, t_max=200, ignore_index=255, gamma=2.0,
                                annealing_function=cosine_annealing):
    y_true = y_true.long()
    if y_true.dim() == 4:
        y_true = torch.squeeze(y_true, dim=1)
    if y_pred.shape[-1] != y_true.shape[-1]:
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear',align_corners=True)
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        normalizer = losses.sum() / (losses * modulating_factor).sum()
        scales = modulating_factor * normalizer

    if t > t_max:
        scale = scales
    else:
        scale = annealing_function(1, scales, t, t_max)
    losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
    return losses
    
def cross_entropy_class(input, target, weight=None, reduction='mean', ignore_index=255, class_label=None):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def cross_entropy2(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
  
    fun_sigmoid = nn.Sigmoid()
    input_data = fun_sigmoid(torch.argmax(input, dim=1))
    dice_fun = DiceLoss()

    ce_loss = F.cross_entropy(input=input, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    
    dic_loss = dice_fun(input_data, target)
    dic_loss.requires_grad_(True)

    return  0.01 * dic_loss + ce_loss


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss
 
class MulticlassDiceLoss(nn.Module):
	"""
	requires one hot encoded target. Applies DiceLoss on each class iteratively.
	requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
	  batch size and C is number of classes
	"""
	def __init__(self):
		super(MulticlassDiceLoss, self).__init__()
 
	def forward(self, input, target, weights=None):
 
		C = target.shape[1]
 
		# if weights is None:
		# 	weights = torch.ones(C) #uniform weights for all classes
 
		dice = DiceLoss()
		totalLoss = 0
 
		for i in range(C):
			diceLoss = dice(input[:,i], target[:,i])
			if weights is not None:
				diceLoss *= weights[i]
			totalLoss += diceLoss
 
		return totalLoss


class FocalLoss2(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class=2, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        # pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
