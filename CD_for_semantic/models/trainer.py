import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy,FocalLoss,softmax_focalloss,annealing_softmax_focalloss
import models.losses as losses
from models.losses2 import Hybrid,Dice
import torch.nn.functional as F

from misc.logger_tool import Logger, Timer
# from spikingjelly.activation_based import functional
from utils import de_norm
import time
import math
# import wandb
import os 
class_to_color = {
    0: (0, 0, 0),      # 类别 1 对应黑色
    1: (255, 255, 0),  # 类别 2 对应黄色
    2: (255, 0, 0)    # 类别 3 对应红色
}

color_to_class = {
    (0, 0, 0) : 0,    # 类别 1 对应黑色 -> 非变化
    (1, 1, 0) : 1,    # 类别 2 对应黄色 -> 拆除
    (1, 0, 0) : 2    # 类别 3 对应红色 -> 新建
}


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

def BCELoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    return bce

def CELoss(input, target, weight=None, reduction='mean',ignore_index=255):
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

def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)

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

class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.n_class = args.n_class
        self.n_class2 = args.n_class2
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        # print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "Adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
        elif args.optimizer == "SGD":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4) # default: 5e-4
        elif args.optimizer == "AdamW":
            self.optimizer_G = torch.optim.AdamW(self.net_G.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
        else:
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4) # default: 5e-4

        # self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=3)

        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = 0
        for single_dataloaders in dataloaders:
            self.steps_per_epoch += len(single_dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.class_pre = None
        self.G_pred = None
        self.G_pred_binary = None
        self.G_pred_v = None
        self.G_pred_v1 = None
        self.G_pred_v2 = None
        self.output_binary = None
        self.G_pred1 = None
        self.G_pred2 = None
        self.G_pred3 = None
        self.latent_loss = None
        self.attention_loss = None
        self.loss_res = None 
        self.loss_att = None
        self.loss_lat = None

        self.Edge1 = None
        self.Edge2 = None

        self.d6_out = None
        self.d5_out = None
        self.d4_out = None
        self.d3_out = None
        self.d2_out = None

        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        self.loss = args.loss
        if args.loss == 'ce':
            hybrid_loss = Hybrid()
            dice = Dice()
            self._pxl_loss = cross_entropy
        elif args.loss == 'focal':
            self._pxl_loss = annealing_softmax_focalloss
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'cd_loss':
            self._pxl_loss = losses.cd_loss
        elif args.loss == 'hybrid':
            hybrid_loss = Hybrid()
            self._pxl_loss = cross_entropy
            self._pxl_loss2 = hybrid_loss
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est
    
    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1)
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()
        # pred_vis = pred * 255

        B, H, W = pred.shape
        pred_vis = torch.zeros((B, 3, H, W), dtype=torch.uint8)

        # 遍历输入张量，根据像素值设置对应的 RGB 值
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    pixel_value = pred[b, i, j].item()
                    pred_vis[b, :, i, j] = torch.tensor(class_to_color[pixel_value])
        return pred_vis
    
    def _visualize_gt(self):
        target = self.batch['L'].long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        B, H, W = target.shape
        pred_vis = torch.zeros((B, 3, H, W), dtype=torch.uint8)

        # 遍历输入张量，根据像素值设置对应的 RGB 值
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    pixel_value = target[b, i, j].item()
                    pred_vis[b, :, i, j] = torch.tensor(class_to_color[pixel_value])
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """

        compute_semantic = self.batch['compute_semantic'].to(self.device)

        compute_semantic = compute_semantic.to(self.batch['L'].device)

        if compute_semantic.any():
            target_true = self.batch['L'][compute_semantic].to(self.device).detach()
            G_pred_true = self.G_pred[compute_semantic].detach()
            G_pred_true = torch.argmax(G_pred_true, dim=1)
            
            current_score = self.running_metric.update_cm(pr=G_pred_true.cpu().numpy(), gt=target_true.cpu().numpy())
        else:
            # current_score_true = 0 
            target_false = self.batch['L_binary'][~compute_semantic].to(self.device).detach()
            
            G_pred_binary_false = torch.where(self.G_pred_binary[~compute_semantic] > 0.5, 
                                            torch.ones_like(self.G_pred_binary[~compute_semantic]), 
                                            torch.zeros_like(self.G_pred_binary[~compute_semantic])).long()
            
            current_score = self.running_metric.update_cm(pr=G_pred_binary_false.cpu().numpy(), gt=target_false.cpu().numpy())

        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()
        m=0
        for single_dataloaders in self.dataloaders:
            m = m + len(single_dataloaders['train'])
        if self.is_training is False:
            for single_dataloaders in self.dataloaders:
                m = m+len(single_dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)


    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        
        # img_in1 = batch['A_aug'].to(self.device)
        # img_in2 = batch['B_aug'].to(self.device)
        
        imgs = batch['aug'].to(self.device)
        self.G_pred, self.G_pred_middle1, self.G_pred_middle2, self.G_pred_v, self.G_pred_binary, self.G_pred_v_binary = self.net_G(imgs) 

    def _backward_G(self):
        compute_semantic = self.batch['compute_semantic'].to(self.device)
        gt = self.batch['L'].to(self.device).float()
        gt_binary = self.batch['L_binary'].to(self.device).float()


        # 分别计算 True 和 False 样本的损失
        if compute_semantic.any():
            true_loss = true_loss = BCEDiceLoss(self.G_pred_binary[compute_semantic], gt_binary[compute_semantic]) +\
                0.5*BCEDiceLoss(self.G_pred_v_binary[compute_semantic], gt_binary[compute_semantic]) + CELoss(self.G_pred[compute_semantic], gt[compute_semantic]) +\
                    0.5*(CELoss(self.G_pred_middle1[compute_semantic], F.interpolate(gt, size=self.G_pred_middle1.shape[2:])[compute_semantic]) +\
                            CELoss(self.G_pred_middle2[compute_semantic], F.interpolate(gt, size=self.G_pred_middle2.shape[2:])[compute_semantic]) +\
                                0.5*CELoss(self.G_pred_v[compute_semantic], F.interpolate(gt, size=self.G_pred_v.shape[2:])[compute_semantic]))  #+0.01*self.loss_res.mean() # (self.loss_att.mean() + self.loss_lat.mean()) #+0.5*self.loss_res.mean()
        else:
            true_loss = torch.tensor(0.0, device=self.device)

        if (~compute_semantic).any():    
            false_loss = BCEDiceLoss(self.G_pred_binary[~compute_semantic], gt_binary[~compute_semantic]) + 0.5*BCEDiceLoss(self.G_pred_v_binary[~compute_semantic], gt_binary[~compute_semantic])
        else:
            false_loss = torch.tensor(0.0, device=self.device)
        # 计算最终损失并求平均
        self.G_loss = (true_loss.sum() + false_loss.sum()) 

        self.G_loss.backward()

    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True

            starttime = time.time()
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            
            start = 0
            for single_dataloaders in self.dataloaders:
                print('single_dataloaders:',len(single_dataloaders['train']))
                    
                for self.batch_id, batch in enumerate(single_dataloaders['train'], 0):
                # for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                    self._forward_pass(batch)
                    # update G
                    self.optimizer_G.zero_grad()
                    self._backward_G()
                    self.optimizer_G.step()
                    self._collect_running_batch_states()
                    self._timer_update()
                    # functional.reset_net(self.net_G) 
                start += self.batch_id

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            endtime = time.time()-starttime
            self.logger.write('epoch time: %0.3f \n' % endtime)

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            start = 0
            for single_dataloaders in self.dataloaders:
                for self.batch_id, batch in enumerate(single_dataloaders['val'], 0):
            # for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                    with torch.no_grad():
                        self._forward_pass(batch)
                    self._collect_running_batch_states()
                start += self.batch_id
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()
        
        # wandb.finish()

