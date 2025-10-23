import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from numpy.core.fromnumeric import choose
from utils import de_norm
import utils


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    mean_recall = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    mean_precision = np.nanmean(precision)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1, mean_precision, mean_recall

class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        # print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        logger_path2 = os.path.join(args.checkpoint_dir, 'log_F1_test.txt')
        self.logger2 = Logger(logger_path2)
        self.logger2.write_dict_str(args.__dict__)

        logger_path3 = os.path.join(args.checkpoint_dir, 'log_PRE_test.txt')
        self.logger3 = Logger(logger_path3)
        self.logger3.write_dict_str(args.__dict__)

        logger_path4 = os.path.join(args.checkpoint_dir, 'log_RECALL_test.txt')
        self.logger4 = Logger(logger_path4)
        self.logger4.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.batch_size = args.batch_size
        self.steps_per_epoch = len(dataloader)
        self.attention_loss = None

        self.G_pred5 = None
        self.G_pred4 = None
        self.G_pred3 = None
        self.G_pred2 = None
        self.G_pred1 = None
        self.G_pred = None

        self.d6_out = None
        self.d5_out = None
        self.d4_out = None
        self.d3_out = None
        self.d2_out = None

        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)
        if os.path.exists(self.vis_dir+"/predict") is False:
            os.mkdir(self.vis_dir+"/predict")
        if os.path.exists(self.vis_dir+"/label") is False:
            os.mkdir(self.vis_dir+"/label")


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            # self.net_G = torch.nn.DataParallel(self.net_G)
            # {k.replace('module.',''):v for k,v in checkpoint['model_G_state_dict'].items()}
            # checkpoint['model_G_state_dict']
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        
        # G_pred = self.G_pred.detach()
        # G_pred = torch.argmax(G_pred, dim=1)
        
        G_pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        F1, Pre, Rec = cm2F1(self.running_metric.value())
        self.logger2.write('%s\n' % F1)
        self.logger3.write('%s\n' % Pre)
        self.logger4.write('%s\n' % Rec)

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)


        # if np.mod(self.batch_id, 2) == 0:#LEVIR
        # if np.mod(self.batch_id, 2) == 1:#WHU
        if np.mod(self.batch_id, 1) == 0:
            

            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        # img_in1 = batch['A'].to(self.device)
        # img_in2 = batch['B'].to(self.device)
        imgs = batch['aug'].to(self.device)
        self.G_pred, self.G_pred_middle1, self.G_pred_middle2, self.G_pred_v = self.net_G(imgs)
       
    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
