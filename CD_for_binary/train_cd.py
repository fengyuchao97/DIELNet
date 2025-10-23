from argparse import ArgumentParser
import torch
from models.trainer import *
import warnings
warnings.filterwarnings("ignore")

# print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='6,7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='MLLANet-mamba-251011-wo-swap', type=str) #ASCNet_chunk2
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str) # /mnt/16t/fyc

    # data
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--dataset', default='CDDataset_binary', type=str)
    parser.add_argument('--data_name', default='All', type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--split', default="train_binary", type=str) # _binary_wo_Inria
    parser.add_argument('--split_val', default="val_binary", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='MLLANet', type=str, # ASCNet_multi_granularity
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8| '
                             'cnn_trans_fuse')
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--mode', default='rsp_100', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='SGD | Adam | AdamW')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step | CosineAnnealing')
    parser.add_argument('--lr_decay_iters', default=200, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    # print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
