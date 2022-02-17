# Utility functions & configuration constants
import os
import argparse
import json
import torch.cuda


def parse_args(fold=0,seg_num=1,seg_level=0):
    base = '/usr/local/micapollo01/MIC/DATA/STAFF/smahdi0/tmp/'

    parser = argparse.ArgumentParser(description='PyTorch Autoencoder')

    # general
    parser.add_argument('--name', default='8k_journal_mnorm_24dim_dv2_uncorrected', type=str, help='name of experiment')
    parser.add_argument('--comment', default='first run of triplet loss with the updated and corrected data',
                        type=str)
    parser.add_argument('--seed', default=7, type=int, help='set random seed')
    parser.add_argument('--train', default=True, type=bool, help='training mode')
    parser.add_argument('--test', default=False, type=bool, help='test mode')
    parser.add_argument('--test_val', default=False, type=bool, help='find all trainset projections')
    parser.add_argument('--test_controls', default=False, type=bool, help='test controls mode')
    parser.add_argument('--resume', default=False, type=bool, help='resuming from checkpoint')
    parser.add_argument('--checkpoint_epoch', default=600, type=int)
    parser.add_argument('--cuda', action='store_true', default=True and torch.cuda.is_available(),
                        help='enables CUDA training')

    # data
    # parser.add_argument('--total_train_size', default=2277, help='# trainset ')
    parser.add_argument('--data_dir',
                        default='path-to-folder/Batch' + str(
                            fold) + '_trainset.mat',
                        type=str, metavar='PATH', help='data path')
    parser.add_argument('--test_data_dir',
                        default='path-to-folder/Batch' + str(
                            fold) + '_test.mat',
                        type=str, metavar='PATH', help='data path')
    parser.add_argument('--avg_data_dir',
                        default='path-to-folder/avg_all.mat',
                        type=str, metavar='PATH', help='average of all meshes')
    parser.add_argument('--seg_label',
                        default='path-to-folder/Segmentation_eqMap.mat',
                        type=str, metavar='PATH', help='average of all meshes')

    parser.add_argument('--nVal', default=200, help='# validation samples')
    parser.add_argument('--sample_dir',
                        default='data/eq_sample/',
                        type=str, metavar='PATH', help='directory containing downsample matrices')
    parser.add_argument('--mean_normalize', default=True, type=bool, help='subtract avg from faces')
    parser.add_argument('--ismodular', default=True, type=bool, help='modular implementation (meannormalized to be considered)')

    # Visdom
    parser.add_argument('--visdom', default=True, type=bool, help='plot training curves in visdom')
    parser.add_argument('--visdom_port', default=8098, type=int)

    # training hyperparameters
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--weighted', default=True, type=bool)

    # network hyperparameters
    parser.add_argument('--out_channels', default=[32, 32, 32, 64], nargs='+', type=int)
    parser.add_argument('--latent_channels', default=24, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--spiral_length', default=[9, 9, 9, 9], type=int, nargs='+')
    parser.add_argument('--dilation', default=[1, 1, 1, 1], type=int, nargs='+')
    parser.add_argument('--pooling', default='None', type=str)

    #triplet loss hyperparams
    parser.add_argument('--ts', type=str, default='batch_random', metavar='S', help='TripletSelector to use')
    parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--with_softmax',default=False, type=bool,help='add softmax to loss?')
    parser.add_argument('--texture_only', default=False, type=bool, help='network for texture data only?')
    parser.add_argument('--tl_weight', type=float, default=8, help='weight for tripletloss factor in loss')
    parser.add_argument('--grayscale', default=False, type=bool, help='grayscale')

    # optimizer hyperparmeters
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay', default=0.99, type=float)
    parser.add_argument('--decay_step', default=1, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)

    args = parser.parse_args()
    args.seg_num = seg_num
    args.seg_level = seg_level
    if (args.texture_only == True):
        args.name = "texture_" + args.name + "_level" + str(seg_level) + "_segment" + str(seg_num) + "_batch" + str(
            fold)
    else:
        args.name = args.name + "_level" + str(seg_level) + "_segment" + str(seg_num) + "_batch" + str(fold)

    return args


def save_args(args, path):
    """Saves parameters to json file"""
    json_path = "{}/args.json".format(path)
    with open(json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def load_args(json_path):
    """Loads parameters from json file"""
    with open(json_path) as f:
        params = json.load(f)
    return argparse.Namespace(**params)
