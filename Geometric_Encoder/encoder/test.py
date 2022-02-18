import os
import sys
os.environ["OMP_DYNAMIC"]="FALSE"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import scipy.io as sio
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset import FacialDataset
from models import AE
from functions import run, eval_error
from utils import utils, dataloader, generate_spiral_seq
from arguments import load_args

def main(fold):
    run = 'path-to-checkpoint-name'
    checkpoint_epoch = 600

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs/', run)
    args = load_args(path+"/args.json")
    args.train = False
    args.test = True

    writer = utils.Writer(args)
    viz = utils.VisdomPlotter(args.visdom_dir, args.name, args.visdom_port, args.visdom)

    device_idx = 0
    torch.cuda.get_device_name(device_idx)
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

    # load dataset
    print("Loading data ...")
    dataloaders = dataloader.DataLoaders(args, FacialDataset(args))

    # load transform matrices
    print("Loading transformation matrices ...")
    meshdata = sio.loadmat(os.path.join(args.sample_dir + 'meshdata.mat'))
    A = np.ndarray.tolist(meshdata['A'])[0]
    D = np.ndarray.tolist(meshdata['D'])[0]
    U = np.ndarray.tolist(meshdata['U'])[0]
    F = np.ndarray.tolist(meshdata['F'])[0]

    M = sio.loadmat(os.path.join(args.sample_dir + 'M.mat'))
    M['v'] = np.ndarray.tolist(M['v'])[0]
    M['f'] = np.ndarray.tolist(M['f'])[0]

    spiral_path = spiral_path = args.sample_dir+'spirals'+''.join([str(elem) for elem in args.spiral_length])+'.mat'
    if os.path.exists(spiral_path):
        spirals = sio.loadmat(spiral_path)['spirals'][0]
    else:
        spirals = [
            np.asarray(generate_spiral_seq.extract_spirals(M['v'][idx], A[idx],
                                    args.spiral_length[idx],args.dilation[idx]))
            for idx in range(len(args.spiral_length))
        ]
        spirals = np.asarray(spirals)
        sio.savemat(spiral_path, {'spirals': spirals})
    spirals = spirals.tolist()
    spirals = [torch.tensor(elem).to(device) for elem in spirals]

    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in D
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in U
    ]

    model = AE(args.in_channels, args.out_channels, args.latent_channels,
               spirals, down_transform_list,
               up_transform_list).to(device)

    model = writer.load_checkpoint(model, None, None, device)

    # loss_fn=torch.nn.CrossEntropyLoss(weight=(weights.float()))
    loss_fn = torch.nn.CrossEntropyLoss()
    print("%%%%%",args.results_dir)
    eval_error(model, dataloaders.test_loader, loss_fn, device, args.out_dir, args.results_dir, checkpoint_epoch, viz, fold=fold, plot_emb=False)

if __name__ == '__main__':
    for i in range(10):
        main(fold=i+1)