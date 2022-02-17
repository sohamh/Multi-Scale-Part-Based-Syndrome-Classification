import os
import sys
os.environ["OMP_DYNAMIC"]="FALSE"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import scipy.io as sio
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset import FacialDataset
from models import Encoder, Encoder_sf, TripletNet, TripletNet_Softmax
import TripletSelector
from functions import run, eval_error_tripletloss
from functions_with_softmax import run_with_softmax
from utils import utils, dataloader, generate_spiral_seq
from arguments import parse_args, save_args

def main(fold=0,seg_num=1,seg_level=0):
    args = parse_args(fold=fold,seg_num=seg_num,seg_level=seg_level)

    args.work_dir = os.path.dirname(os.path.realpath(__file__))
    args.out_dir = os.path.join(args.work_dir, 'runs_journal/' + args.name)
    args.checkpoints_dir = os.path.join(args.out_dir, 'checkpoints')
    args.results_dir = os.path.join(args.out_dir,'results')
    args.visdom_dir = os.path.join(args.work_dir, 'visdom')
    print(args)

    utils.makedirs(args.out_dir)
    utils.makedirs(args.checkpoints_dir)
    utils.makedirs(args.results_dir)
    utils.makedirs(args.visdom_dir)
    save_args(args, args.out_dir)

    writer = utils.Writer(args)
    viz = utils.VisdomPlotter(args.out_dir, args.name, args.visdom_port, args.visdom)


    device_idx = 0
    torch.cuda.get_device_name(device_idx)
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

    # deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False # will benchmark several algorithms and pick the fastest, rule of thumb: useful if you have fixed input sizes
    cudnn.deterministic = True

    # load dataset
    print("Loading data ...")
    dataloaders = dataloader.DataLoaders(args, FacialDataset(args))
    # print("********",dataloaders.get_weights())
    # weights=torch.tensor(dataloaders.get_weights()).to(device)

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

    if (args.with_softmax == False):
        model = Encoder(args.in_channels, args.out_channels, args.latent_channels,
                   spirals, down_transform_list,
                   up_transform_list).to(device)
        tnet = TripletNet(model, TripletSelector.get_selector(args), args)
    else:
        model = Encoder_sf(args.in_channels, args.out_channels, args.latent_channels,
                        spirals, down_transform_list,
                        up_transform_list).to(device)
        tnet = TripletNet_Softmax(model, TripletSelector.get_selector(args), args)
    params = utils.count_parameters(model)
    print('Number of parameters: {}'.format(params))
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                args.decay_step,
                                                gamma=args.lr_decay)
    writer.print_model(model, params, optimizer)



    # # loss_fn = torch.nn.functional.l1_loss

    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)
    if(args.with_softmax):
        if(args.weighted):
            loss_softmax=torch.nn.CrossEntropyLoss(weight=(weights.float()))
        else:
            loss_softmax = torch.nn.CrossEntropyLoss()


    if args.resume:

        start_epoch, model, optimizer, scheduler = writer.load_checkpoint(model, device, optimizer, scheduler, True)
        print('Resuming from epoch %s' % (str(start_epoch)))
        tnet = TripletNet(model, TripletSelector.get_selector(args), args)
        if (args.with_softmax == False):
            run(tnet, dataloaders.train_loader, dataloaders.val_loader, loss_fn, args.epochs, optimizer, scheduler,
                writer, viz, device, start_epoch)
        else:

            run_with_softmax(tnet, dataloaders.train_loader, dataloaders.val_loader, loss_fn, loss_softmax, args.tl_weight, args.epochs, optimizer, scheduler,
                writer, viz, device, start_epoch)
    else:
        if (args.with_softmax == False):
            run(tnet, dataloaders.train_loader, dataloaders.val_loader, loss_fn, args.epochs, optimizer, scheduler,
                writer, viz, device)
        else:
            run_with_softmax(tnet, dataloaders.train_loader, dataloaders.val_loader, loss_fn, loss_softmax, args.tl_weight, args.epochs, optimizer, scheduler,
                writer, viz, device)

    # if args.test:
    #     eval_error(tnet, dataloaders.test_loader, loss_fn, device, args.out_dir, args.results_dir,
    #                args.epochs, viz, True)

if __name__ == '__main__':
    seg_size=[1,2,4,8]
    for i in range(1):
        for seg_level in range(1):
            for seg_num in range(1,seg_size[seg_level]+1):
                main(fold=i+1,seg_num=seg_num,seg_level=seg_level)