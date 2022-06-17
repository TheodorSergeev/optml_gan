from __future__ import print_function
# from src.file import *
PATH = './'


import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from src.data_handling import *
from src.utils import *
from src.model import *
from src.losses import *
from src.fid import *

loss_dict = {
    "kl": (loss_dis_kl, loss_gen_kl),
    "wass": (loss_dis_wasser, loss_gen_wasser),
    "hinge": (loss_dis_hinge, loss_gen_hinge)
}

# FID

from src.training import *
from src.visualisation import *
from src.serialisation import *

# https://keras.io/examples/generative/conditional_gan/
from src.architectures import *

from src.gridsearch import *




def do_something():
    pass


if __name__ == '__main__':

    # Root directory for dataset
    dataroot = PATH + "data/"

    # Dataset name
    dataset_name = 'mnist'  # 'cifar10' or 'mnist'

    # Number of workers for dataloader
    workers = 2

    # Spatial size of training images. All images will be resized to this size using a transformer
    image_size = 28  # 28 for mnist, 64 for others

    # Size of z latent vector (i.e. size of generator input)
    nz = 128

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # set batch size
    batch_size = 128

    dataset, nc = get_dataset(dataset_name, image_size, dataroot)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    loss_name='wass'
    lrD = 1e-4
    lrG = 1e-4
    beta1= 0.9
    shuffle = True
    num_epochs = 150
    plot = True
    save_stats = True
    create_dir = True
    save_epochs = True
    momentumD, momentumG = 0.0,0.0
    optimizer_name = 'adam'

    iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef = set_loss_params(loss_name)

    stats, dataloader, netG, netD = run_experiment(ngpu, device, dataset, workers, batch_size, 
                    shuffle, num_epochs, plot, lrD, lrG, beta1, nc, nz, loss_name, '', save_stats, create_dir,
                    iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef,
                    save_epochs, save_models, momentumD, momentumG, optimizer_name, PATH)

    img_list = stats['img_list']
    G_losses = stats['G_losses']
    D_losses = stats['D_losses']

    save_path = PATH + 'img/real_vs_fake'
    plot_loss(G_losses, D_losses, save_path, save=True)
    plot_realvsfake(dataloader, device, img_list, PATH + 'img/loss', save=True)

    pass