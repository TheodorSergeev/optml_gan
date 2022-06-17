import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import time
import os

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from .architectures import *
from .model import *

# Adapted from :
# https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid/notebook
# this didn't use batches, it was limited to one batch
# https://github.com/mseitzer/pytorch-fid
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
# this was designed to be run from the command line, and to take images in a folder as input
# we wanted to be able to run this from a samples created during runtime, rather than in
# a folder


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of the default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def get_activations(isreal, dataloader, num_samples, model, device, dims=2048):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model

    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_samples : ...

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((num_samples, dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        # if isreal:
        # batch = batch
        batch = batch[0].to(device)
        # broadcast the 1 grayscale channel to 3 channels
        batch = batch.repeat_interleave(repeats=3, dim=1)
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(isreal, dataloader, num_samples, model, device, dims=2048
                                    ):
    """Calculation of the statistics used by the FID.
    Params:

    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """

    act = get_activations(isreal, dataloader, num_samples, model, device, dims)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_frechet(device, real_dataloader, fake_dataloader, inception_model, num_samples):
    mu_1, std_1 = calculate_activation_statistics(True, real_dataloader, num_samples,
                                                  inception_model, device)
    mu_2, std_2 = calculate_activation_statistics(False, fake_dataloader, num_samples,
                                                  inception_model, device)

    """get Frechet distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value


def sample_gen_dataset(n_samples, batch_size, netG, nz, workers, device, shuffle=True):

    with torch.no_grad():
        noise = torch.randn(n_samples, nz, 1, 1, device=device)
        fake = netG(noise)
    fake = fake.to(device)
    fake_dataset = torch.utils.data.TensorDataset(fake)
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=workers)
    return fake_dataloader


def calculate_fid(num_samples, real_dataloader, batch_size_eval, device, inception_model, netG, nz, workers):
    with torch.no_grad():
        # sample the generator (and output a dataset from that)
        fake_dataloader = sample_gen_dataset(
            num_samples, batch_size_eval, netG, nz, workers, device, shuffle=True)

        t_frechet = time.time()
        frechet_dist = calculate_frechet(
            device, real_dataloader, fake_dataloader, inception_model, num_samples)
        print('frechet dist:', frechet_dist,
              '| time to calculate :', time.time()-t_frechet, 's')

    return frechet_dist

# fid scores


def get_fid_scores(ngpu, num_samples, real_dataloader, batch_size_eval, device, inception_model, nc, nz, workers,
                   list_paths,  # paths_adam paths_sgd paths_rmsprop
                   which_iterations=[50, 100],  # [0,50,100,150,200,250,290]
                   fix_extension=False,
                   calculate_frechet_bool=True,
                   which_lrs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                   ):

    all_lr_scores = {}

    for path in list_paths:
        folder = path[17:]
        print(folder)
        param_list = folder.split('_')

        optimizer_name = param_list[0]
        loss_name = param_list[3][:-4]
        lr = param_list[4][3:]
        # print()
        print(optimizer_name, lr)
        score_list = []
        print(lr)
        print(float(lr))
        if float(lr) in which_lrs:
            for file in os.listdir(path+'/models/'):
                # print(file[:7])
                if file[:7] == 'model_G':
                    # print(file)
                    # split because sometimes file has extension .zip sometimes doesn't
                    number = int(file[8:].split('.')[0])
                    # print(number)
                    if number in which_iterations:  # epochs
                        # score_list[float(lr)] = number

                        print(number)
                        # print(path+'/models/'+file)
                        # print(len(file[8:].split('.')))
                        if fix_extension:  # if files are not in .zip extension, use this
                            if len(file[8:].split('.')) == 1:
                                print(path+'/models/'+file)
                                os.rename(path+'/models/'+file,
                                          path+'/models/'+file+'.zip')

                        if calculate_frechet_bool:
                            netG = load_G(ngpu, nc, nz, Generator,
                                          path+'/models/'+file, device)
                            frechet_dist = calculate_fid(
                                num_samples, real_dataloader, batch_size_eval, device, inception_model, netG, nz, workers)
                            score_list.append(frechet_dist)
            all_lr_scores[float(lr)] = score_list

    return all_lr_scores
