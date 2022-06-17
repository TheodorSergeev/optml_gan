import torch


''' Different loss functions that can be used to train gans, finally only wassertein loss was loss'''
# stability constant
EPS = 1e-15


# KL-divergence
def loss_gen_kl(dis_output, eps=EPS):
    return - torch.log(dis_output + eps).mean()


def loss_dis_kl(dis_output_real, dis_output_fake, eps=EPS):
    return - (torch.log(dis_output_real + eps)).mean() - (torch.log(1. - dis_output_fake + eps)).mean()


# Wasserstein distance
# Requires special output of the network + weight clipping / grad penalty
def loss_gen_wasser(dis_output, eps=EPS):
    return - dis_output.mean()


def loss_dis_wasser(dis_output_real, dis_output_fake, eps=EPS):
    return - (dis_output_real.mean() - dis_output_fake.mean())


# Hinge loss
def loss_gen_hinge(dis_output, eps=EPS):
    return - dis_output.mean()


def loss_dis_hinge(dis_output_real, dis_output_fake, eps=EPS):
    return torch.nn.ReLU()(1.0 - dis_output_real).mean() + torch.nn.ReLU()(1.0 + dis_output_fake).mean()
