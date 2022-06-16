import time

import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils

from src.utils import generate_paths, model_paths
from src.serialisation import save_models, pickle_save
from src.losses import *


loss_dict = {
    "kl": (loss_dis_kl, loss_gen_kl),
    "wass": (loss_dis_wasser, loss_gen_wasser),
    "hinge": (loss_dis_hinge, loss_gen_hinge)
}


def init_optimizers(optimizer_name, netD, netG, lrD, lrG, beta1, nz, device, momentumD, momentumG):
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    if optimizer_name == 'adam':
        # Setup Adam optimisers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    elif optimizer_name == 'sgd':
        optimizerD = optim.SGD(netD.parameters(), lr=lrD, momentum=momentumD, dampening=0, weight_decay=0)
        optimizerG = optim.SGD(netG.parameters(), lr=lrG, momentum=momentumG, dampening=0, weight_decay=0)

    elif optimizer_name == 'rmsprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr=lrD, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        optimizerG = optim.RMSprop(netG.parameters(), lr=lrG, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    return fixed_noise, real_label, fake_label, optimizerD, optimizerG


def init_losses(loss_type):
    if loss_type not in loss_dict.keys():
        raise Exception("Unknown loss type")

    return loss_dict[loss_type]


def gradient_penalty(device, discriminator, data_gen, data_real, lambda_reg=0.1):
    alpha = torch.rand(data_real.shape[0], 1).to(device)
    dims_to_add = len(data_real.size()) - 2
    for i in range(dims_to_add):
        alpha = alpha.unsqueeze(-1)

    interpolates = (alpha * data_real + ((1. - alpha) * data_gen)).to(device)

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size()).to(device)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    grad_penalty_coef = ((gradients.norm(2, dim=1) - 1)
                         ** 2).mean() * lambda_reg

    return grad_penalty_coef


def discriminator_step(optimizerD, f_loss_dis, netD, netG, data, device, real_label, fake_label, gp_coef, nz=128):
    netD.zero_grad()

    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    output_real = netD(real_cpu).view(-1)

    noise = torch.randn(b_size, nz, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    output_fake = netD(fake.detach()).view(-1)

    errD = f_loss_dis(output_real, output_fake)

    if gp_coef != 0.0:
        errD += gp_coef * gradient_penalty(device, netD, fake, real_cpu)

    errD.backward()
    optimizerD.step()

    D_x = output_real.mean().item()
    D_G_z1 = output_fake.mean().item()

    return D_x, D_G_z1, errD, label, fake, real_cpu


def generator_step(optimizerG, f_loss_gen, netD, netG, label, fake, real_label):
    netG.zero_grad()
    output = netD(fake).view(-1)

    errG = f_loss_gen(output)
    errG.backward()

    D_G_z2 = output.mean().item()

    optimizerG.step()
    return D_G_z2, errG


class Training:
    def __init__(self, loss_name, netD, netG, device, real_label, fake_label, dataloader, num_epochs,
                 fixed_noise, lrD, lrG, beta1, experiment_prefix, save_models,
                 PATH, save_stats, create_dir, iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef,
                 optimizerD, optimizerG, save_epochs=10):
        self.optimizerD, self.optimizerG = optimizerD, optimizerG

        self.loss_name = loss_name
        self.netD, self.netG = netD, netG
        self.device = device
        self.real_label, self.fake_label = real_label, fake_label
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.fixed_noise = fixed_noise
        self.iter_per_epoch_dis, self.iter_per_epoch_gen = iter_per_epoch_dis, iter_per_epoch_gen
        self.grad_penalty_coef = grad_penalty_coef

        self.save_models = save_models
        self.PATH = PATH
        self.experiment_prefix = experiment_prefix
        self.loss_name = loss_name
        self.lrD = lrD
        self.lrG = lrG
        self.beta1 = beta1
        self.create_dir = create_dir
        self.save_stats = save_stats
        self.save_epochs = save_epochs
        self.experiment_path, self.stats_path, self.models_path = generate_paths(self.PATH, self.experiment_prefix,
                                                                                 self.loss_name, self.lrD, self.lrG,
                                                                                 self.beta1, self.iter_per_epoch_dis, self.iter_per_epoch_gen,
                                                                                 self.grad_penalty_coef, self.create_dir)

    def _output_training_stats(self, epoch, i, size, errD, errG, D_x, D_G_z1, D_G_z2, t0):
        if i == size:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f t: %2.3f'
                  % (epoch, self.num_epochs, i, len(self.dataloader),
                     errD, errG, D_x, D_G_z1, D_G_z2, time.time()-t0))
        elif i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, self.num_epochs, i, len(self.dataloader),
                     errD, errG, D_x, D_G_z1, D_G_z2))

    def _save_gen_output(self, iters, epoch, i):
        if iters % 500 == 0 or epoch == self.num_epochs-1 and i == len(self.dataloader)-1:
            with torch.no_grad():
                fake = self.netG(self.fixed_noise).detach().cpu()
            self.img_list.append(vutils.make_grid(
                fake, padding=2, normalize=True))
            self.img_list_nogrid.append(fake)

    def train(self):
        f_loss_dis, f_loss_gen = init_losses(self.loss_name)

        G_losses, D_losses = [], []
        self.img_list = []
        self.img_list_nogrid = []

        iters = 0

        print("Starting Training Loop...")

        D_x, D_G_z1, errD, label, fake = None, None, None, None, None
        D_G_z2, errG = None, None

        for epoch in range(self.num_epochs):
            t0 = time.time()

            for i, data in enumerate(self.dataloader, 0):
                for _ in range(self.iter_per_epoch_dis):
                    D_x, D_G_z1, errD, label, fake, real_cpu = discriminator_step(self.optimizerD,
                                                                                  f_loss_dis, self.netD, self.netG, data, self.device,
                                                                                  self.real_label, self.fake_label, self.grad_penalty_coef
                                                                                  )

                for _ in range(self.iter_per_epoch_gen):
                    D_G_z2, errG = generator_step(self.optimizerG,
                                                  f_loss_gen, self.netD, self.netG, label, fake, self.real_label
                                                  )

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                size = len(self.dataloader) - 1
                self._output_training_stats(
                    epoch, i, size, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, t0)

                # Check how the generator is doing by saving G's output on fixed_noise
                self._save_gen_output(iters, epoch, i)

                iters += 1

            # save the model every self.save_epochs epochs
            if self.save_models and (epoch % self.save_epochs == self.save_epochs - 1):
                self.save_path_G, self.save_path_D = model_paths(self.experiment_path, epoch, self.models_path)
                save_models(self.netG, self.netD, self.save_path_G, self.save_path_D)

        stats = {
            'img_list': self.img_list,
            'img_list_nogrid': self.img_list_nogrid,
            'G_losses': G_losses,
            'D_losses': D_losses
        }
        # Save stats at the end of training
        if self.save_stats:
            pickle_save(stats, self.stats_path)

        return stats
