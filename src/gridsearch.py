import torch

from src.model import init_net
from src.architectures import Generator, Discriminator
from src.training import init_optimizers, Training
from src.visualisation import plot_loss, plot_realvsfake
from src.utils import set_seeds


def set_loss_params(loss_name):
    iter_dis, iter_gen, grad_penalty_coef = 1, 1, 0.0

    if loss_name == "wass":
        iter_dis, grad_penalty_coef = 5, 10.0

    return iter_dis, iter_gen, grad_penalty_coef


def run_experiment(ngpu, device, dataset, workers,
                   batch_size, shuffle, num_epochs, plot, lrD, lrG, beta1, nc, nz,
                   loss_name, experiment_prefix, save_stats, create_dir,
                   iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef,
                   save_epochs, save_models, momentumD, momentumG, optimizer_name,
                   PATH, img_list=None, G_losses=None, D_losses=None):

    netG = init_net(Generator(ngpu, nc, nz), device, ngpu)
    netD = init_net(Discriminator(ngpu, nc, loss_name), device, ngpu)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    fixed_noise, real_label, fake_label, optimizerD, optimizerG = init_optimizers(
        optimizer_name, netD, netG, lrD, lrG, beta1, nz, device, momentumD, momentumG)

    experiment_prefix = experiment_prefix + optimizer_name + \
        '_mG'+str(momentumD) + '_mD'+str(momentumG) + '_'

    gan_training = Training(loss_name, netD, netG, device, real_label, fake_label, dataloader, num_epochs,
                 fixed_noise, lrD, lrG, beta1, experiment_prefix, save_models,
                 PATH, save_stats, create_dir, iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef,
                 optimizerD, optimizerG, save_epochs=10)

    stats = gan_training.train()

    return stats, dataloader, netG, netD


def grid_search(ngpu, device, dataset, workers,
                experiment_prefix, batch_size_list, shuffle_list,
                num_epochs_list, loss_name_list, optimizer_name_list,
                beta1_list, lr_list, momentums_list, plot, save_stats, create_dir,
                save_epochs, save_models, manualSeed, nc, nz,
                PATH, img_list=None, G_losses=None, D_losses=None):

    for batch_size in batch_size_list:
        for shuffle in shuffle_list:
            for num_epochs in num_epochs_list:
                for loss_name in loss_name_list:

                    iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef = set_loss_params(loss_name)
                    for optimizer_name in optimizer_name_list:
                        for beta1 in beta1_list:
                            for lr in lr_list:
                                for (momentumD, momentumG) in momentums_list:
                                    lrD = lr
                                    lrG = lr
                                    # set seed before every experiment
                                    set_seeds(manualSeed=manualSeed)

                                    print('====================PARAMETERS===================')
                                    print('batch_size =', batch_size)
                                    print('shuffle =', shuffle)
                                    print('num_epoch =', num_epochs)
                                    print('loss_name =', loss_name)
                                    print('optimizer_name =', optimizer_name)
                                    print('beta1 =', beta1)
                                    print('lr =', lr)
                                    print('iter_per_epoch_dis =', iter_per_epoch_dis)
                                    print('iter_per_epoch_gen =', iter_per_epoch_gen)
                                    print('grad_penalty_coef =', grad_penalty_coef)

                                    stats, dataloader, netG, netD = run_experiment(ngpu, device, dataset, workers,
                                                                                   batch_size, shuffle, num_epochs, plot, lrD, lrG, beta1, nc, nz,
                                                                                   loss_name, experiment_prefix, save_stats, create_dir,
                                                                                   iter_per_epoch_dis, iter_per_epoch_gen, grad_penalty_coef,
                                                                                   save_epochs, save_models, momentumD, momentumG, optimizer_name,
                                                                                   PATH, img_list, G_losses, D_losses)
