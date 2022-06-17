import torchvision.transforms.functional as tff

import numpy as np
import matplotlib.pyplot as plt

from .visualisation import *
from .fid import *
from .serialisation import *
from matplotlib import gridspec


def show(imgs, save_path, save_fig, show_plot):
    '''
    Function used by get_loss_plots to genrate the 1x5 plots 
    of fake samples
    '''
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig = plt.figure(figsize=(8, 4))

    ax = [fig.add_subplot(1, len(imgs), i+1) for i in range(len(imgs))]

    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

    fig.subplots_adjust(wspace=0, hspace=0)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = tff.to_pil_image(img)
        ax[i].imshow(np.asarray(img), cmap='gray')
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_fig:
        plt.savefig(save_path, format="png", dpi=400)
    if show_plot:
        plt.show()


def get_loss_plots(generated_data_path, list_paths, PATH,
                   plot_losses=True,
                   save_gen_samples=True, show_plot=True, save_fig=False):
    '''
    This function loads the saved stat files in generated_data and plots the
    generator and discriminator loss, and also plots fake samples from the generator
    '''

    for path in list_paths:
        folder = path[17:]

        param_list = folder.split('_')

        optimizer_name = param_list[0]
        loss_name = param_list[3][:-4]
        lr = param_list[4][3:]

        print(optimizer_name, lr)
        score_list = []

        stats_path = path + '/stat.pickle'
        stats = pickle_load(stats_path)
        # 8x8 images fake generatred images in one picture
        img_list = stats['img_list']
        G_losses = stats['G_losses']
        D_losses = stats['D_losses']
        save_path = PATH + 'img/real_vs_fake_'+folder+'.png'

        if plot_losses:
            plot_loss(G_losses, D_losses, save_path, title_fontsize=18, x_fontsize=15, y_fontsize=15, yticks_size=12,
                      xticks_size=12, save=True, show_plot=show_plot)

        if save_gen_samples:
            save_path = PATH + 'img/'+optimizer_name+lr+'.png'
            # 64 fake generatred images in a list
            img_list_nogrid = stats['img_list_nogrid']
            img_list_nogrid_2 = img_list_nogrid[-1]
            show([img.squeeze() for img in img_list_nogrid_2[:5]],
                 save_path, save_fig, show_plot)


def color_for_lr(lr,
                 color_dict={1e-07: 'C6',
                             1e-06: 'C5',
                             1e-05: 'C4',
                             0.0001: 'C3',
                             0.001: 'C2',
                             0.01: 'C1',
                             0.1: 'C0'
                             }
                 ):
    '''
    returns correct color for a label in plot_FID
    '''
    color = color_dict[lr]
    return color


def plot_FID(x,  # = [0, 50, 100, 150, 200, 250, 290]
             y, save_fig, save_path, show_plot, handles=None, labels=None, x_fontsize=15, y_fontsize=15, yticks_size=15,
             xticks_size=15, label_size=15, show_legend=True):
    '''
    FID plots
    Input 
        x(list) : contains epochs at which FID was evaluated
        y(dict) : one of all_lr_scores_xxx where xxx is either adam, rmsprop or sgd
                  this dict can be loaded form the saved file all_lr_scores_xxx.pkl
    Output
        handles, labels : to be used if multiple plots with same label are desired
    '''
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for key, value in zip(y.keys(), y.values()):
        ax.plot(x,
                value, label='lr={:.0e}'.format(key), color=color_for_lr(key))

    ax.set_ylabel('FID', fontsize=y_fontsize)
    ax.set_xlabel('epoch', fontsize=x_fontsize)
    if show_legend:
        if handles is not None:
            fig.legend(handles, labels, fontsize=label_size)
        else:
            plt.legend(fontsize=label_size)
    plt.yticks(fontsize=yticks_size)
    plt.xticks(fontsize=xticks_size)
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()

    if save_fig == True:
        plt.savefig(save_path, format="pdf", dpi=400)
    if show_plot:
        plt.show()
    return handles, labels
