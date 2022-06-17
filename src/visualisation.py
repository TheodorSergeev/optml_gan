import numpy as np
import matplotlib.pyplot as plt

import torchvision.utils as vutils


def plot_loss(G_losses, D_losses, save_path, title_fontsize=12, x_fontsize=12, y_fontsize=12, yticks_size=10,
              xticks_size=10, save=False, show_plot=True):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training",
              fontsize=title_fontsize)
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations", fontsize=x_fontsize)
    plt.ylabel("Loss", fontsize=y_fontsize)
    plt.legend(fontsize=y_fontsize)
    plt.tight_layout()
    plt.yticks(fontsize=yticks_size)
    plt.xticks(fontsize=xticks_size)
    if save == True:
        plt.savefig(save_path, format="png", dpi=400)
    if show_plot:
        plt.show()


def plot_realvsfake(dataloader, device, img_list, save_path, save=False, show_plot=True):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
               :64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    if save == True:
        plt.savefig(save_path, format="png", dpi=400)
    if show_plot:
        plt.show()
