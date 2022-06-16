import numpy as np
import matplotlib.pyplot as plt

import torchvision.utils as vutils


def plot_loss(G_losses, D_losses, PATH, save=False):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    if save == True:
        plt.savefig(PATH + 'img/loss', format="png", dpi=400)

    plt.show()


def plot_realvsfake(dataloader, device, img_list, PATH, save=False):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    if save == True:
        plt.savefig(PATH + 'img/real_vs_fake', format="png", dpi=400)
    plt.show()
