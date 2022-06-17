import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils


# there are problems with downloading CelebA
# see https://stackoverflow.com/questions/65528568/how-do-i-load-the-celeba-dataset-on-google-colab-using-torch-vision-without-ru

def get_dataset(name, image_size, dataroot):
    ''' Load either mnist or cifar10 as a dataset

    Input:
        name(str): either cifar10 or mnist
        image_size(int): 28 for mnist and 32 for cifar
        dataroot(str): where to save or where the datasets are already
    Output:
        dataset(torch.Dataset): 
        nc(int): numebr of channels
    '''
    dataset = None

    # number of channels in the training images (3 for colour, 1 for grayscale)
    nc = None

    if name == 'cifar10':
        nc = 3

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                                 )])

        dataset = torchvision.datasets.CIFAR10(dataroot, download=True,
                                               train=True,  transform=transform)

    elif name == 'mnist':
        nc = 1

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        dataset = torchvision.datasets.MNIST(dataroot, download=True,
                                             train=True,  transform=transform)

    else:
        raise ValueError("Unknown dataset name")

    return dataset, nc


def plot_img(dataloader, dataset_name, device, PATH):
    ''' Plot some training images'''
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                             padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(PATH + 'img/training_images_' +
                dataset_name, format="png", dpi=400)
