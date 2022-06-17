import torch.nn as nn


# Custom weights initialisation called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_net(model, device, ngpu):
    # Create the generator
    net = model.to(device)

    # Handle multi-gpu if desired
    if device.type == 'cuda' and ngpu > 1:
        net = nn.DataParallel(net, list(range(ngpu)))

    # Apply the weights_init function to randomly initialise all weights
    #  to mean=0, std=0.02.
    # net.apply(weights_init)

    # Print the model
    # print(net)
    return net
