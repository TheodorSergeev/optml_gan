import json
import pickle
import torch

from src.model import init_net


def save_models(netG, netD, save_path_G, save_path_D):
    '''
    save the generator and discriminator models
    '''
    torch.save(netG.state_dict(), save_path_G)
    torch.save(netD.state_dict(), save_path_D)
    print('GAN saved')


def load_models(ngpu, Discriminator, Generator, save_path_G, save_path_D, nc, nz, loss_name, device):
    '''
    load the generator and discriminator models simultaneously
    '''
    netD = init_net(Discriminator(ngpu, nc, loss_name), device, ngpu)
    netD.load_state_dict(torch.load(save_path_D))
    netD.eval()

    netG = init_net(Generator(ngpu, nc, nz), device, ngpu)
    netG.load_state_dict(torch.load(save_path_G))
    netG.eval()

    print('GAN loaded')
    return netD, netG


def save_dict(dict, dict_path):
    with open(dict_path, 'w') as file:
        file.write(json.dumps(dict))


def read_dict(dict_path):
    with open(dict_path) as f:
        data = f.read()
    data = json.loads(data)
    return data


def pickle_save(something, path):
    with open(path, 'wb') as handle:
        pickle.dump(something, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path, 'rb') as handle:
        something = pickle.load(handle)
    return something
