import os
import random
import torch


def count_parameters(model):
    # from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_repo_paths(PATH):
    data_path = PATH + '/data'
    generated_data_path = PATH + '/generated_data'
    img_path = PATH + '/img'
    src_path = PATH + '/src'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(generated_data_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(src_path, exist_ok=True)


# Paths to load and save the models
def generate_paths(PATH, extra_word, loss_name, lrD, lrG, beta1, iter_dis, iter_gen, grad_penalty_coef, create_dir):
    param_str = loss_name + 'Loss_' + 'lrd' + \
        str(lrD) + '_lrg' + str(lrG) + '_b1' + 'b' + str(beta1)
    param_str = param_str + '_itd' + \
        str(iter_dis) + '_itg' + str(iter_gen) + \
        '_gpv' + str(grad_penalty_coef) + '_'

    experiment_path = PATH + "generated_data/" + extra_word + param_str

    models_path = experiment_path + '/models/'

    stats_path = experiment_path + '/stat.pickle'

    if create_dir:
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(experiment_path, exist_ok=True)

    return experiment_path, stats_path, models_path


def model_paths(experiment_path, epoch, models_path):

    model_name_G = 'model_G_' + str(epoch)
    save_path_G = models_path + model_name_G

    model_name_D = 'model_D_'+str(epoch)
    save_path_D = models_path + model_name_D

    return save_path_G, save_path_D


def set_seeds(manualSeed=123):
    # Set random seed for reproducibility

    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
