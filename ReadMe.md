# Project Title

## Description

Generative adversarial networks (GANs) are a powerful tool for creating synthetic data that has been successfully used for tasks such as high fidelity image generatrion. However, GANs are notoriously difficult to train as the process is unstable, involves a lot of hyperparameters, and there are little to no theoretical guarantees for convergence. In this work, we systematically study how the choice of an optimiser and initial learning rate affects the speed and stability of GAN training with Wasserstein loss on the MNIST dataset. We show that the choice of an appropriate initial learning rate can have a bigger impact on the end quality of the generated images, while the choice of an appropriate optimiser can help to avoid loss explosions and make the training more stable.

<p align="center">
  <img width="800" height="350" src="https://github.com/TheodorSergeev/optml_gan/blob/aa8ebb5822128ca39377c1f96254e47774828f6d/img/readme_img.png">
</p>

## Requirements

```
ipython=8.4.0
matplotlib=3.5.1
numpy=1.21.5
torch=1.11.0
torchvision=0.12.0
scipy=1.7.3
pandas=1.3.5
jupyterlab=3.3.2
nb_conda_kernels=2.3.1
ipywidgets=7.7.0
sk-learn=1.1.1
tqdm=4.62.3
```
An Nvidia GPU or access to Google Colab to speed up training.

## How to run

If you want to run the code directly from Google Colab:

- Click on <a href="https://colab.research.google.com/github/TheodorSergeev/optml_gan/blob/main/gan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for  `gan.ipynb`
- Click on <a href="https://colab.research.google.com/github/TheodorSergeev/optml_gan/blob/main/metrics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for `metrics.ipynb`
- Or open [this Google Drive folder](https://drive.google.com/drive/folders/17c7PySAorwY0P0VVEdMLnEwskU3yQMyT?usp=sharing)
 and open `gan.ipynb` or `metrics.ipynb` in Google Colab, the goold drive folder contains our saved experiments so that you wouldn't be required to train the models again (the gridsearch takes multiple hours even on a P100 GPU). We recommend you download the files in this drive and place them in a folder with the same name in your own google drive.
- You can also use the provided `run.py` file to run an experiment where a single gan is trained and outputs some plots into the img directory


If you are running the code locally, you can create an Anaconda environment with the following commands. First create the environment:

```
conda create -y -n optmlgan python=3.7.13 scipy pandas numpy matplotlib
```

Then activate the environment:

```
conda activate optmlgan
conda install jupyterlab nb_conda_kernels
conda install -c conda-forge ipywidgets
pip install -U scikit-learn
```

If you have a GPU:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

If you do not have a GPU:

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

# Project Contents

Folder structure:

- `data`: This folder contains the MNIST dataset
- `generated_data`: All the saved models and generated stats are save in this folder. The gridsearch experiments are saved here with a structured folder name
- `img`: This folder contains the plots in the report, produced from the code in this repository
- `src`: This folder contains the scripts required to run the experiments to generate the plots and results in our report
- `run.py`: Script that produces the plots and results presented in the report 
- `dcgan.ipynb`: Contains the code to produce the plots and results presented in the report with more in-depth explanations
