# Project Title

## Description:

Optimization for machine learning mini-project

## Links:

Google drive: https://drive.google.com/drive/folders/17c7PySAorwY0P0VVEdMLnEwskU3yQMyT?usp=sharing

## Requirements:

```
ipython==8.4.0
matplotlib==3.5.1
numpy==1.21.5
protobuf==4.21.1
torch==1.11.0torchvision==0.12.0
```

## How to run:

If you want to run the code directly from google colab :

<a href="https://colab.research.google.com/github/TheodorSergeev/optml_gan/blob/main/dcgan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

TODO : the colab notebook can work on it's own ( check for paths issues, could just use os.makedir to create all the paths we need)

If you are running the code locally you can create an anaconda environement with the following commands. First create the environment:

```
conda create -y -n optmlgan python=3.7.13 scipy pandas numpy matplotlib
```

Then activate the environment:

```
conda activate optmlgan
conda install jupyterlab nb_conda_kernels
conda install -c conda-forge ipywidgets
```

If you have a gpu :

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

If you do not have a gpu :

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

TO DO : still didnt put all packages

# Contents of the Project:

Folder structure :

- `data`
- `generated_data`
- `img`
- `src `
