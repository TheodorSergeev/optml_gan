# Project Title

## Description

Optimization for Machine Learning Mini-Project

## Main findings

TO DO : maybe a plot from the report here if it's pretty?

## Links

...

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
 and open `gan.ipynb` or `metrics.ipynb` in Google Colab, the goold drive folder contains our saved experiments so that you wouldn't be required to train the models again (the gridsearch takes multiple hours even on a P100 GPU)



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

- `data`: This folder contains the datasets
- `generated_data`: If the option to serialise models and stats when running a grid search is activated, the models are saved in different subfolders of this folder
- `img`: This folder contains the plots in the report, produced from the code in this repository
- `src`: This folder contains the scripts required to run the experiments to generate the plots and results in our report
- `run.py`: Script that produces the plots and results presented in the report 
- `dcgan.ipynb`: Contains the code to produce the plots and results presented in the report with more in-depth explanations
