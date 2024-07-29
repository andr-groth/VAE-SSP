# Decadal prediction of climate data subject to different GHG emission scenarios using a variational autoencoder

## Overview

Jupyter notebooks for the implementation of a variational autoencoder (VAE) for climate data modelling and prediction.

The Jupyter notebooks demonstrate the process of training and exploring a Variational Autoencoder (VAE) on precipitation and sea-surface temperature data. The VAE is trained on different Shared Socio-economic Pathways (SSPs) data of CMIP6.

## Requirements

1. The Jupyter notebooks requires the __VAE package__, which is available at:

    > https://github.com/andr-groth/VAE-project

2. Sample data used in the notebook is included in the [`data/`](/data/) folder. The data is in netCDF format and has been prepared with the help of the __CDO scripts__, which are available at:

    > https://andr-groth.github.io/CDO-scripts

    For more information on the data preparation see [`data/README.md`](/data/README.md).


## Usage

This repository contains two Jupyter notebooks for model training and prediction:

- To train a VAE on a range of different GHG emission scenarios: [`VAEp_train.ipynb`](VAEp_train.ipynb)

- To make predictions with the VAE on new GHG emission scenarios: [`VAEp_explore.ipynb`](VAEp_explore.ipynb)

The notebooks use data provided in the [`data/`](/data/) folder in this repository.