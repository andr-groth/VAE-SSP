# Decadal prediction of climate data subject to different GHG emission scenarios using a variational autoencoder

## Overview

Jupyter notebooks for the implementation of a variational autoencoder (VAE) for climate data modelling and prediction.

The Jupyter notebooks demonstrate the process of training and exploring a Variational Autoencoder (VAE) on precipitation and sea-surface temperature data. The VAE is trained on different Shared Socio-economic Pathways (SSPs) data of CMIP6.

The VAE framework is based on Groth & Chavez (2023). _submitted_.

## Requirements

1. The Jupyter notebooks requires the __VAE package__, which is available at:

    > https://github.com/andr-groth/VAE-project

2. Sample data used in the notebook is included in the `data/` folder. The data is in netCDF format and has been prepared with the help of the __CDO scripts__, which are available at:

    > https://andr-groth.github.io/cdo-scripts

    For more information on the data preparation see the `data/README.md` file.


## Examples

For example runs of the Jupyter notebooks see the `examples/` folder of this repository. The examples are based on the sample data in the `data/` folder.

- Example of training a VAE a range of different GHG emission scenarios: [`VAEp_train.ipynb`](https://github.com/andr-groth/VAE-SSP/blob/main/examples/VAEp_train.md)

- Example of exploring a VAE to make predictions on new GHG emission scenarios: [`VAEp_explore.ipynb`](https://github.com/andr-groth/VAE-SSP/blob/main/examples/VAEp_explore.md)