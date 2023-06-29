# Example data

In this folder, we provide example data for the Jupyter notebook examples in the repositories `examples/` folder. The data is in netCDF format and has been prepared with the help of the CDO scripts, which are available at:

> https://github.com/andr-groth/CDO-scripts

## Data sources

### CMIP6

The data in the `cmip6/ssp/` folder is based on the CMIP6 SSP experiments. The different variables are stored in separate subfolders, .i.e, `cmip6/ssp/pr/` for precipitation and `cmip6/ssp/tos/` for sea-surface temperature.

The original CMIP6 data can be downloaded from the ESGF data portal with the help of the `wget` scripts, which are available in the `raw/` folders of the different subfolders.

The scripts are taken from the ESGF data portal at:

> https://aims2.llnl.gov/metagrid/search/?project=CMIP6

### Forcing data

The data in the `forcings/` folder is based on the CMIP6 SSP forcing data from:

> Riahi et al. (2017). The Shared Socioeconomic Pathways and their energy, land use, and greenhouse gas emissions implications: An overview. _Global Environmental Change_, __42__, 153-168. https://doi.org/10.1016/j.gloenvcha.2016.05.009.

### Crop data

The data in the `crop/` folder is based on the GGCMI crop calendar Phase 3 dataset, which is available at:

> https://zenodo.org/record/5062513

## Data preparation

### CMIP6

The data preparation of the CMIP6 data involves the following steps:

1. Download data from the ESGF data portal with the help of the `wget` scripts in the `raw/` folders (see above).
2. Merge the data into single files with the help of the `merge.sh` script from the CDO-scripts repository.
3. Create the anomalies, EOFs and PCs with the help of the `prepare_data.sh` script from the CDO-scripts repository.

The configuration of the `prepare_data.sh` script is stored in `anom.cfg` files in the variable subfolders, .i.e, `cmip6/ssp/pr/anom.cfg` for precipitation and `cmip6/ssp/tos/anom.cfg` for sea-surface temperature.

### Crop data

The data has been projected onto the CMIP6 EOFs with the help of the CDO tool `cdo remapcon`.
