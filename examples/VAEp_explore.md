# Explore VAE on SSP data

The notebook demonstrates the process of exploring a variational auto-encoder (VAE) model trained on Shared Socioeconomic Pathways (SSPs) data from CMIP6. The process involves various steps, including model and data loading, analysis of the latent space, model output generation, and reconstruction in grid space. The key steps and components involved are outlined as follows:

1. The configuration parameters of the model are loaded from the `LOG_DIR` folder.

2. The VAE model consists of four components: _encoder_, _latent sampling_, _decoder_, and a _second decoder for prediction_. Separate model instances are created for each component:
    * _Encoder_: The encoder takes a sample `x` and returns the mean `z_mean` and logarithmic variance `z_log_var` of the latent variable `z`.
    * _Latent Sampling_: The latent sampling takes `z_mean` and `z_log_var` as inputs and generates a random latent sample `z`.
    * _Decoder_: The decoder reconstructs the input `x` by taking the latent sample `z` and producing the decoded output `y`. The decoding is done backward in time, maintaining the input order.
   * _Decoder for Prediction_: The second decoder also takes the latent sample `z` but generates a forward-time prediction output.
   
3. The model weights from the training process are loaded from the `LOG_DIR` folder.

4. CMIP data is loaded from netCDF files, with different variables stacked along the channel axis.

5. Forcing data is loaded from a CSV file used for model training. Additional forcing data is loaded from a second CSV file, which will be used as a new trajectory of the condition for the decoder and prediction.

6. The `encoder` and `decoder` properties are explored. KL divergence, histograms, and temporal behavior of the latent variables are analyzed. Invariance of the latent space with respect to forcing is tested.

7. The model outputs are obtained for the CMIP data and the new forcing data. The outputs of the `decoder` and `prediction` are collected separately and aligned with the target month. The VAE output is restricted to specific time lags for reducing file size.

8. The model output is projected into grid space by forming the scalar product with the EOFs (Empirical Orthogonal Functions).The corresponding climatological mean fields are loaded and added to obtain absolute values.

9. Selected model runs with the highest KL divergence are used to form the ensemble forecast. Various ensemble statistics are exported as as netCDF files in the specified `EXPORT_DIR` folder.

## Imports


```python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```


```python
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow.keras as ks
import yaml
from IPython.display import display
from matplotlib import dates
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy import signal
```


```python
from tensorflow import get_logger
from tensorflow.compat.v1 import disable_eager_execution, disable_v2_behavior

get_logger().setLevel('ERROR')
disable_eager_execution()
disable_v2_behavior()
```


```python
from VAE import generators, models
from VAE.utils import fileio
from VAE.utils import plot as vplt
```


```python
FIGWIDTH = 12
VERBOSE = 1
%config InlineBackend.figure_formats = ['retina']
%matplotlib inline
plt.style.use('seaborn-notebook')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 75
np.set_printoptions(formatter={'float_kind': lambda x: f'{x: .3f}'}, linewidth=120)
```

## Parameters

We load the configuration from the the folder `LOG_DIR`. The model output is written to netCDF files in the folder given in `EXPORT_DIR`.


```python
EPOCH = 20
LOG_DIR = r'logs/2023-05-30T18.58'
MODEL_FILE = f'model.{EPOCH:02d}.h5'
EXPORT_DIR = r'results/2023-05-30T18.58'
```


```python
print('LOG_DIR    :', LOG_DIR)
print('MODEL_FILE :', MODEL_FILE)
print('EXPORT_DIR :', EXPORT_DIR)
```

    LOG_DIR    : logs/2023-05-30T18.58
    MODEL_FILE : model.20.h5
    EXPORT_DIR : results/2023-05-30T18.58
    

First let's load the parameters from the model training in `trainer_config.yaml`.


```python
fn = os.path.join(LOG_DIR, 'trainer_config.yaml')
with open(fn, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

print('Load configuration from:', os.path.normpath(fn))

assert params['model'].get('__version__') == models.__version__, 'Model version mismatch.'
assert params['fit_generator'].get('__version__') == generators.__version__, 'Generator version mismatch.'

params = SimpleNamespace(**params)
```

    Load configuration from: logs\2023-05-30T18.58\trainer_config.yaml
    

Make some modifications to the parameters


```python
params.model['beta'] = 1.  # no beta scheduler needed at inference time
params.fit_generator['shuffle'] = False  # do not shuffle samples
# params.fit_generator['repeat_samples'] = 1
```


```python
# print(yaml.dump(params.__dict__))
```

## Model

The VAE model consists of four components: encoder, latent sampling, decoder, and a second decoder for prediction. Separate model instances are created for each component.

### Encoder

The encoder takes a sample `x` and returns `z_mean` and `z_log_var`:


```python
encoder = models.Encoder(**params.model, name='encoder')
```

### Latent sampling

The latent sampling takes the two inputs `z_mean` and `z_log_var` and returns a set of `set_size=1` random latent sample `z`:


```python
latent_sampling = models.LatentSampling(**params.model, name='latent')
```

### Decoder

The decoder, finally, takes a latent sample `z` and returns the decoded output `y` to reconstruct `x`. The decoding works backward in time and we set `output_reverse=True` so that the order of decoder output matches the input to the encoder.


```python
decoder = models.Decoder(output_shape=params.model.get('input_shape'),
                         decoder_blocks=params.model.get('encoder_blocks'),
                         output_reverse=True,
                         **params.model,
                         name='decoder')
```

### Decoder for prediction

Like the decoder, the second decoder takes the same latent sample `z` and it's output will provide the prediction. In contrast to the `decoder`, we set `output_reverse=False` so that the output of `prediction` is forward in time.


```python
prediction = models.Decoder(output_shape=params.model.get('prediction_shape'),
                            output_reverse=False,
                            **{
                                'decoder_blocks': params.model.get('encoder_blocks'),
                                **params.model,
                                **params.prediction
                            },
                            name='prediction')
```

### Full model

Now that we have the four components, we a ready to create the full model.


```python
model = models.VAEp(encoder, decoder, latent_sampling, prediction, **params.model)
```

Let's plot the model


```python
ks.utils.plot_model(model, show_shapes=True, dpi=75, rankdir='LR')
```




    
![png](VAEp_explore_files/VAEp_explore_35_0.png)
    



and summarize the model


```python
model.summary(line_length=120)
```

    Model: "mVAEp"
    ________________________________________________________________________________________________________________________
    Layer (type)                           Output Shape               Param #       Connected to                            
    ========================================================================================================================
    encoder_input (InputLayer)             [(None, 1, 12, 40)]        0                                                     
    ________________________________________________________________________________________________________________________
    encoder_cond (InputLayer)              [(None, 1, 57)]            0                                                     
    ________________________________________________________________________________________________________________________
    encoder (Functional)                   [(None, 32), (None, 32)]   185340        encoder_input[0][0]                     
                                                                                    encoder_cond[0][0]                      
    ________________________________________________________________________________________________________________________
    latent (Functional)                    (None, 1, 32)              0             encoder[0][0]                           
                                                                                    encoder[0][1]                           
    ________________________________________________________________________________________________________________________
    decoder_cond (InputLayer)              [(None, 1, 57)]            0                                                     
    ________________________________________________________________________________________________________________________
    prediction_cond (InputLayer)           [(None, 1, 57)]            0                                                     
    ________________________________________________________________________________________________________________________
    decoder (Functional)                   (None, 1, 12, 40)          213412        latent[0][0]                            
                                                                                    decoder_cond[0][0]                      
    ________________________________________________________________________________________________________________________
    prediction (Functional)                (None, 1, 12, 40)          213412        latent[0][0]                            
                                                                                    prediction_cond[0][0]                   
    ========================================================================================================================
    Total params: 612,164
    Trainable params: 609,620
    Non-trainable params: 2,544
    ________________________________________________________________________________________________________________________
    

### Load model weights

We load the model weights from the training.


```python
fn = os.path.join(LOG_DIR, MODEL_FILE)
model.load_weights(fn, by_name=True)
print('Load model weights from:', os.path.normpath(fn))
```

    Load model weights from: logs\2023-05-30T18.58\model.20.h5
    

## Data

### CMIP data

We load the netCDF files of CMIP data.


```python
_variables, _dimensions, _attributes = fileio.read_netcdf_multi(**params.data, num2date=True)
```

    data\cmip6\ssp\pr\pcs\pcs*.nc  : 100 file(s) found.
    data\cmip6\ssp\tos\pcs\pcs*.nc : 100 file(s) found.
    200/200 [==============================] - 20s 101ms/file
    

We group the netCDF files and their variables by the global attributes `source_id` + `experiment_id`. The attribute `source_id` refers to the model name (e.g. `ACCESS-CM2`) and the attribute `experiment_id` to the experiment (e.g. `ssp126`).


```python
variables = {}
dimensions = {}
attributes = {}

key1 = 'source_id'
key2 = 'experiment_id'
for dataset_name, values in _variables.items():
    target_key = (
        _attributes[dataset_name]['.'][key1],
        _attributes[dataset_name]['.'][key2],
    )

    variables.setdefault(target_key, {})
    dimensions.setdefault(target_key, {})
    attributes.setdefault(target_key, {})

    variables[target_key] |= {k: pd.DataFrame(v, index=_dimensions[dataset_name]['time']) for k, v in values.items()}
    dimensions[target_key] |= {k: _dimensions[dataset_name] for k in values.keys()}
    attributes[target_key] |= {k: _attributes[dataset_name] for k in values.keys()}
```

We make a few tests to check the integrity of the data.


```python
variable_names = {tuple(val.keys()) for val in variables.values()}
if len(variable_names) > 1:
    raise ValueError(f'More than one variable combination found: {variable_names}')
else:
    variable_names, *_ = variable_names
    print('\N{check mark} One variable combination found:', variable_names)

variable_channels = {tuple(v.shape[-1] for v in val.values()) for val in variables.values()}
if len(variable_channels) > 1:
    raise ValueError(f'More than one channel combination found: {variable_channels}')
else:
    variable_channels, *_ = variable_channels
    print('\N{check mark} One channel combination found:', variable_channels)
```

    ✓ One variable combination found: ('pr', 'tos')
    ✓ One channel combination found: (20, 20)
    

The following table summarizes the models and their different runs.


```python
print('Number of model runs found :', len(variables))
df = pd.crosstab(*list(zip(*list(variables.keys()))), rownames=[key1], colnames=[key2])
df.loc['--- Total ---'] = df.sum(axis=0)
display(df.replace(0, ''))
```

    Number of model runs found : 100
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>experiment_id</th>
      <th>ssp126</th>
      <th>ssp245</th>
      <th>ssp370</th>
      <th>ssp585</th>
    </tr>
    <tr>
      <th>source_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACCESS-CM2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ACCESS-ESM1-5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BCC-CSM2-MR</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CAMS-CSM1-0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CAS-ESM2-0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CESM2-WACCM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CMCC-CM2-SR5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CMCC-ESM2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CanESM5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CanESM5-1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>E3SM-1-0</th>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>E3SM-1-1</th>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>E3SM-1-1-ECA</th>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>FGOALS-f3-L</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>FGOALS-g3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GFDL-ESM4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>IITM-ESM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>INM-CM4-8</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>INM-CM5-0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>IPSL-CM5A2-INCA</th>
      <td>1</td>
      <td></td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>IPSL-CM6A-LR</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>KACE-1-0-G</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MIROC6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MPI-ESM1-2-LR</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MRI-ESM2-0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>NESM3</th>
      <td>1</td>
      <td>1</td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>NorESM2-LM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>NorESM2-MM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>--- Total ---</th>
      <td>25</td>
      <td>24</td>
      <td>24</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>


#### Plot data

In the following plot, we compare the different CMIP datasets.


```python
cols = 3
rows = 2

dataset_names = list(variables.keys())
_, first_idx, color_idx = np.unique([dataset_name[1] for dataset_name in dataset_names],
                                    return_index=True,
                                    return_inverse=True)

cm = plt.cm.get_cmap('tab10', 10)
color_dict = {name: cm(idx) for name, idx in zip(dataset_names, color_idx)}

for variable_name in variable_names:
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(FIGWIDTH, 3 * rows), squeeze=False)
    fig.suptitle(variable_name.upper(), fontweight='bold')
    for idx, dataset_name in enumerate(dataset_names):
        for channel, (ax, values) in enumerate(zip(axs.flat, variables[dataset_name][variable_name].values.T)):
            label = dataset_name[1] if idx in first_idx else None
            ax.plot(variables[dataset_name][variable_name].index,
                    values,
                    color=color_dict[dataset_name],
                    label=label,
                    alpha=0.5)

            ax.set_title(f'Channel {channel}')
            ax.grid(visible=True, linestyle=':')
            locator = dates.YearLocator(25)
            ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
            ax.xaxis.set_major_locator(locator)

    axs.flat[0].legend()
```


    
![png](VAEp_explore_files/VAEp_explore_53_0.png)
    



    
![png](VAEp_explore_files/VAEp_explore_53_1.png)
    


#### Stack variables

The variables are stack along the last axis, the channels. We add a leading singleton dimension to `dataset` for `set_size=1`.


```python
dataset_names_rand = list(variables.keys())
```


```python
data_stack = [
    pd.concat([variables[dataset_name][variable_name] for variable_name in variable_names], axis=1, join='inner')
    for dataset_name in dataset_names_rand
]

time = [d.index for d in data_stack]
dataset = [d.to_numpy()[None, ...] for d in data_stack]
```

For the training, the dataset was furthermore split into one set for model training and one set for model validation. At inference time, we keep the entier dataset in order to generate results for all CMIP runs.


```python
validation_split = 0
print('Size of training dataset   :', len(dataset[:validation_split]))
print('Size of validation dataset :', len(dataset[validation_split:]))
```

    Size of training dataset   : 0
    Size of validation dataset : 100
    

### Forcing data

#### SSP forcings

We load the forcing data from a csv file used to train the model. The first column is considered as the date and the remaining columns as the corresponding forcing data. The column header of the forcing data must match the `experiment_id`.


```python
filename = params.forcing['filename']
forcing_df = pd.read_csv(filename, index_col=0, parse_dates=True)
print('Load forcing data from:', os.path.relpath(filename))
display(forcing_df)
```

    Load forcing data from: data\cmip6\ssp\forcings\SSP_CMIP6_world_C02.csv
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ssp126</th>
      <th>ssp245</th>
      <th>ssp370</th>
      <th>ssp585</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>31652.73</td>
      <td>31648.76</td>
      <td>31648.76</td>
      <td>31652.73</td>
    </tr>
    <tr>
      <th>2010-01-01</th>
      <td>36452.73</td>
      <td>36448.76</td>
      <td>36448.76</td>
      <td>36452.73</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>39152.73</td>
      <td>39148.76</td>
      <td>39148.76</td>
      <td>39152.73</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>39804.01</td>
      <td>40647.53</td>
      <td>44808.04</td>
      <td>43712.35</td>
    </tr>
    <tr>
      <th>2030-01-01</th>
      <td>34734.42</td>
      <td>43476.06</td>
      <td>52847.36</td>
      <td>55296.58</td>
    </tr>
    <tr>
      <th>2040-01-01</th>
      <td>26509.18</td>
      <td>44252.90</td>
      <td>58497.97</td>
      <td>68775.70</td>
    </tr>
    <tr>
      <th>2050-01-01</th>
      <td>17963.54</td>
      <td>43462.19</td>
      <td>62904.06</td>
      <td>83298.22</td>
    </tr>
    <tr>
      <th>2060-01-01</th>
      <td>10527.98</td>
      <td>40196.48</td>
      <td>66568.37</td>
      <td>100338.61</td>
    </tr>
    <tr>
      <th>2070-01-01</th>
      <td>4476.33</td>
      <td>35235.43</td>
      <td>70041.98</td>
      <td>116805.25</td>
    </tr>
    <tr>
      <th>2080-01-01</th>
      <td>-3285.04</td>
      <td>26838.37</td>
      <td>73405.23</td>
      <td>129647.04</td>
    </tr>
    <tr>
      <th>2090-01-01</th>
      <td>-8385.18</td>
      <td>16324.39</td>
      <td>77799.05</td>
      <td>130576.24</td>
    </tr>
    <tr>
      <th>2100-01-01</th>
      <td>-8617.79</td>
      <td>9682.86</td>
      <td>82725.83</td>
      <td>126287.31</td>
    </tr>
  </tbody>
</table>
</div>


The forcing data is interpolated so that it can be resampled at the time points of the CMIP data.


```python
forcing_df_daily = forcing_df.asfreq('D').interpolate(method='akima')
```

Instead of providing only the current value of the forcing as condition, we can optionally provide past values as well. The parameter `input_length` in `params.forcing` determines the number of past values (in years) that will be provided as condition.


```python
scale = params.forcing.get('scale', 1)
input_length = params.forcing.get('input_length', 1)

forcing_data = [
    np.stack([
        forcing_df_daily[experiment_id].reindex(t - pd.DateOffset(years=years), method='nearest').to_numpy() * scale
        for years in range(input_length)
    ],
             axis=1) for t, (_, experiment_id) in zip(time, dataset_names_rand)
]
```

#### New forcings

We furthermore load forcing data from a second csv file that will be used as condition for `decoder` and `prediction`. The first column is considered as the date and the second column as the corresponding forcing data. The new forcing data is the same for all dataset, i.e. the same for different `experiment_id`.


```python
filename2 = r'data\cmip6\ssp\forcings\SSP_jumps_world_C02.csv'

forcing2_df = pd.read_csv(filename2, index_col=0, parse_dates=True)
print('Load forcing data from:', os.path.relpath(filename2))
display(forcing2_df)
```

    Load forcing data from: data\cmip6\ssp\forcings\SSP_jumps_world_C02.csv
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ssp_jumps</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>31648.76</td>
    </tr>
    <tr>
      <th>2010-01-01</th>
      <td>36448.76</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>44808.04</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>44808.04</td>
    </tr>
    <tr>
      <th>2030-01-01</th>
      <td>44808.04</td>
    </tr>
    <tr>
      <th>2039-01-01</th>
      <td>44808.04</td>
    </tr>
    <tr>
      <th>2040-01-01</th>
      <td>54252.90</td>
    </tr>
    <tr>
      <th>2050-01-01</th>
      <td>54252.90</td>
    </tr>
    <tr>
      <th>2060-01-01</th>
      <td>54252.90</td>
    </tr>
    <tr>
      <th>2069-01-01</th>
      <td>54252.90</td>
    </tr>
    <tr>
      <th>2070-01-01</th>
      <td>64252.90</td>
    </tr>
    <tr>
      <th>2080-01-01</th>
      <td>64252.90</td>
    </tr>
    <tr>
      <th>2090-01-01</th>
      <td>64252.90</td>
    </tr>
    <tr>
      <th>2100-01-01</th>
      <td>64252.90</td>
    </tr>
  </tbody>
</table>
</div>


The new forcing data is likewise interpolated and embedded as the original forcings.


```python
forcing2_df_daily = forcing2_df.asfreq('D').interpolate(method='akima')
forcing2_name, *_ = forcing2_df_daily.keys()

forcing2_data = [
    np.stack([
        forcing2_df_daily[forcing2_name].reindex(t - pd.DateOffset(years=years), method='nearest').to_numpy() * scale
        for years in range(input_length)
    ],
             axis=1) for t in time
]
```

#### Compare forcings

We compare the original SSP forcings with the modified forcing.


```python
fig, ax = plt.subplots(1, figsize=(FIGWIDTH / 2, FIGWIDTH / 3))
for n, (forcing_name, val) in enumerate(forcing_df.items()):
    ax.plot(val, 'o', color=cm(n), label=forcing_name)
    ax.plot(forcing_df_daily[forcing_name], '-', color=cm(n))

new_color = cm(len(forcing_df) + 1)
for n, (forcing_name, val) in enumerate(forcing2_df.items()):
    ax.plot(val, 'o', color=new_color, label=forcing_name)
    ax.plot(forcing2_df_daily[forcing_name], '-', color=new_color, linewidth=3)

ax.legend()
ax.grid(linestyle=':')
```


    
![png](VAEp_explore_files/VAEp_explore_75_0.png)
    


## Prepare generator

### Validation generator

The validation generator takes the full set of CMIP data.

During training, the same CO2 concentration data was provided as condition to the encoder and decoder corresponding to the input and target data they were trained on. During inference, here, different values of CO2 concentration data are provided as input to the decoder. This allows the VAE to generate forecasts for different CO2 concentration scenarios while sampling from the latent space.


```python
month = [t.to_numpy().astype('datetime64[M]').astype(int) for t in time]
_, model_index = np.unique([dataset_name[0] for dataset_name in dataset_names_rand], return_inverse=True)
```


```python
val_gen = generators.FitGenerator(dataset[validation_split:],
                                  **params.fit_generator,
                                  time=month[validation_split:],
                                  ensemble_index=model_index[validation_split:],
                                  condition={
                                      'encoder': forcing_data[validation_split:],
                                      'decoder': forcing2_data[validation_split:]
                                  })
val_gen.summary()
```

    Number of datasets : 100
    Total data size    : 4,079,520
    Total data length  : 101,988
    Strides            : 1
    Number of samples  : 99,688
    Batch size         : 128
    Number of batches  : 779
    Sample repetitions : 5
    Actual batch size  : 128 * 5 = 640
    Shuffle            : False
    Ensemble condition
      size : 29
      type : index
    Input channels     : all
    Predicted channels : all
    Output shapes
      inputs
        encoder_input    : (640, 1, 12, 40)
        encoder_cond     : (640, 1, 57)
        decoder_cond     : (640, 1, 57)
        prediction_cond  : (640, 1, 57)
      targets
        decoder          : (640, 1, 12, 40)
        prediction       : (640, 1, 12, 40)
    

## Latent space

The module `VAE.utils.plot` provides multiple functions to plot and analyze properties of the `encoder` and the `decoder`. First let's start with the `encoder` and explore properties of the latent space.

### KL divergence


```python
fig = plt.figure(0, figsize=(FIGWIDTH, 6))
fig, ax, z_order, kl_div = vplt.encoder_boxplot(encoder, val_gen, plottype='kl', name=0, verbose=VERBOSE)
```


    
![png](VAEp_explore_files/VAEp_explore_84_0.png)
    


The plot shows the KL divergence of the latent variables for each of the latent dimension separately. The dimensions are sorted in descending order of the KL divergence. Latent dimensions with a high KL divergence are more important for the reconstruction with the decoder. Latent dimensions that have a KL divergence close to zero are unused dimensions; i.e. they are practically not important for the reconstruction.

A good property is when we observe a separation into a few important and vanishing (zero KL) dimensions, which means that the VAE is operating in a so-called __polarized regime__. This avoids overfitting with too many latent dimensions.


```python
active_units = np.sum(np.mean(kl_div, axis=0) > 0.1)
print('Number of active units:', active_units)
```

    Number of active units: 13
    

### Temporal behavior

Next, we analyze the temporal behavior of the latent variables. In doing so, we obtain the latent variables of the validation dataset.


```python
z_mean, z_log_var = encoder.predict(val_gen, verbose=VERBOSE)
z_sample = latent_sampling.predict([z_mean, z_log_var])
z_sample = np.squeeze(z_sample)
```

The latent variables are split into the different model runs.


```python
val_gen_splits = np.cumsum([(len(t) - val_gen.input_length - val_gen.prediction_length + 1) * val_gen.repeat_samples
                            for t in time[validation_split:]][:-1])

z_sample_list = np.split(z_sample, val_gen_splits, axis=0)
```


```python
rows = 7
fig, axs = plt.subplots(rows,
                        2,
                        figsize=(FIGWIDTH, 2 * rows),
                        sharex='col',
                        sharey='col',
                        squeeze=False,
                        gridspec_kw={
                            'width_ratios': [2.5, 1],
                            'wspace': 0.1
                        })

r = val_gen.repeat_samples
nfft = 2**12
fs = 12

for idx, (t, z,
          dataset_name) in enumerate(zip(time[validation_split:], z_sample_list,
                                         dataset_names_rand[validation_split:])):
    label = dataset_name[1] if idx in first_idx else None
    for (lax, rax), k in zip(axs, z_order):
        lax.plot(t[val_gen.input_length:-val_gen.prediction_length + 1],
                 z[:, k].reshape(-1, r).mean(axis=-1),
                 color=color_dict[dataset_name])
        lax.set_ylabel(f'{k=}')
        lax.grid(axis='x', linestyle=':')

        f, pxx = signal.welch(z[:, k].reshape(-1, r).mean(axis=-1),
                              nfft=nfft,
                              fs=fs,
                              nperseg=512,
                              axis=0,
                              scaling='spectrum')
        rax.plot(f, pxx, color=color_dict[dataset_name], label=label)

axs[0, 1].legend()
lax.margins(x=0.005)
locator = dates.YearLocator(10)
ax.xaxis.set_major_locator(locator)
lax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
lax.xaxis.set_minor_locator(dates.YearLocator(2))

rax.set_xlim((0, 0.7))
rax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
rax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
_ = rax.set_xlabel('Cycles per year')
```


    
![png](VAEp_explore_files/VAEp_explore_92_0.png)
    


The plot shows the temporal dynamics of the latent variables and the corresponding power spectrum. The success of disentanglement can be seen in a clear spearation between different dynamics, e.g. different power spectra. In the leading three latent dimensions, we can identify a low-frequency component and an oscillatory pair. The oscillatory pair is characterized by similar power specra.


```python
k_trend = getattr(params, '__notes__', {}).get('k_trend', z_order[0])
k_pair = getattr(params, '__notes__', {}).get('k_pair', z_order[[1, 2]])
```

### Test invariance of latent space wrt forcing

In the following, we test whether the leading latent dimensions are not specific to the different forcings. If this is the case, the VAE has learned a posterior that is invariant to the different forcings. This, in turn, is important if we want the output of the VAE to depend only on the forcings, while drawing a random sample from the posterior aggregrated over different forcings.

#### Forcing vs. low-frequency component

First we analyze the low-frequency component.


```python
variable_name = variable_names[1]
k_pc = 0

fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH, FIGWIDTH * 0.75))
kw_args = dict(marker='.', markersize=2, linestyle='none')
for idx, (dataset_name, forcing,
          z) in enumerate(zip(dataset_names_rand[validation_split:], forcing_data[validation_split:], z_sample_list)):
    forcing = np.atleast_2d(forcing)[:, 0] / params.forcing.get('scale', 1)
    data = variables[dataset_name][variable_name].values
    label = dataset_name[1] if idx in first_idx else None
    axs[0, 0].plot(data[val_gen.input_length:-val_gen.prediction_length + 1, k_pc],
                   forcing[val_gen.input_length:-val_gen.prediction_length + 1],
                   color=color_dict[dataset_name],
                   label=label,
                   **kw_args)

    axs[0, 1].plot(z[:, k_trend].reshape(-1, r),
                   forcing[val_gen.input_length:-val_gen.prediction_length + 1],
                   color=color_dict[dataset_name],
                   **kw_args)

    axs[1, 1].plot(z[:, k_trend].reshape(-1, r),
                   data[val_gen.input_length:-val_gen.prediction_length + 1, k_pc],
                   color=color_dict[dataset_name],
                   **kw_args)

for ax in axs.flat:
    ax.grid(linestyle=':')
axs[1, 0].remove()

axs[0, 0].legend(markerscale=10, loc='lower right')
axs[0, 0].set_ylabel('Forcing')
axs[1, 1].set_xlabel(f'z(k={k_trend})')
axs[0, 0].set_xlabel(f'{variable_name.upper()}   channel {k_pc + 1}')
axs[1, 1].set_ylabel(f'{variable_name.upper()}   channel {k_pc + 1}')

invert = np.mean([z[-1, k_trend] for z in z_sample_list]) < 0
if invert:
    axs[0, 1].invert_xaxis()
    axs[1, 1].invert_xaxis()
```


    
![png](VAEp_explore_files/VAEp_explore_99_0.png)
    


In the upper left panel, we compare the forcing with the leading PC in the SST, i.e. the trend in the SST. There is clearly a strong relationship between the two. In the upper right panel, we compare the forcing with the low frequency component. We see that the distributions of the $z$ values have a strong overlap, indicating success of the VAE in learning an invariant representation of the low-frequency dynamics. In the lower right panel, the low-frequency component is compared to the SST trend. We see that the SST trend does not depend on the identified low-frequency component, indicating that the VAE has identified low-frequency dynamics that is not specific to the forcing.

#### Forcing vs. oscillatory mode

Next, we analyze the oscillatory pair.


```python
k0, k1 = k_pair

fig, ax = plt.subplots(1, figsize=(FIGWIDTH * 0.5, FIGWIDTH * 0.5))
for dataset_name, forcing, z in zip(dataset_names_rand[validation_split:], forcing_data[validation_split:],
                                    z_sample_list):
    forcing = np.atleast_2d(forcing)[:, 0] / params.forcing.get('scale', 1)
    ax.plot(z[:, k0].reshape(-1, r).mean(axis=-1),
            z[:, k1].reshape(-1, r).mean(axis=-1),
            linestyle='-',
            color=color_dict[dataset_name])

ax.set_xlabel(f'z(k={k0})')
ax.set_ylabel(f'z(k={k1})')
ax.grid(linestyle=':')
```


    
![png](VAEp_explore_files/VAEp_explore_103_0.png)
    


We see that the distributions of the $z$ values have a strong overlap, indicating success of the VAE in learning an invariant representation of the oscillatory dynamics as well.

## Model output

We first obtain the model outputs from the input data given to `val_gen`.


```python
xcs, ycs = model.predict(val_gen, verbose=VERBOSE)
```

The `decoder` and `prediction` outputs, `xcs` and `ycs`, are concatenated along the lag/lead dimension and the singleton dimension for `set_size=1` is removed.


```python
xcs = np.concatenate([xcs, ycs], axis=2)
xcs = np.squeeze(xcs, axis=1)
```

The model output is split into the different model runs.


```python
xcs_list = np.split(xcs, val_gen_splits, axis=0)
```

Then, the model output is aligned with the target month and split into the different variables. To reduce the later size of the netCDF files, we restrict the VAE output to specific time lags given in `export_lags`.


```python
export_lags = [-1]
# export_lags = np.arange(-val_gen.input_length, val_gen.prediction_length)
```


```python
channel_splits = np.cumsum(variable_channels)
level = np.arange(-val_gen.input_length, val_gen.prediction_length)
lag_idx = val_gen.input_length + np.array(export_lags)

level = level[lag_idx]

r = val_gen.repeat_samples
xcs_variables = {}
xcs_dimensions = {}
xcs_attributes = {}
for dataset_name, values, t in zip(dataset_names_rand[validation_split:], xcs_list, time[validation_split:]):
    # restrict to given time lags
    values = values[:, lag_idx, :]
    # average over repeat samples
    values = values.reshape(-1, r, *values.shape[1:]).mean(axis=1)
    # align  with target month
    values = np.pad(values, ((val_gen.input_length, val_gen.prediction_length - 1), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=np.nan)
    values = np.stack([np.roll(values[:, n, :], lag, axis=0) for n, lag in enumerate(level)], axis=1)
    # split channels into variables
    splits = np.split(values, channel_splits, axis=-1)

    xcs_variables[dataset_name] = dict(zip(variable_names, splits))
    xcs_dimensions[dataset_name] = {variable_name: {'time': t, 'level': level} for variable_name in variable_names}
    xcs_attributes[dataset_name] = {
        variable_name: {
            'level': {
                'long_name': 'Time lag',
                'units': '',
                'axis': 'Z'
            }
        }
        for variable_name in variable_names
    }
```

In the following plot, we show examples of the model output.


```python
lag_idx = np.searchsorted(export_lags, -1)
cols = 3
rows = 2

for variable_name in variable_names:
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(FIGWIDTH, 3 * rows), squeeze=False)
    fig.suptitle(variable_name.upper(), fontweight='bold')
    for idx, dataset_name in enumerate(dataset_names_rand[validation_split:]):
        for channel, (ax, values) in enumerate(zip(axs.flat, xcs_variables[dataset_name][variable_name].T)):
            label = dataset_name[1] if idx in first_idx else None
            ax.plot(xcs_dimensions[dataset_name][variable_name]['time'],
                    values.T[:, lag_idx],
                    color=color_dict[dataset_name],
                    label=label,
                    alpha=0.5)

            ax.set_title(f'Channel {channel}')
            ax.grid(visible=True, linestyle=':')
            locator = dates.YearLocator(25)
            ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
            ax.xaxis.set_major_locator(locator)

    axs.flat[0].legend()
```


    
![png](VAEp_explore_files/VAEp_explore_116_0.png)
    



    
![png](VAEp_explore_files/VAEp_explore_116_1.png)
    


Since the decoder is provided with a different CO2 concentration scenario than the one used for the encoder, the decoder output differs from the input data to the encoder. Instead, the decoder output provides a forecast for the given CO2 concentration scenario.

## Reconstruction in grid space

In the following, the model output is projected into the grid space by forming the scalar product of the model output with the EOFs. The reults are exported as netCDF files.

### Load EOFs

First, we load the EOFs from the `eofs.nc` files, which can also be found in the data folders.


```python
eof_files = [os.path.join(os.path.dirname(filename), 'eofs.nc') for filename in params.data['filename']]
_eof_variables, _eof_dimensions, _eof_attributes = fileio.read_netcdf_multi(filename=eof_files,
                                                                            time_range=params.data.get('level_range'),
                                                                            dtype=params.data.get('dtype'))
```

    data\cmip6\ssp\pr\pcs\eofs.nc  : 1 file(s) found.
    data\cmip6\ssp\tos\pcs\eofs.nc : 1 file(s) found.
    2/2 [==============================] - 0s 16ms/file
    


```python
eof_variables = {}
eof_dimensions = {}
eof_attributes = {}
for dataset_name, values in _eof_variables.items():
    eof_variables |= values

    eof_dimensions |= {k: _eof_dimensions[dataset_name] for k in values}
    eof_attributes |= {k: _eof_attributes[dataset_name] for k in values}
```

### Load climatological mean

To obtain absolute values, we also load the corresponding climatological mean fields. The netCDF files will be looked up in the folder `mean_path`, relative to the data folder.


```python
mean_path = '../mean/*.nc'
mean_files = [os.path.join(os.path.dirname(filename), mean_path) for filename in params.data['filename']]
_mean_variables, _mean_dimensions, _mean_attributes = fileio.read_netcdf_multi(filename=mean_files,
                                                                               num2date=True,
                                                                               dtype=params.data.get('dtype'))
```

    data\cmip6\ssp\pr\mean\*.nc  : 100 file(s) found.
    data\cmip6\ssp\tos\mean\*.nc : 100 file(s) found.
    200/200 [==============================] - 3s 14ms/file
    

We group the netCDF files and their variables by the global attributes `source_id` + `experiment_id`, as for the CMIP data.


```python
mean_variables = {}
mean_dimensions = {}
mean_attributes = {}

key1 = 'source_id'
key2 = 'experiment_id'
for dataset_name, values in _mean_variables.items():
    target_key = (
        _mean_attributes[dataset_name]['.'][key1],
        _mean_attributes[dataset_name]['.'][key2],
    )

    mean_variables.setdefault(target_key, {})
    mean_dimensions.setdefault(target_key, {})
    mean_attributes.setdefault(target_key, {})

    mean_variables[target_key] |= values
    mean_dimensions[target_key] |= {k: _mean_dimensions[dataset_name] for k in values}
    mean_attributes[target_key] |= {k: _mean_attributes[dataset_name] for k in values}
```

### Select model runs

We restrict the export of netCDF files to model runs with highest KL divergence, i.e. the model runs that are most important in the construction of the aggregated posterior. We exclude the low-frequency component from the calculation of the mean KL divergence.


```python
topk = 5
```


```python
kl_div_list = np.split(kl_div, val_gen_splits, axis=0)
k = list(set(z_order) - set([k_trend]))
# k = z_order
kl_div_mean = np.array([kl[:, k].mean() for kl in kl_div_list])
```


```python
source_ids, experiment_ids = list(zip(*dataset_names_rand))

df = pd.DataFrame({'source_id': source_ids, 'experiment_id': experiment_ids, 'KL div': kl_div_mean})
df = df.pivot(values='KL div', index='source_id', columns='experiment_id')
mean = df.mean(axis=1).sort_values(ascending=False)
df = df.reindex(mean.index)

display(
    df.style.background_gradient(
        'coolwarm',
        text_color_threshold=0,
        axis=0,
    ).highlight_quantile(
        axis=0,
        q_left=(len(df) - topk) / len(df),
        props='font-weight:900',
    ).format(precision=2))
```


<style type="text/css">
#T_5ee3b_row0_col0 {
  background-color: #dd5f4b;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row0_col1, #T_5ee3b_row0_col2, #T_5ee3b_row0_col3, #T_5ee3b_row1_col0 {
  background-color: #b40426;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row1_col1, #T_5ee3b_row2_col3 {
  background-color: #d0473d;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row1_col2 {
  background-color: #bd1f2d;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row1_col3 {
  background-color: #d24b40;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row2_col0 {
  background-color: #e9785d;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row2_col1 {
  background-color: #de614d;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row2_col2 {
  background-color: #df634e;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row3_col0 {
  background-color: #f39778;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row3_col1, #T_5ee3b_row5_col2, #T_5ee3b_row5_col3 {
  background-color: #f0cdbb;
  color: #000000;
}
#T_5ee3b_row3_col2 {
  background-color: #ef886b;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row3_col3 {
  background-color: #f7b89c;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row4_col0 {
  background-color: #f3c7b1;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row4_col1 {
  background-color: #f1ccb8;
  color: #000000;
}
#T_5ee3b_row4_col2 {
  background-color: #f3c8b2;
  color: #000000;
}
#T_5ee3b_row4_col3 {
  background-color: #f5c0a7;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row5_col0, #T_5ee3b_row10_col3 {
  background-color: #d7dce3;
  color: #000000;
}
#T_5ee3b_row5_col1 {
  background-color: #f7ac8e;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row6_col0 {
  background-color: #edd2c3;
  color: #000000;
}
#T_5ee3b_row6_col1 {
  background-color: #f08b6e;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row6_col2, #T_5ee3b_row13_col2 {
  background-color: #d4dbe6;
  color: #000000;
}
#T_5ee3b_row6_col3, #T_5ee3b_row9_col3 {
  background-color: #d3dbe7;
  color: #000000;
}
#T_5ee3b_row7_col0 {
  background-color: #f3c7b1;
  color: #000000;
}
#T_5ee3b_row7_col1 {
  background-color: #cdd9ec;
  color: #000000;
}
#T_5ee3b_row7_col2 {
  background-color: #f7ac8e;
  color: #000000;
}
#T_5ee3b_row7_col3, #T_5ee3b_row10_col0 {
  background-color: #e6d7cf;
  color: #000000;
}
#T_5ee3b_row8_col0, #T_5ee3b_row11_col0, #T_5ee3b_row12_col3 {
  background-color: #c7d7f0;
  color: #000000;
}
#T_5ee3b_row8_col1, #T_5ee3b_row8_col3, #T_5ee3b_row14_col0, #T_5ee3b_row14_col1, #T_5ee3b_row14_col2, #T_5ee3b_row18_col0, #T_5ee3b_row18_col1, #T_5ee3b_row18_col2, #T_5ee3b_row19_col2, #T_5ee3b_row20_col0, #T_5ee3b_row20_col1, #T_5ee3b_row20_col2 {
  background-color: #000000;
  color: #000000;
}
#T_5ee3b_row8_col2 {
  background-color: #f7aa8c;
  color: #000000;
  font-weight: 900;
}
#T_5ee3b_row9_col0 {
  background-color: #efcfbf;
  color: #000000;
}
#T_5ee3b_row9_col1 {
  background-color: #edd1c2;
  color: #000000;
}
#T_5ee3b_row9_col2 {
  background-color: #ecd3c5;
  color: #000000;
}
#T_5ee3b_row10_col1 {
  background-color: #aec9fc;
  color: #000000;
}
#T_5ee3b_row10_col2 {
  background-color: #dedcdb;
  color: #000000;
}
#T_5ee3b_row11_col1 {
  background-color: #b9d0f9;
  color: #000000;
}
#T_5ee3b_row11_col2 {
  background-color: #d8dce2;
  color: #000000;
}
#T_5ee3b_row11_col3 {
  background-color: #e2dad5;
  color: #000000;
}
#T_5ee3b_row12_col0 {
  background-color: #dcdddd;
  color: #000000;
}
#T_5ee3b_row12_col1 {
  background-color: #b1cbfc;
  color: #000000;
}
#T_5ee3b_row12_col2 {
  background-color: #e4d9d2;
  color: #000000;
}
#T_5ee3b_row13_col0 {
  background-color: #bad0f8;
  color: #000000;
}
#T_5ee3b_row13_col1 {
  background-color: #e1dad6;
  color: #000000;
}
#T_5ee3b_row13_col3, #T_5ee3b_row14_col3, #T_5ee3b_row16_col3 {
  background-color: #c5d6f2;
  color: #000000;
}
#T_5ee3b_row15_col0 {
  background-color: #c1d4f4;
  color: #000000;
}
#T_5ee3b_row15_col1 {
  background-color: #a1c0ff;
  color: #000000;
}
#T_5ee3b_row15_col2 {
  background-color: #ccd9ed;
  color: #000000;
}
#T_5ee3b_row15_col3 {
  background-color: #c9d7f0;
  color: #000000;
}
#T_5ee3b_row16_col0, #T_5ee3b_row18_col3 {
  background-color: #a6c4fe;
  color: #000000;
}
#T_5ee3b_row16_col1 {
  background-color: #b2ccfb;
  color: #000000;
}
#T_5ee3b_row16_col2 {
  background-color: #c6d6f1;
  color: #000000;
}
#T_5ee3b_row17_col0 {
  background-color: #bbd1f8;
  color: #000000;
}
#T_5ee3b_row17_col1 {
  background-color: #85a8fc;
  color: #000000;
}
#T_5ee3b_row17_col2 {
  background-color: #cbd8ee;
  color: #000000;
}
#T_5ee3b_row17_col3 {
  background-color: #b3cdfb;
  color: #000000;
}
#T_5ee3b_row19_col0 {
  background-color: #98b9ff;
  color: #000000;
}
#T_5ee3b_row19_col1 {
  background-color: #a3c2fe;
  color: #000000;
}
#T_5ee3b_row19_col3 {
  background-color: #a9c6fd;
  color: #000000;
}
#T_5ee3b_row20_col3, #T_5ee3b_row22_col1, #T_5ee3b_row23_col0 {
  background-color: #94b6ff;
  color: #000000;
}
#T_5ee3b_row21_col0 {
  background-color: #b7cff9;
  color: #000000;
}
#T_5ee3b_row21_col1 {
  background-color: #84a7fc;
  color: #000000;
}
#T_5ee3b_row21_col2 {
  background-color: #92b4fe;
  color: #000000;
}
#T_5ee3b_row21_col3 {
  background-color: #8db0fe;
  color: #000000;
}
#T_5ee3b_row22_col0 {
  background-color: #7a9df8;
  color: #000000;
}
#T_5ee3b_row22_col2, #T_5ee3b_row22_col3 {
  background-color: #9bbcff;
  color: #000000;
}
#T_5ee3b_row23_col1 {
  background-color: #81a4fb;
  color: #000000;
}
#T_5ee3b_row23_col2 {
  background-color: #779af7;
  color: #000000;
}
#T_5ee3b_row23_col3 {
  background-color: #97b8ff;
  color: #000000;
}
#T_5ee3b_row24_col0 {
  background-color: #7699f6;
  color: #000000;
}
#T_5ee3b_row24_col1, #T_5ee3b_row24_col3, #T_5ee3b_row27_col2 {
  background-color: #455cce;
  color: #000000;
}
#T_5ee3b_row24_col2, #T_5ee3b_row25_col1 {
  background-color: #485fd1;
  color: #000000;
}
#T_5ee3b_row25_col0 {
  background-color: #4b64d5;
  color: #000000;
}
#T_5ee3b_row25_col2 {
  background-color: #536edd;
  color: #000000;
}
#T_5ee3b_row25_col3 {
  background-color: #4358cb;
  color: #000000;
}
#T_5ee3b_row26_col0 {
  background-color: #6180e9;
  color: #000000;
}
#T_5ee3b_row26_col1, #T_5ee3b_row26_col2, #T_5ee3b_row26_col3, #T_5ee3b_row27_col0, #T_5ee3b_row27_col1 {
  background-color: #3b4cc0;
  color: #000000;
}
#T_5ee3b_row27_col3 {
  background-color: #3f53c6;
  color: #000000;
}
</style>
<table id="T_5ee3b">
  <thead>
    <tr>
      <th class="index_name level0" >experiment_id</th>
      <th id="T_5ee3b_level0_col0" class="col_heading level0 col0" >ssp126</th>
      <th id="T_5ee3b_level0_col1" class="col_heading level0 col1" >ssp245</th>
      <th id="T_5ee3b_level0_col2" class="col_heading level0 col2" >ssp370</th>
      <th id="T_5ee3b_level0_col3" class="col_heading level0 col3" >ssp585</th>
    </tr>
    <tr>
      <th class="index_name level0" >source_id</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5ee3b_level0_row0" class="row_heading level0 row0" >NorESM2-MM</th>
      <td id="T_5ee3b_row0_col0" class="data row0 col0" >0.23</td>
      <td id="T_5ee3b_row0_col1" class="data row0 col1" >0.26</td>
      <td id="T_5ee3b_row0_col2" class="data row0 col2" >0.26</td>
      <td id="T_5ee3b_row0_col3" class="data row0 col3" >0.26</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row1" class="row_heading level0 row1" >CESM2-WACCM</th>
      <td id="T_5ee3b_row1_col0" class="data row1 col0" >0.25</td>
      <td id="T_5ee3b_row1_col1" class="data row1 col1" >0.25</td>
      <td id="T_5ee3b_row1_col2" class="data row1 col2" >0.25</td>
      <td id="T_5ee3b_row1_col3" class="data row1 col3" >0.25</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row2" class="row_heading level0 row2" >NorESM2-LM</th>
      <td id="T_5ee3b_row2_col0" class="data row2 col0" >0.23</td>
      <td id="T_5ee3b_row2_col1" class="data row2 col1" >0.24</td>
      <td id="T_5ee3b_row2_col2" class="data row2 col2" >0.24</td>
      <td id="T_5ee3b_row2_col3" class="data row2 col3" >0.25</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row3" class="row_heading level0 row3" >FGOALS-f3-L</th>
      <td id="T_5ee3b_row3_col0" class="data row3 col0" >0.22</td>
      <td id="T_5ee3b_row3_col1" class="data row3 col1" >0.20</td>
      <td id="T_5ee3b_row3_col2" class="data row3 col2" >0.23</td>
      <td id="T_5ee3b_row3_col3" class="data row3 col3" >0.21</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row4" class="row_heading level0 row4" >MIROC6</th>
      <td id="T_5ee3b_row4_col0" class="data row4 col0" >0.20</td>
      <td id="T_5ee3b_row4_col1" class="data row4 col1" >0.20</td>
      <td id="T_5ee3b_row4_col2" class="data row4 col2" >0.21</td>
      <td id="T_5ee3b_row4_col3" class="data row4 col3" >0.21</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row5" class="row_heading level0 row5" >IITM-ESM</th>
      <td id="T_5ee3b_row5_col0" class="data row5 col0" >0.18</td>
      <td id="T_5ee3b_row5_col1" class="data row5 col1" >0.22</td>
      <td id="T_5ee3b_row5_col2" class="data row5 col2" >0.21</td>
      <td id="T_5ee3b_row5_col3" class="data row5 col3" >0.20</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row6" class="row_heading level0 row6" >MRI-ESM2-0</th>
      <td id="T_5ee3b_row6_col0" class="data row6 col0" >0.19</td>
      <td id="T_5ee3b_row6_col1" class="data row6 col1" >0.23</td>
      <td id="T_5ee3b_row6_col2" class="data row6 col2" >0.19</td>
      <td id="T_5ee3b_row6_col3" class="data row6 col3" >0.19</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row7" class="row_heading level0 row7" >ACCESS-ESM1-5</th>
      <td id="T_5ee3b_row7_col0" class="data row7 col0" >0.20</td>
      <td id="T_5ee3b_row7_col1" class="data row7 col1" >0.18</td>
      <td id="T_5ee3b_row7_col2" class="data row7 col2" >0.22</td>
      <td id="T_5ee3b_row7_col3" class="data row7 col3" >0.20</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row8" class="row_heading level0 row8" >IPSL-CM5A2-INCA</th>
      <td id="T_5ee3b_row8_col0" class="data row8 col0" >0.18</td>
      <td id="T_5ee3b_row8_col1" class="data row8 col1" >nan</td>
      <td id="T_5ee3b_row8_col2" class="data row8 col2" >0.22</td>
      <td id="T_5ee3b_row8_col3" class="data row8 col3" >nan</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row9" class="row_heading level0 row9" >KACE-1-0-G</th>
      <td id="T_5ee3b_row9_col0" class="data row9 col0" >0.20</td>
      <td id="T_5ee3b_row9_col1" class="data row9 col1" >0.20</td>
      <td id="T_5ee3b_row9_col2" class="data row9 col2" >0.20</td>
      <td id="T_5ee3b_row9_col3" class="data row9 col3" >0.19</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row10" class="row_heading level0 row10" >CMCC-ESM2</th>
      <td id="T_5ee3b_row10_col0" class="data row10 col0" >0.19</td>
      <td id="T_5ee3b_row10_col1" class="data row10 col1" >0.17</td>
      <td id="T_5ee3b_row10_col2" class="data row10 col2" >0.19</td>
      <td id="T_5ee3b_row10_col3" class="data row10 col3" >0.19</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row11" class="row_heading level0 row11" >CAMS-CSM1-0</th>
      <td id="T_5ee3b_row11_col0" class="data row11 col0" >0.18</td>
      <td id="T_5ee3b_row11_col1" class="data row11 col1" >0.17</td>
      <td id="T_5ee3b_row11_col2" class="data row11 col2" >0.19</td>
      <td id="T_5ee3b_row11_col3" class="data row11 col3" >0.20</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row12" class="row_heading level0 row12" >CMCC-CM2-SR5</th>
      <td id="T_5ee3b_row12_col0" class="data row12 col0" >0.19</td>
      <td id="T_5ee3b_row12_col1" class="data row12 col1" >0.17</td>
      <td id="T_5ee3b_row12_col2" class="data row12 col2" >0.20</td>
      <td id="T_5ee3b_row12_col3" class="data row12 col3" >0.18</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row13" class="row_heading level0 row13" >GFDL-ESM4</th>
      <td id="T_5ee3b_row13_col0" class="data row13 col0" >0.17</td>
      <td id="T_5ee3b_row13_col1" class="data row13 col1" >0.19</td>
      <td id="T_5ee3b_row13_col2" class="data row13 col2" >0.19</td>
      <td id="T_5ee3b_row13_col3" class="data row13 col3" >0.18</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row14" class="row_heading level0 row14" >E3SM-1-1</th>
      <td id="T_5ee3b_row14_col0" class="data row14 col0" >nan</td>
      <td id="T_5ee3b_row14_col1" class="data row14 col1" >nan</td>
      <td id="T_5ee3b_row14_col2" class="data row14 col2" >nan</td>
      <td id="T_5ee3b_row14_col3" class="data row14 col3" >0.18</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row15" class="row_heading level0 row15" >MPI-ESM1-2-LR</th>
      <td id="T_5ee3b_row15_col0" class="data row15 col0" >0.17</td>
      <td id="T_5ee3b_row15_col1" class="data row15 col1" >0.16</td>
      <td id="T_5ee3b_row15_col2" class="data row15 col2" >0.19</td>
      <td id="T_5ee3b_row15_col3" class="data row15 col3" >0.18</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row16" class="row_heading level0 row16" >ACCESS-CM2</th>
      <td id="T_5ee3b_row16_col0" class="data row16 col0" >0.16</td>
      <td id="T_5ee3b_row16_col1" class="data row16 col1" >0.17</td>
      <td id="T_5ee3b_row16_col2" class="data row16 col2" >0.18</td>
      <td id="T_5ee3b_row16_col3" class="data row16 col3" >0.18</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row17" class="row_heading level0 row17" >IPSL-CM6A-LR</th>
      <td id="T_5ee3b_row17_col0" class="data row17 col0" >0.17</td>
      <td id="T_5ee3b_row17_col1" class="data row17 col1" >0.15</td>
      <td id="T_5ee3b_row17_col2" class="data row17 col2" >0.19</td>
      <td id="T_5ee3b_row17_col3" class="data row17 col3" >0.18</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row18" class="row_heading level0 row18" >E3SM-1-0</th>
      <td id="T_5ee3b_row18_col0" class="data row18 col0" >nan</td>
      <td id="T_5ee3b_row18_col1" class="data row18 col1" >nan</td>
      <td id="T_5ee3b_row18_col2" class="data row18 col2" >nan</td>
      <td id="T_5ee3b_row18_col3" class="data row18 col3" >0.17</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row19" class="row_heading level0 row19" >NESM3</th>
      <td id="T_5ee3b_row19_col0" class="data row19 col0" >0.16</td>
      <td id="T_5ee3b_row19_col1" class="data row19 col1" >0.17</td>
      <td id="T_5ee3b_row19_col2" class="data row19 col2" >nan</td>
      <td id="T_5ee3b_row19_col3" class="data row19 col3" >0.17</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row20" class="row_heading level0 row20" >E3SM-1-1-ECA</th>
      <td id="T_5ee3b_row20_col0" class="data row20 col0" >nan</td>
      <td id="T_5ee3b_row20_col1" class="data row20 col1" >nan</td>
      <td id="T_5ee3b_row20_col2" class="data row20 col2" >nan</td>
      <td id="T_5ee3b_row20_col3" class="data row20 col3" >0.16</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row21" class="row_heading level0 row21" >CAS-ESM2-0</th>
      <td id="T_5ee3b_row21_col0" class="data row21 col0" >0.17</td>
      <td id="T_5ee3b_row21_col1" class="data row21 col1" >0.15</td>
      <td id="T_5ee3b_row21_col2" class="data row21 col2" >0.16</td>
      <td id="T_5ee3b_row21_col3" class="data row21 col3" >0.16</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row22" class="row_heading level0 row22" >FGOALS-g3</th>
      <td id="T_5ee3b_row22_col0" class="data row22 col0" >0.15</td>
      <td id="T_5ee3b_row22_col1" class="data row22 col1" >0.16</td>
      <td id="T_5ee3b_row22_col2" class="data row22 col2" >0.17</td>
      <td id="T_5ee3b_row22_col3" class="data row22 col3" >0.17</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row23" class="row_heading level0 row23" >BCC-CSM2-MR</th>
      <td id="T_5ee3b_row23_col0" class="data row23 col0" >0.16</td>
      <td id="T_5ee3b_row23_col1" class="data row23 col1" >0.15</td>
      <td id="T_5ee3b_row23_col2" class="data row23 col2" >0.16</td>
      <td id="T_5ee3b_row23_col3" class="data row23 col3" >0.16</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row24" class="row_heading level0 row24" >CanESM5-1</th>
      <td id="T_5ee3b_row24_col0" class="data row24 col0" >0.15</td>
      <td id="T_5ee3b_row24_col1" class="data row24 col1" >0.13</td>
      <td id="T_5ee3b_row24_col2" class="data row24 col2" >0.14</td>
      <td id="T_5ee3b_row24_col3" class="data row24 col3" >0.13</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row25" class="row_heading level0 row25" >INM-CM5-0</th>
      <td id="T_5ee3b_row25_col0" class="data row25 col0" >0.13</td>
      <td id="T_5ee3b_row25_col1" class="data row25 col1" >0.13</td>
      <td id="T_5ee3b_row25_col2" class="data row25 col2" >0.14</td>
      <td id="T_5ee3b_row25_col3" class="data row25 col3" >0.13</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row26" class="row_heading level0 row26" >CanESM5</th>
      <td id="T_5ee3b_row26_col0" class="data row26 col0" >0.14</td>
      <td id="T_5ee3b_row26_col1" class="data row26 col1" >0.12</td>
      <td id="T_5ee3b_row26_col2" class="data row26 col2" >0.13</td>
      <td id="T_5ee3b_row26_col3" class="data row26 col3" >0.13</td>
    </tr>
    <tr>
      <th id="T_5ee3b_level0_row27" class="row_heading level0 row27" >INM-CM4-8</th>
      <td id="T_5ee3b_row27_col0" class="data row27 col0" >0.13</td>
      <td id="T_5ee3b_row27_col1" class="data row27 col1" >0.12</td>
      <td id="T_5ee3b_row27_col2" class="data row27 col2" >0.14</td>
      <td id="T_5ee3b_row27_col3" class="data row27 col3" >0.13</td>
    </tr>
  </tbody>
</table>



The table shows the KL divergence averaged over the different model runs. In each scenario, we select the `topk` model runs with the highest KL divergence (in bold).


```python
dataset_names_topk = []
for column in df:
    names = df[column].dropna().sort_values(ascending=False).iloc[:topk].index.tolist()
    dataset_names_topk += [(name, column) for name in names]
```

### Export to netCDF

We form the dot product of the selected model outputs with the EOFs and add the climatological mean. Optionally, the log transform is reverted. The result is then written to netCDF files in the folder initially given in `EXPORT_DIR`.


```python
os.makedirs(EXPORT_DIR, exist_ok=True)
```


```python
sup_time = np.unique(np.concatenate(time).astype('datetime64[M]'))

for variable_name in variable_names:
    print('-' * 3, variable_name, '-' * (77 - len(variable_name)))
    filename = '{prefix:s}' + variable_name + '.{type:s}.nc'
    filename = os.path.join(EXPORT_DIR, filename)

    # align model outputs
    values = []
    for dataset_name in dataset_names_topk:
        _value = xcs_variables[dataset_name][variable_name]
        _time = xcs_dimensions[dataset_name][variable_name]['time'].to_numpy().astype('datetime64[M]')
        idx = np.isin(sup_time, _time, assume_unique=True)
        value = np.full_like(_value, np.nan, shape=(len(sup_time), *_value.shape[1:]))
        value[idx, ...] = _value

        values.append(value)
        nc_dimensions = mean_dimensions[dataset_name][variable_name] | dimensions[dataset_name][
            variable_name] | xcs_dimensions[dataset_name][variable_name]
        nc_attributes = mean_attributes[dataset_name][variable_name] | attributes[dataset_name][
            variable_name] | xcs_attributes[dataset_name][variable_name]

    values = np.stack(values)
    # scalar product
    nc_variables = np.tensordot(values, eof_variables[variable_name], axes=1)

    nc_dimensions['time'] = sup_time
    kwargs = dict(dimensions=nc_dimensions, attributes=nc_attributes)

    # save anomalies
    fileio.write_netcdf(filename.format(prefix='anom_', type='ensmean'),
                        variables={variable_name: np.mean(nc_variables, axis=0)},
                        **kwargs)

    prcs = {'ensmedian': 50, 'enspctl10': 10, 'enspctl90': 90}
    nc_prcs = np.percentile(nc_variables, list(prcs.values()), axis=0)
    for type, value in zip(prcs, nc_prcs):
        fileio.write_netcdf(filename.format(prefix='anom_', type=type), variables={variable_name: value}, **kwargs)

    # revert to absolute values
    months = pd.to_datetime(nc_dimensions['time']).month
    for nc_variable, dataset_name in zip(nc_variables, dataset_names_topk):
        mean_months = pd.to_datetime(mean_dimensions[dataset_name][variable_name]['time']).month
        for month in mean_months:
            nc_variable[months == month, ...] += mean_variables[dataset_name][variable_name][mean_months == month, ...]

    # invert log transform
    if '-log ' in nc_attributes['.']['history']:
        nc_variables = np.exp(nc_variables)

    # save absolute values
    fileio.write_netcdf(filename.format(prefix='', type='ensmean'),
                        variables={variable_name: np.mean(nc_variables, axis=0)},
                        **kwargs)

    nc_prcs = np.percentile(nc_variables, list(prcs.values()), axis=0)
    for type, value in zip(prcs, nc_prcs):
        fileio.write_netcdf(filename.format(prefix='', type=type), variables={variable_name: value}, **kwargs)
```

    --- pr ---------------------------------------------------------------------------
    Write: results\2023-05-30T18.58\anom_pr.ensmean.nc
    Write: results\2023-05-30T18.58\anom_pr.ensmedian.nc
    Write: results\2023-05-30T18.58\anom_pr.enspctl10.nc
    Write: results\2023-05-30T18.58\anom_pr.enspctl90.nc
    Write: results\2023-05-30T18.58\pr.ensmean.nc
    Write: results\2023-05-30T18.58\pr.ensmedian.nc
    Write: results\2023-05-30T18.58\pr.enspctl10.nc
    Write: results\2023-05-30T18.58\pr.enspctl90.nc
    --- tos --------------------------------------------------------------------------
    Write: results\2023-05-30T18.58\anom_tos.ensmean.nc
    Write: results\2023-05-30T18.58\anom_tos.ensmedian.nc
    Write: results\2023-05-30T18.58\anom_tos.enspctl10.nc
    Write: results\2023-05-30T18.58\anom_tos.enspctl90.nc
    Write: results\2023-05-30T18.58\tos.ensmean.nc
    Write: results\2023-05-30T18.58\tos.ensmedian.nc
    Write: results\2023-05-30T18.58\tos.enspctl10.nc
    Write: results\2023-05-30T18.58\tos.enspctl90.nc
    
