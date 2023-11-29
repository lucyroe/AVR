# AffectiveVR – **code**

    Last update:    November 29, 2023
    Status:         work in progress

***

## Description

*List relevant information one needs to know about the code of this research project.
For instance, one could describe the computational model that was applied,
and which statistical approach has been chosen for.*

## Preprocessing

*General information regarding preprocessing could be written in the data [README.md](../data/README.md).
One could add here more implementation-specific information (e.g., which toolboxes were used).*

## Codebase

*Refer to the corresponding code/scripts written for the analysis/simulation/etc.
Which languages (Python, R, Matlab, ...) were used? Are there specific package versions,
which one should take care of? Or is there a container (e.g., Docker) or virtual environment?*

### Python

Python code (in the structure of a python package) is stored in `./code/AVR/`

To install the research code as package, run the following code in the project root directory:

```shell
pip install -e ".[develop]"
```

#### Jupyter Notebooks

Jupyter notebooks are stored in `./code/notebooks/`

### R

*Initialize a new R-project in the project root of `AVR` with `RStudio`.
R-scripts can be stored in `./code/Rscripts/`.
Use R-packages in Python with, e.g., [rpy2](https://rpy2.github.io/), or use Python packages in R using,
e.g., [reticulate](https://rstudio.github.io/reticulate/)*.

### Configs

Paths to data, parameter settings, etc. are stored in the config file: `./code/configs/config.toml`

Private config files that contain, e.g., passwords, and therefore should not be shared,
or mirrored to a remote repository can be listed in: `./code/configs/private_config.toml`

Both files will be read out by the script in `./code/AVR/configs.py`.
Keep both config toml files and the script in the places, where they are.

To use your configs in your python scripts, do the following:

```python
from AVR.configs import config, paths

# check out which paths are set in config.toml
paths.show()

# get the path to data
path_to_data = paths.DATA

# Get parameter from config
weight_decay = config.params.weight_decay

# Get private parameter from config
api_key = config.service_x.api_key
```

*Fill the corresponding `*config.toml` files with your data.*

For other programming languages, corresponding scripts must be implemented to use these `*config.toml` files in a similar way.

## COPYRIGHT/LICENSE

*One could add information about code sharing
(e.g., is the code published on GitLab or GitHub ...),
license and copy right issues*
