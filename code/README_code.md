# AffectiveVR – **code**

    Last update:    August 28, 2024
    Status:         work in progress

***

## Description

The main analysis method used in this project is **Hidden Markov Models (HMMs)**. A Hidden Markov Model (HMM) is a data-driven generative statistical model that can be used to identify hidden states in dynamically changing systems, e.g., the brain or the heart ([Khalifa et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S1566253520304140?via%3Dihub); [Quinn et al., 2018](https://www.frontiersin.org/article/10.3389/fnins.2018.00603/full)). The hidden states are mutually exclusive, such that only one state occurs at any one point in time, and Markovian, such that one state is only dependent on the previous state. Each state has an initial probability 𝝅. As the states are hidden and cannot be observed directly in the data, observation models (or emission probabilities, 𝜽) link the data with each hidden state, representing the probability distribution from which the data is drawn while the state is active. Transitions between the states are represented in the transition probability matrix A. Given only a set of observations X, the Baum-Welch algorithm can be used to iteratively compute a set of model parameters λ = (𝝅, A, 𝜽) using both forward and Viterbi algorithms based on the expectation-maximization method. The such identified states can then be compared across different models or conditions.

HMMs were implemented in this project using the Python package `hmmlearn` ([version 0.3.2](https://hmmlearn.readthedocs.io/en/latest/index.html)).

![image](../publications/thesis/figures/phase3_hmm.png)

## Preprocessing

Two different directories exist for the preprocessing of annotation (`./code/AVR/preprocessing/annotation`) and physiological data (`./code/AVR/preprocessing/physiological`). There are scripts for preprocessing AVR data from the different phases.


# TODO: After here
The following open-source datasets are used for comparison:
* *A Dataset of Continuous Affect Annotations and Physiological Signals for Emotion Analysis* **(CASE)** by [Sharma et al. (2019)](https://www.nature.com/articles/s41597-019-0209-0), accessible using this [link](https://springernature.figshare.com/collections/A_dataset_of_continuous_affect_annotations_and_physiological_signals_for_emotion_analysis/4260668). Sharma and colleagues used 1-3 min long 2D videos to elicit diverse emotions in viewers, and asked participants to continuously rate their emotions using a joystick. They additionally measured physiological data such as ECG, respiration, BVP, EDA and skin temperature.

* *Continuous Physiological and Behavioral Emotion Annotation Dataset for 360° Videos* **(CEAP-360VR)** by [Xue et al. (2021)](https://dl.acm.org/doi/10.1145/3411764.3445487), accessible using this [link](https://www.dis.cwi.nl/ceap-360vr-dataset/). Xue and colleagues used 60s long videos in virtual reality to elicit diverse emotions in viewers, and asked participants to continuously rate their emotions using a Joy-Con. They additionally measured physiological data such as pupil dilation, EDA, BVP, HR and skin temperature.

* *EmotionCompass* **(EmoCompass)** by [McClay et al. (2023)](https://www.nature.com/articles/s41467-023-42241-2), accessible using this [link](https://osf.io/s8g5n/). McClay and colleagues used 2 min long purposefully created songs to elicit diverse emotions in listeners, and asked participants to continuously rate their emotions using a mouse controller.

**Preprocessing** of both annotation and physiological data consists of two main steps:

1. **Formatting**, including a) rescaling valence and arousal values to a range of -1 to 1 for all annotations; b) shifting the start of all time series to 0.05 s; and c) putting all data into a dataframe that is structured the same across datasets.

2. **Resampling**, consisting of a) linearly interpolating values if the sampling frequency of the (annotation or physiological) input device does not match the stimuli's frame rates as was the case for the CEAP dataset (using `np.interp` or `interp1d` ); and b) downsampling the physiological data to fit with the sampling frequency of the annotation data in order to be able to compare both (it remains an open question whether this is necessary for the CASE dataset, see ToDos below).

For the CASE and the CEAP dataset, already preprocessed physiological data was used as an input to the aforementioned preprocessing steps. For the physiological AVR data from phase 3 of the project, the data was preprocessed in the following way:

*TODO: Add infos about preprocessing physiological AVR data here after writing that script.*

## Codebase

All code is written in Python 3.11. See below on how to install the code as a python package. Change point analysis was performed using the python package :hammer_and_wrench: `ruptures` by [Truong et al. (2020)](https://linkinghub.elsevier.com/retrieve/pii/S0165168419303494). See their [documentation](https://centre-borelli.github.io/ruptures-docs/) of the package for detailed information.   
The `ruptures` package allows for the usage of different algorithms and models to define change points. For pragmatic reasons (especially the lack of a pre-defined number of change points), I decided to use the `Pelt` algorithm that linearly penalizes the segmentation of the time series. For this, a penalty parameter `pen` needs to be set. The optimal penalty parameter can be assessed by plotting different penalty values on the x-axis and the corresponding number of change points on the y-axis. In the resulting elbow plot, the optimal penalty value can be visually found at the 'knee' of the curve.

![elbow_plot](./pics/elbow_plot.png)

In this case, the penalty value would be defined as 1. Furthermore, the algorithm can use different models, defined in the `model` parameter. I decided to follow McClay et al. and use the least squared deviation `l2` model that detects the mean-shifts in a signal, but it's also possible to use the median or other models altogether. Different models, penalty and `jump` parameters will be compared in the future to see if they yield different results (see To Dos below).

A detailed description of the scripts and their contents is given here:

|               script               |                  contents                   |
| :--------------------------------: | :-----------------------------------------: |
|           `./code/AVR/cpa.py`                 | Script to perform a change point analysis (CPA) on continuous annotation or physiological data. Script includes functions to perform a CPA, to plot the results of a CPA, and to test the significance of the results for participants from a given dataset individually. If you want to perform a CPA for averaged data across participants, use the script `./code/AVR/cpa_averaged.py`|
|       `.code/AVR/cpa_averaged.py`            | Script performs a CPA for averaged data across participants. If you want to perform a CPA for participants individually, use the script `cpa.py`
|`./code/AVR/datacomparison/01_preprocessing.py`| Script to put CEAP and AVR phase 1 data in the same format so they can be compared later on.|
|`./code/AVR/datacomparison/02_summary_stats.py`| Script to calculate summary statistics for CEAP dataset.|
|`.code/AVR/datacomparison/03_plots_ceap_affectivevr.py`| Script to plot CEAP data against AVR phase 1 data in order to be able to compare both datasets. |

The directories `./code/AVR/datavisualization` and `./code/AVR/modelling` contain nothing so far.

### Python

Python code (in the structure of a python package) is stored in `./code/AVR/`

To install the research code as package, run the following code in the project root directory:

```shell
pip install -e ".[develop]"
```

### R

The directory `./code/AVR/Rscripts` so far only contains old scripts performing statistical analyses for phase 1 of AVR. As it is planned to do all statistical analyses in Python for phase 2 and 3, this directory is only used for documentation purposes. However, it is possible to use R-packages in Python with, e.g., [rpy2](https://rpy2.github.io/), or use Python packages in R using, e.g., [reticulate](https://rstudio.github.io/reticulate/).

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

## LICENSE

MIT
