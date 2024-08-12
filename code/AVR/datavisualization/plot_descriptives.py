"""
Plotting descriptive statistics for the AVR data.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 12 August 2024
Last updated: 12 August 2024
"""


def plot_descriptives(  # noqa: C901, PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    show_plots=False,
):
    """
    Plot descriptive statistics for the AVR data.

    The following steps are performed:
        1. TODO: Add steps.
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # Path where the data is stored
    data_dir = Path(data_dir) / "phase3" / "AVR" / "derivatives"
    annotation_dir = Path(data_dir) / "preproc" / "avg" / "beh"
    physiological_dir = Path(data_dir) / "features" / "avg" / "eeg"
    events_dir = Path(data_dir) / "preproc"

    # Path where results should be saved
    results_dir_descriptives = Path(results_dir) / "phase3" / "AVR" / "avg"

    # List of datastreams
    datastreams = {"annotation": ["valence", "arousal"],
                "physiological": ["ibi", "hrv", "lf_hrv", "hf_hrv",
                "posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"]}

    # Colors
    colors = {"annotation": {"valence": "#0072B2", "arousal": "#56B4E9"},   # dark blue, light blue
                "physiological": {"ibi": "#F0E442", "hrv": "#CC79A7", "lf_hrv": "#E69F00", "hf_hrv": "#D55E00",
                # yellow, pink, light orange, dark orange
                "posterior_alpha": "#0072B2", "frontal_alpha": "#56B4E9", "frontal_theta": "#009E73",
                "gamma": "#6C6C6C", "beta": "#CC79A7"}    # dark blue, light blue, green, grey, pink
    }

    mark_significant_differences = True  # if True, significant differences will be marked in the boxplots

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    def plot_average_timeseries(data, variable, color, ax):
        pass

    def plot_boxplot(data, variable, color, ax):
        pass

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% STEP 1. LOAD DATA
    # Create one dataframe for all datastreams
    data = pd.DataFrame()
    # Load the annotation data
    annotations = pd.read_csv(annotation_dir / "avg_task-AVR_beh_preprocessed.tsv", sep="\t")
    for variable in datastreams["annotation"]:
        data[variable] = annotations[variable]

    # Load the physiological data
    physiological = pd.read_csv(physiological_dir / "avg_task-AVR_physio_features.tsv", sep="\t")   # TODO: This file does not exist yet, implement in feature_extraction.py
    for variable in datastreams["physiological"]:
        # Check if the variable has the same length as the annotations
        for annotation_variable in datastreams["annotation"]:
            if len(physiological[variable]) != len(data[annotation_variable]):
                # Interpolate the annotation data to match the length of the physiological data
                data[annotation_variable] = np.interp(np.linspace(0, len(physiological[variable]), len(data[annotation_variable])), np.arange(len(data[annotation_variable])), data[annotation_variable])
        data[variable] = physiological[variable]


    # Get events
    events = pd.read_csv(events_dir / "events_experiment.tsv", sep="\t")

    # Create a list of the different phases that has the same length as the data
    phase = []
    for i in range(len(data)):
        for j in range(len(events)):
            if i >= events["onset"][j] and i <= events["offset"][j]:
                phase.append(events["event_name"][j].split("_")[1])
                break
    data["video"] = phase


    # %% STEP 2. PLOT AVERAGE TIMESERIES
    pass

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    plot_descriptives()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
