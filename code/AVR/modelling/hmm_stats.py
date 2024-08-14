"""
Script to calculate summary stats and test for differences between the hidden states of the different HMMs.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 14 August 2024
Last update: 14 August 2024
"""
# %%
data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"
results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/"
subjects=["001", "002", "003"]
debug=True
show_plots=False
# TODO: wrap in function

"""
Calculate summary statistics for the HMMs

Inputs: Features with hidden states from the HMMs

Outputs: Summary statistics for the HMMs

Functions: TODO: define

Steps: TODO: define
"""

# %% Import
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
annotation_path = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "preproc"
resultpath = Path(results_dir) / "phase3" / "AVR"

# Which HMMs to analyze
models = ["cardiac", "neural", "integrated"]
# Which features are used for which HMM
models_features = {
    "cardiac": ["ibi", "hf-hrv"],
    "neural": ["posterior_alpha"],
    "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
}

# Number of states (four quadrants of the Affect Grid)
number_of_states = 4

# Colors of the hidden states
colors_states = ["#009E73", "#CC79A7", "#0072B2", "#D55E00"]  # green, pink, dark blue, dark orange

# Only analyze one subject if debug is True
if debug:
    subjects = [subjects[0]]

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# %% STEP 1. GET DATA
# Loop over all models
for model in models:
    print("+++++++++++++++++++++++++++++++++")
    print(f"Calculating stats for {model} model...")
    print("+++++++++++++++++++++++++++++++++\n")

    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print("---------------------------------")
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject)) + "...")
        print("---------------------------------\n")

        # Load the features with hidden states
        hmm_path = resultpath / f"sub-{subject}" / "hmm" / model
        features_string = "_".join(models_features[model])
        hidden_states_file = f"sub-{subject}_task-AVR_{model}_model_data_{features_string}.tsv"
        hidden_states_data = pd.read_csv(hmm_path / hidden_states_file, sep="\t")

        # Load the annotations
        annotation_file = f"sub-{subject}_task-AVR_beh_preprocessed.tsv"
        annotations = pd.read_csv(annotation_path / f"sub-{subject}" / "beh" / annotation_file, sep="\t")
        # Delete subject column, flubber_frequency, and flubber_amplitude column
        annotations = annotations.drop(columns=["subject", "flubber_frequency", "flubber_amplitude"])

        # Check if the timestamps have the same length
        if len(hidden_states_data) != len(annotations):
            # Cut the longer one to the length of the shorter one
            min_length = min(len(hidden_states_data), len(annotations))
            hidden_states_data = hidden_states_data[:min_length]
            annotations = annotations[:min_length]

        # Merge the hidden states with the annotations
        data = pd.merge(hidden_states_data, annotations, on="timestamp")

        # Get a list of all features
        features = [col for col in data.columns if col not in ["timestamp", "state"]]

        # %% STEP 2. CALCULATE STATS
        # Loop over all features
        summary_stats_all = pd.DataFrame()
        for feature in features:
            print(f"Calculating stats for feature {feature}...")

            summary_stats = pd.DataFrame()
            # Calculate the mean and standard deviation for each hidden state
            summary_stats["mean"] = data.groupby("state")[feature].mean()
            summary_stats["std"] = data.groupby("state")[feature].std()
            # Calculate the minimum and maximum for each hidden state
            summary_stats[f"min"] = data.groupby("state")[feature].min()
            summary_stats[f"max"] = data.groupby("state")[feature].max()
            summary_stats["feature"] = feature

            # Append the summary stats for the feature to the summary stats for all features
            summary_stats_all = pd.concat([summary_stats_all, summary_stats], axis=0)



# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
# %%
