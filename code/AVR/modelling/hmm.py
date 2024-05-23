"""
Script to perform a Hidden Markov Model (HMM) analysis on Annotation, ECG and EEG Data.

Inputs: Preprocessed Annotation, ECG and EEG data.

Outputs: TODO: define

Functions:


Steps:
1. 

Required packages: 

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: May 23rd, 2024
"""

# %% Import
import os
from pathlib import Path
import glob

import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import seaborn as sns

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/NeVRo/preprocessed/"

modalities = ["physiological", "annotations"]
physiological_measures = ["ECG"]
#, "EEG"

# if debug is True, only one subject will be analyzed
debug = True

# Number of states (low and high HR) = hyperparameter that needs to be chosen
number_of_states = 2

# Number of iterations for training the HMM
iterations = 1000

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def create_model(data, num_states, iterations=1000):
    """
    Create and train a Hidden Markov Model (HMM) on the given data.
    Then predicts the hidden states.

    Args:
        data (np.array): The data to train the HMM on.
        num_states (int): The number of states of the HMM.
        iterations (int): The number of iterations to train the HMM (defaults to 1000).

    Returns:
        model (GaussianHMM): The trained HMM model.
        hidden_states (np.array): The predicted hidden states.
    """
    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=iterations)
    model.fit(data)
    hidden_states = model.predict(data)

    return model, hidden_states

def plot_hidden_states(data, hidden_states, num_states, title):
    """
    Plot the data with the hidden states marked in color vertically.

    Args:
        data (np.array): The data to plot.
        hidden_states (np.array): The hidden states of the data.
        num_states (int): The number of states.
        title (str): The title of the plot.
    
    Returns:
        figure: The plot.
    """
    figure = plt.figure()
    plt.plot(data)
    plt.title(title)
    for i in range(num_states):
        plt.fill_between(np.arange(len(data)), 0, 200, where=(hidden_states == i), color="C" + str(i), alpha=0.3)
    
    plt.xlabel("Time (s)")
    plt.ylabel(str(data))

    plt.show()

    return figure

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. GET DATA
if __name__ == "__main__":
    for modality in modalities:
        # get right file path for the dataset
        filepath = Path(datapath) / modality
        if modality == "physiological":
            for physiological_measure in physiological_measures:
                # list all files in the physiological folder corresponding to the physiological measure
                pattern_measure = f"*_{physiological_measure}_preprocessed.tsv"
                file_list = glob.fnmatch.filter(os.listdir(filepath), pattern_measure)

                # delete all files that are not subject files
                file_list = [file for file in file_list if file.startswith("sub")]

                # sort the list in ascending order
                file_list = sorted(file_list)

                # get participant list from the file list
                subject_list = [file.split("_")[1][0:3] for file in file_list]

                # only analyze one subject if debug is True
                if debug:
                    subject_list = [subject_list[0]]

                # Loop over all subjects
                for subject_index, subject in enumerate(subject_list):
                    print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

                    # Get the right file for the subject
                    pattern_subject = f"sub_{subject}_mov_Space_{physiological_measure}_preprocessed.tsv"
                    subject_file = glob.fnmatch.filter(file_list, pattern_subject)[0]

                    # Read in data
                    subject_data = pd.read_csv(filepath / subject_file, sep="\t")

                    # Get measure
                    if physiological_measure == "ECG":
                        data = subject_data["HR"].dropna().values.reshape(-1, 1)
                    else:
                        ...

                    # Plot data
                    plt.figure()
                    plt.plot(data)

        else:
            # list all files in the annotations folder
            file_list = os.listdir(filepath)

            # deletes all hidden files from the list
            file_list = [file for file in file_list if not file.startswith(".")]

            # delete all files that are not subject files
            file_list = [file for file in file_list if file.startswith("sub")]

            # sort the list in ascending order
            file_list = sorted(file_list)

            # get participant list from the file list
            subject_list = [file.split("_")[1][0:3] for file in file_list]

            # only analyze one subject if debug is True
            if debug:
                subject_list = [subject_list[0]]

            # Loop over all subjects
            for subject_index, subject in enumerate(subject_list):
                print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

                # Get the right file for the subject
                pattern_subject = f"sub_{subject}_mov_space_arousal_preprocessed.tsv"
                subject_file = glob.fnmatch.filter(file_list, pattern_subject)[0]

                # Read in data
                subject_data = pd.read_csv(filepath / subject_file, sep="\t")

                # Get the annotations
                arousal = subject_data["cr_a"].dropna().values.reshape(-1, 1)

                # Plot the annotations
                plt.figure()
                plt.plot(arousal)
                plt.ylim(-1, 1)


                # %% STEP 2. HMM

                # Create and train the Hidden Markov Model
                print("\nCreating and training the Hidden Markov Model...")

                model_arousal, hidden_states_arousal = create_model(arousal, number_of_states, iterations)

                print("\nMeans and variances of hidden states for arousal:")
                for i in range(model_arousal.n_components):
                    print("Hidden state", i+1)
                    print("Mean = ", round(model_arousal.means_[i][0], 3))
                    print("Variance = ", np.diag(model_arousal.covars_[i][0], 3))
                
                # Plot the annotations with the hidden states marked in color vertically
                plot_hidden_states(arousal, hidden_states_arousal, number_of_states, "Arousal with Hidden States")
                # TODO: this does not work atm


# %%
