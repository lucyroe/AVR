"""
Script to perform a Hidden Markov Model (HMM) analysis on ECG and EEG Data.

Inputs: Preprocessed ECG and EEG data.

Outputs: TODO: define

Functions:


Steps:
1. 

Required packages: mne, neurokit

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: May 22th, 2024
"""

# %% Import
import os
from pathlib import Path

import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
physiopath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/NeVRo/preprocessed/physiological/"

debug = True

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. GET DATA
if __name__ == "__main__":
    # get the file path for each file of the dataset
    file_list = os.listdir(physiopath)

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
        subject_list = [subject_list[1]]

    # Loop over all subjects
    for subject_index, subject in enumerate(subject_list):
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

        # Read in data
        subject_data = pd.read_csv(physiopath + "sub_" + subject + "_mov_Space_ECG_preprocessed.tsv", sep="\t")

        # Get HR
        hr = subject_data["HR"].dropna().values.reshape(-1, 1)

        # Plot HR
        plt.figure()
        plt.plot(hr)

        # %% STEP 2. HMM

        # Create and train the Hidden Markov Model
        print("\nCreating and training the Hidden Markov Model...")
        num_states = 2  # Number of states (low and high HR) = hyperparameter that needs to be chosen
        model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000)
        model.fit(hr)

        # Predict the hidden states of HMM
        hidden_states = model.predict(hr)

        print("\nMeans and variances of hidden states:")
        for i in range(model.n_components):
            print("Hidden state", i+1)
            print("Mean = ", round(model.means_[i][0], 3))
            print("Variance = ", np.diag(model.covars_[i][0], 3))

        # Plot the hidden states
        plt.figure()
        plt.plot(hidden_states)
        plt.title("Hidden States")

        # Plot the heart rate with the hidden states marked in color vertically
        plt.figure()
        plt.plot(hr)
        plt.title("Heart Rate")
        for i in range(num_states):
            plt.fill_between(np.arange(len(hr)), 0, 200, where=(hidden_states == i), color="C" + str(i), alpha=0.3)



# %%
