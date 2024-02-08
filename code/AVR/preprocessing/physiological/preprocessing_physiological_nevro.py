"""
Script to preprocess physiological data for a given dataset

Inputs: Raw data

Outputs: Preprocessed data (EEG, ECG) in csv files

Functions: TODO: define

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: February 8th, 2024
"""

# %% Import
import os
import re
import ast
import warnings
from pathlib import Path
import matplotlib
import mne

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Volumes/LUCY_MEMORY/NeVRo/Data/EEG/01_raw/"

conditions = ["mov", "nomov"]

debug = True

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o



# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    
    for condition_index, condition in enumerate(conditions):
        print(f"Processing {condition} condition (" + str(condition_index+1) + " of " + str(len(conditions)) + " conditions)...")

        # get the file path for each file of the dataset in the current condition
        file_list = os.listdir(datapath + f"{condition}_SETs")
        
        # delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
        if ".DS_Store" in file_list:
            file_list.remove(".DS_Store")

        # sort the list in ascending order
        file_list = sorted(file_list)

        # get participant list from the file list
        subject_list = [file.split("_")[1].split()[1:2] for file in file_list[::2]]

        if debug:
            subject_list = ["01"]
        
        for subject_index, subject in enumerate(subject_list):
            print(f"Processing subject {subject} of " + str(len(subject_list)) + "...")

            # load the data
            raw_data = mne.io.read_raw_eeglab(datapath + f"{condition}_SETs" + "/" + "NVR_S" + subject + condition + ".set", preload=True)

            # get raw data and times
            data, times = raw_data[:]

            print(raw_data.info)
            raw_data.compute_psd(fmax=50).plot(picks="data", exclude="bads")
            raw_data.plot(duration=5, n_channels=30)
            plt.show()