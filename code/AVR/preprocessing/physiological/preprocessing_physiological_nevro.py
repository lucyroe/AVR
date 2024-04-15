"""
Script to preprocess physiological data for a given dataset

Inputs: Raw data

Outputs: Preprocessed data (EEG, ECG) in csv files

Functions: TODO: define

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: April 14th, 2024
"""

# %% Import
import os
from pathlib import Path
import mne

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import neurokit2 as nk

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Volumes/LUCY_MEMORY/NeVRo/Data/EEG/01_raw/"
preprocessed_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/NeVRo/preprocessed/physiological/"

# create the preprocessed folder if it does not exist
Path(preprocessed_path).mkdir(parents=True, exist_ok=True)

# define modalities to analyze
modalities = ["EEG", "ECG"]
# define conditions to analyze (mov = movement, nomov = no movement)
conditions = ["mov"]
# "nomov"

# markers of the conditions
# S 30 : Movement Start
# S 35 : Movement End
# S130 : No Movement Start
# S135 : No Movement End
markers_conditions = {"mov": ["S 30", "S 35"], "nomov": ["S130", "S135"]}

# sections to analyze
sections = ["Space"]
# "Break", "Anden"

# markers of the sections
# S 30 : Space Movement Start
# S 31 : Space Movement End
# S 32 : Break Movement Start
# S 33 : Break Movement End
# S 34 : Anden Movement Start
# S 35 : Anden Movement End
# S130 : Space No Movement Start
# S131 : Space No Movement End
# S132 : Break No Movement Start
# S133 : Break No Movement End
# S134 : Anden No Movement Start
# S135 : Anden No Movement End
markers_sections = {
    "mov": {"Space": ["S 30", "S 31"], "Break": ["S 32", "S 33"], "Anden": ["S 34", "S 35"]},
    "nomov": {"Space": ["S130", "S131"], "Break": ["S132", "S133"], "Anden": ["S134", "S135"]},
}

# length of these sections in seconds
section_lengths = {"Space": 153, "Break": 30, "Anden": 97}

# sampling rates for each modality in Hz
sampling_rates = {"EEG": 500, "ECG": 500}

# define how much to trim from the beginning and end of the data in seconds
# (to remove artifacts from the beginning and end of the rollercoaster ride)
trim_seconds = 2.5

# only analyze one subject when debug mode is on
debug = True


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def crop_data(raw_data, markers, sampling_rate) -> mne.io.Raw:
    """
    Crops the raw data to the given markers
    """
    # Get events from annotations
    events = mne.events_from_annotations(raw_data)

    # Get index of the markers in the events
    markers_indeces = [events[1][marker] for marker in markers]

    # Get corresponding times for the markers and transform them to seconds
    start_time = events[0][markers_indeces[0]][0] / sampling_rate
    end_time = events[0][markers_indeces[1]][0] / sampling_rate

    # Get the data for the current condition
    cropped_data = raw_data.copy().crop(tmin=start_time, tmax=end_time)

    return cropped_data


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% Get Files and Format Data
if __name__ == "__main__":
    # get the file path for each file of the dataset
    file_list = os.listdir(datapath + f"full_SETs")

    # deletes all hidden files from the list
    file_list = [file for file in file_list if not file.startswith(".")]

    # delete all files that are not .set files or .fdt files
    file_list = [file for file in file_list if file.endswith(".set") or file.endswith(".fdt")]

    # sort the list in ascending order
    file_list = sorted(file_list)

    # get participant list from the file list
    # we have two files per subject (.set and .fdt)
    # therefore, we only need to extract the subject number from the first file (skip every second file in the list)
    subject_list = [file.split("_")[1][1:3] for file in file_list[::2]]

    # only analyze one subject if debug is True
    if debug:
        subject_list = [subject_list[0]]

    # Loop over all subjects
    for subject in subject_list:
        print(f"Processing subject {subject} of " + str(len(subject_list)) + "...")

        # Get .set file for the current subject
        subject_file = (
            datapath + f"full_SETs/" + [file for file in file_list if subject in file and file.endswith(".set")][0]
        )

        # Read in data
        raw_data = mne.io.read_raw_eeglab(subject_file, preload=True)
        # print(raw_data.info)

        # Loop over conditions
        for condition_index, condition in enumerate(conditions):
            print(
                f"Processing condition "
                + condition
                + " (condition "
                + str(condition_index + 1)
                + " out of "
                + str(len(conditions))
                + "..."
            )

            # Get markers for the current condition
            markers_condition = markers_conditions[condition]

            # Crop data to the current condition
            data_condition = crop_data(raw_data, markers_condition, sampling_rates["EEG"])

            # Loop over sections
            for section in sections:
                # Get markers for the current section
                markers_section = markers_sections[condition][section]

                # Crop data to the current section
                data_section = crop_data(raw_data, markers_section, sampling_rates["EEG"])

                # Check if the section differentiates from the defined length
                if round(data_section.times[-1]) - round(data_section.times[0]) != section_lengths[section]:
                    print(
                        f"Section "
                        + section
                        + " is too short or too long. Something wrent wrong. Please check the code."
                    )
                else:
                    print(f"Section " + section + " is the correct length.")

                # Trim trim_seconds from the beginning and end of the data
                data_section_trimmed = data_section.copy().crop(
                    tmin=trim_seconds, tmax=round(data_section.times[-1]) - trim_seconds
                )

                # Separate EEG and ECG data
                if "ECG" in modalities:
                    # select only the ECG channel
                    data_section_ecg = data_section_trimmed.copy().pick(["ECG"])
                if "EEG" in modalities:
                    # exclude GSR, ECG, and EOG channels
                    data_section_eeg = data_section_trimmed.copy().pick(
                        [
                            "Fp1",
                            "Fz",
                            "F3",
                            "F7",
                            "FC5",
                            "FC1",
                            "C3",
                            "T7",
                            "TP9",
                            "CP5",
                            "CP1",
                            "Pz",
                            "P3",
                            "P7",
                            "O1",
                            "Oz",
                            "O2",
                            "P4",
                            "P8",
                            "TP10",
                            "CP6",
                            "CP2",
                            "Cz",
                            "C4",
                            "T8",
                            "FC6",
                            "FC2",
                            "F4",
                            "F8",
                            "Fp2",
                        ]
                    )

            # %% Preprocess Data


# %%
