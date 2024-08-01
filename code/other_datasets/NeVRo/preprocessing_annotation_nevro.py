"""
Script to preprocess annotation data for NeVRo dataset.

Inputs: Raw annotation data

Outputs: Preprocessed annotation data as tsv file (both for subjects separrately and averaged over all subjects)

Steps:
1. LOAD DATA

2. PREPROCESS DATA
    2a. Formatting data (column names, etc.)
    2b. Rescale arousal ratings from 0 to 100 range to -1 to 1 range
    2c. Round arousal ratings to 2 decimal places
    2d. Save preprocessed data

3. AVERAGE OVER ALL PARTICIPANTS
    3a. Concatenate data of all participants into one dataframe
    3b. Save dataframe with all participants (all_..._arousal_preprocessed.tsv)
    3c. Calculate mean arousal data
    3d. Save dataframe with mean arousal data (avg_..._arousal_preprocessed.tsv)

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: July 6th, 2024
"""

# %% Import
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Volumes/LUCY_MEMORY/NeVRo/Data/ratings/continuous/not_z_scored/"
preprocessed_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/NeVRo/preprocessed/annotations/"

# create the preprocessed folder if it does not exist
Path(preprocessed_path).mkdir(parents=True, exist_ok=True)

# define conditions to analyze (mov = movement, nomov = no movement)
conditions = ["mov"]
# "nomov"

# sections to analyze
sections = ["space"]
# "break", "anden"  # noqa: ERA001

# length of these sections in seconds
# the sections space and anden were already trimmed by 2.5s in the beginning and in the end = 5s in total
# that is why the lengths defined here are 5s shorter than the lengths of the physiological datastreams
section_lengths = {"space": 148, "break": 30, "anden": 92}

# sampling rates for annotation data in Hz
sampling_rate = 1

# only analyze one subject when debug mode is on
debug = False

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. LOAD DATA
if __name__ == "__main__":
    for condition in conditions:
        for section in sections:
            # get right file path for the dataset
            filepath = Path(datapath) / condition / section

            # get the file path for each file of the dataset
            file_list = os.listdir(filepath)

            # deletes all hidden files from the list
            file_list = [file for file in file_list if not file.startswith(".")]

            # delete all files that are not .txt files
            file_list = [file for file in file_list if file.endswith(".txt")]

            # sort the list in ascending order
            file_list = sorted(file_list)

            # get participant list from the file list
            subject_list = [file.split("_")[1][1:3] for file in file_list]

            # only analyze one subject if debug is True
            if debug:
                subject_list = [subject_list[0]]

            # Create empty list for data of all participants
            list_data_all = []

            # Loop over all subjects
            for subject_index, subject in enumerate(subject_list):
                print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

                # Get the right file for the subject
                pattern = f"NVR_S{subject}_run_?_{section}_rat_z.txt"

                # Find the file in file_list that matches the pattern
                subject_file = glob.fnmatch.filter(file_list, pattern)[0]

                # Read in data
                subject_data = pd.read_csv(Path(filepath) / subject_file, sep=",", header=None)

                # %% STEP 2. PREPROCESS DATA
                # Formatting data
                # Delete third column
                subject_data = subject_data.drop(columns=2)
                # Rename columns
                subject_data.columns = ["time", "cr_a"]
                # Add column with subject ID as the first column
                subject_data.insert(0, "sj_id", subject)

                # Rescale arousal ratings from 0 to 100 range to -1 to 1 range
                subject_data["cr_a"] = (subject_data["cr_a"] - 50) / 50

                # Round arousal ratings to 2 decimal places
                subject_data["cr_a"] = subject_data["cr_a"].round(2)

                # Save preprocessed data
                subject_data.to_csv(
                    preprocessed_path + f"sub_{subject}_{condition}_{section}_arousal_preprocessed.tsv",
                    sep="\t",
                    index=False,
                )
                # Data is saved as a tsv file with the following columns (in that order):
                # "sj_id": subject ID
                # "time": time (seconds, s), sampling rate = 1 Hz, starts at 1
                # "cr_a": arousal rating (-1-1)

                # Add participant's data to dataframe for all participants
                list_data_all.append(subject_data)

            # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS

            # Concatenate data of all participants into one dataframe
            data_all = pd.concat(list_data_all, ignore_index=True)

            # Save dataframe with all participants in tsv file
            data_all.to_csv(
                preprocessed_path + f"all_{condition}_{section}_arousal_preprocessed.tsv", sep="\t", index=False
            )

            # Drop column with subject number
            data_all = data_all.drop(columns=["sj_id"])
            # Calculate mean arousal data
            data_mean = data_all.groupby("time").mean()
            # Add time column as first column
            data_mean.insert(0, "time", data_mean.index)

            # Save dataframe with mean arousal data in tsv file
            data_mean.to_csv(
                preprocessed_path + f"avg_{condition}_{section}_arousal_preprocessed.tsv", sep="\t", index=False
            )

            # Plot mean arousal data
            plt.figure()
            sns.lineplot(x="time", y="cr_a", data=data_mean)
            plt.title(f"Mean arousal ratings during {section} in {condition} condition")
            plt.xlabel("Time (s)")
            plt.ylabel("Arousal rating")
            plt.ylim(-1, 1)
            plt.show()
# %%
