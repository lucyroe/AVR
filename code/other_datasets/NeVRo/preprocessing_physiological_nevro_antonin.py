"""
Script to cut segments from preprocessed physiological data for NeVRo 2.0.

Inputs: Preprocessed ECG and EEG data

Outputs: Cut and segmented ECG and EEG data in tsv files

Steps:
1. GET FILES
2. CUT DATA
3. SAVE DATA
4. AVERAGE OVER ALL PARTICIPANTS

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Last update: May 28th, 2024
"""

# %% Import
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Volumes/LUCY_MEMORY/NeVRo/NeVRo_preproc_data/"
preprocessed_path = (
    "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/NeVRo/preprocessed/physiological/Antonin/"
)

# create the preprocessed folder if it does not exist
Path(preprocessed_path).mkdir(parents=True, exist_ok=True)

# define modalities to analyze
modalities = ["IBI", "HRV", "ROI_EEGpow"]
# define which variables to analyze for each modality
# IBI: inter beat interval
# HRV: heart rate variability -> low frequency (LF_mirrored), high frequency (HF_mirrored), or all (HRV_all_mirrored)
# ROI_EEGpow: power in the region of interest (ROI) which was the posterior lobe (channels Pz, P3, P4, P7, P8, O1, O2,
# and Oz) -> alpha_power_roi_mirrored, or other frequencies
variables = {"IBI": ["rs_ibi_mirrored"], "HRV": ["HF_mirrored"], "ROI_EEGpow": ["alpha_power_roi_mirrored"]}

# define conditions to analyze (mov = movement, nomov = no movement)
conditions = ["mov"]
# "nomov"

# sections to analyze
section = "Space"
# "Break", "Anden"

# length of the sections in seconds
section_lengths = {"Space": 148, "Break": 30, "Anden": 92}

# define how many seconds to cut at the beginning and end of the time series (because data was mirrored)
cut_time = 35

# only analyze one subject when debug mode is on
debug = False

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. GET FILES
if __name__ == "__main__":
    # loop over all modalities
    for index_modality, modality in enumerate(modalities):
        print(f"Processing {modality} data (modality {index_modality+1} of {len(modalities)})...")

        # loop over all conditions
        for condition in conditions:
            # get the file path for each file of the dataset
            file_list = os.listdir(datapath + modality + "/" + condition + "/SBA/")

            # deletes all hidden files from the list
            file_list = [file for file in file_list if not file.startswith(".")]

            # sort the list in ascending order
            file_list = sorted(file_list)

            # get participant list from the file list
            if modality == "ROI_EEGpow":  # EEG data
                subject_list = [file.split("_")[0][1:3] for file in file_list]
            else:  # ECG data
                subject_list = [file.split("_")[0][2:4] for file in file_list]

            # only analyze one subject if debug is True
            if debug:
                subject_list = [subject_list[0]]

            # Loop over all subjects
            for subject_index, subject in enumerate(subject_list):
                print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

                # Get subject file
                subject_file = (
                    datapath
                    + modality
                    + "/"
                    + condition
                    + "/SBA/"
                    + next(file for file in file_list if subject in file)
                )

                # Read in data
                if modality in ["IBI", "HRV"]:  # ECG data saved as .txt file
                    data_file = pd.read_csv(subject_file, delimiter="\t")
                else:  # EEG data saved as .mat file
                    data_file = loadmat(subject_file)

                # Loop over all variables
                for variable in variables[modality]:
                    print(f"Processing variable {variable}...")

                    # Get data
                    data = data_file[variable] if modality in ["IBI", "HRV"] else data_file[variable][0]

                    # %% STEP 2. CUT DATA
                    # Cut data at the beginning and end
                    all_data = data[cut_time:-cut_time]

                    # Cut section
                    section_data = all_data[: section_lengths["Space"]]

                    # Create dataframe
                    # Add subject id and time column as first and second column
                    if variable == "rs_ibi_mirrored":
                        variable_name = "IBI"
                    elif variable == "HF_mirrored":
                        variable_name = "HF_HRV"
                    else:
                        variable_name = "posterior_alpha_power"

                    data_df = pd.DataFrame(
                        {"sj_id": subject, "time": np.arange(len(section_data)), variable_name: section_data}
                    )

                    # delete index
                    data_df = data_df.reset_index(drop=True)

                    # Plot data for manual check
                    plt.figure()
                    plt.plot(np.arange(len(section_data)), section_data)
                    plt.title(f"Subject {subject}, {condition}, {section}, {variable_name}")
                    plt.ylabel(f"{variable_name}")
                    plt.xlabel("Time (s)")
                    plt.show()

                    # %% STEP 3. SAVE DATA
                    # Save data to tsv file
                    data_df.to_csv(
                        preprocessed_path + f"sub_{subject}_{condition}_{section}_{variable_name}_preprocessed.tsv",
                        sep="\t",
                    )

    # %% STEP 4. AVERAGE OVER ALL PARTICIPANTS
    # Loop over all modalities
    for modality in modalities:
        # Loop over all conditions
        for condition in conditions:
            for variable in variables[modality]:
                # List all files in the path
                file_list = os.listdir(preprocessed_path)

                if variable == "rs_ibi_mirrored":
                    variable_name = "IBI"
                elif variable == "HF_mirrored":
                    variable_name = "HF_HRV"
                else:
                    variable_name = "posterior_alpha_power"

                # Delete all files from other modalities and conditions
                file_list = [file for file in file_list if f"{condition}_{section}_{variable_name}" in file]

                # Average over all participants
                data_all = []

                # Loop over all subjects
                for file in file_list:
                    # Read in data
                    data = pd.read_csv(preprocessed_path + file, delimiter="\t")
                    # Add data to list
                    data_all.append(data)

                # Concatenate all dataframes
                all_data_df = pd.concat(data_all)

                # Average over all participants (grouped by the index = timepoint)
                data_avg = all_data_df.groupby(level=0).mean()[variable_name]

                # Add time column as first column
                data_avg = pd.DataFrame({"time": data_avg.index, variable_name: data_avg})

                # delete index
                data_avg = data_avg.reset_index(drop=True)

                # Plot average data for manual check
                plt.figure()
                plt.plot(data_avg[variable_name])
                plt.title(f"Averaged data, {condition}, {section}, {variable_name}")
                plt.ylabel(f"{variable_name}")
                plt.xlabel("Time (s)")
                plt.show()

                # Save average data to tsv file
                data_avg.to_csv(
                    preprocessed_path + f"avg_{condition}_{section}_{variable_name}_preprocessed.tsv", sep="\t"
                )
# %%
