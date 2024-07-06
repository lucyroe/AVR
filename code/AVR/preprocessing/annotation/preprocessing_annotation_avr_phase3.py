"""
Script to preprocess annotation data for AVR phase 3.

Inputs: Raw annotation data

Outputs: Preprocessed annotation data as tsv file (both for subjects separately and averaged over all subjects)

Steps:
1. LOAD DATA

2. PREPROCESS DATA
    2a. Cutting data
    2b. Format data
    2c. Save preprocessed data as tsv file

3. AVERAGE OVER ALL PARTICIPANTS
    3a. Concatenate data of all participants into one dataframe
    3b. Save dataframe with all participants in tsv file ("all_subjects_task-{task}_beh_preprocessed.tsv")
    3c. Calculate averaged data
    3d. Save dataframe with mean arousal data in tsv file ("avg_task-{task}_beh_preprocessed.tsv")
    3e. Plot mean arousal and valence data as sanity check

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: July 6th, 2024
Last update: July 6th, 2024
"""

# %% Import
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["pilot003"]  # Adjust as needed
# "pilot001", "pilot002" had different event markers, so script does not work with them
subject_task_mapping = {subject: "AVRnomov" if subject == "pilot001" else "AVR" for subject in subjects}
# pilot subject 001 and 002 were the same person but once without movement and once with movement

# Specify the data path info (in BIDS format)
# change with the directory of data storage
data_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/")
exp_name = "AVR"
rawdata_name = "rawdata"  # rawdata folder
derivative_name = "derivatives"  # derivates folder
preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
averaged_name = "avg"  # averaged data folder (inside preprocessed)
datatype_name = "beh"  # data type specification
results_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/phase3/")

# Create the preprocessed data folder if it does not exist
for subject in subjects:
    subject_preprocessed_folder = (
        data_dir / exp_name / derivative_name / preprocessed_name / f"sub-{subject}" / datatype_name
    )
    subject_preprocessed_folder.mkdir(parents=True, exist_ok=True)
avg_preprocessed_folder = data_dir / exp_name / derivative_name / preprocessed_name / averaged_name / datatype_name
avg_preprocessed_folder.mkdir(parents=True, exist_ok=True)
avg_results_folder = results_dir / exp_name / averaged_name / datatype_name
avg_results_folder.mkdir(parents=True, exist_ok=True)

# Define if the first and last 5 seconds of the data should be cut off
# To avoid any potential artifacts at the beginning and end of the experiment
cut_off_seconds = 5

# Only analyze one subject when debug mode is on
debug = True
if debug:
    subjects = [subjects[0]]

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. LOAD DATA
if __name__ == "__main__":
    # Initialize list to store data of all participants
    list_data_all = []
    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")

        # Get right file for the dataset
        directory = Path(data_dir) / exp_name / rawdata_name / f"sub-{subject}" / datatype_name
        file = directory / f"sub-{subject}_task-{subject_task_mapping[subject]}_{datatype_name}.tsv.gz"

        # Unzip and read in data
        with gzip.open(file, "rt") as f:
            data = pd.read_csv(f, sep="\t")

        # Load event markers for subject
        event_markers = pd.read_csv(
            directory / f"sub-{subject}_task-{subject_task_mapping[subject]}_events.tsv", sep="\t"
        )

        # Load mapping for event markers to real events
        event_mapping = pd.read_csv(Path(data_dir) / exp_name / rawdata_name / "events_mapping.tsv", sep="\t")
        # TODO: update event_mapping.tsv with correct mapping when markers are finalized  # noqa: FIX002

        # %% STEP 2. PREPROCESS DATA
        # 2a. Cutting data
        # Get start and end time of the experiment
        start_marker = event_mapping[event_mapping["event_name"] == "start_experiment"]["trial_type"].iloc[0]
        start_time = event_markers[event_markers["trial_type"] == start_marker]["onset"].iloc[1]
        end_marker = event_mapping[event_mapping["event_name"] == "end_experiment"]["trial_type"].iloc[0]
        end_time = event_markers[event_markers["trial_type"] == end_marker]["onset"].iloc[0]

        # Cut data to start and end time
        # And remove first and last 5 seconds of data (if specified above)
        if cut_off_seconds > 0:
            data = data[
                (data["onset"] >= start_time + cut_off_seconds) & (data["onset"] <= end_time - cut_off_seconds)
            ]
        else:
            data = data[(data["onset"] >= start_time) & (data["onset"] <= end_time)]

        # 2b. Format data
        # Set time to start at 0
        data["onset"] = data["onset"] - data["onset"].iloc[0]

        # Remove unnecessary columns
        data = data.drop(columns=["duration"])

        # Reset index
        data = data.reset_index(drop=True)

        # Add subject ID to data as first column
        data.insert(0, "subject", subject)

        # Save preprocessed data as tsv file
        data.to_csv(
            Path(data_dir)
            / exp_name
            / derivative_name
            / preprocessed_name
            / f"sub-{subject}"
            / datatype_name
            / f"sub-{subject}_task-{subject_task_mapping[subject]}_{datatype_name}_preprocessed.tsv",
            sep="\t",
            index=False,
        )

        # Add participant's data to dataframe for all participants
        list_data_all.append(data)

    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS

    print("Finished preprocessing data for all participants. Now averaging over all participants...")

    # 3a. Concatenate data of all participants into one dataframe
    data_all = pd.concat(list_data_all, ignore_index=True)

    # 3b. Save dataframe with all participants in tsv file
    data_all.to_csv(
        Path(data_dir)
        / exp_name
        / derivative_name
        / preprocessed_name
        / averaged_name
        / datatype_name
        / f"all_subjects_task-{subject_task_mapping[subject]}_{datatype_name}_preprocessed.tsv",
        sep="\t",
        index=False,
    )

    # Drop column with subject number
    data_all = data_all.drop(columns=["subject"])
    # 3c. Calculate averaged data
    data_mean = data_all.groupby("onset").mean()
    # Add time column as first column
    data_mean.insert(0, "onset", data_all["onset"].unique())
    # Reset index
    data_mean = data_mean.reset_index(drop=True)
    # TODO: check if this works when real data is there (do time points match up?)  # noqa: FIX002

    # 3d. Save dataframe with mean arousal data in tsv file
    data_mean.to_csv(
        Path(data_dir)
        / exp_name
        / derivative_name
        / preprocessed_name
        / averaged_name
        / datatype_name
        / f"avg_task-{subject_task_mapping[subject]}_{datatype_name}_preprocessed.tsv",
        sep="\t",
        index=False,
    )

    # 3e. Plot mean arousal and valence data
    plt.figure(figsize=(20, 4))
    plt.plot(data_mean["onset"], data_mean["arousal"], label="Arousal")
    plt.plot(data_mean["onset"], data_mean["valence"], label="Valence")
    plt.title(f"Mean ratings for {subject_task_mapping[subject]} (n={len(subjects)})")
    # Transform x-axis to minutes
    plt.xticks(
        ticks=range(0, int(data_mean["onset"].max()) + 1, 60),
        labels=[str(int(t / 60)) for t in range(0, int(data_mean["onset"].max()) + 1, 60)],
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Ratings")
    plt.legend(["Arousal", "Valence"])
    plt.ylim(-1.2, 1.2)
    plt.show()

    # Save plot
    plt.savefig(
        Path(results_dir)
        / exp_name
        / averaged_name
        / f"avg_task-{subject_task_mapping[subject]}_{datatype_name}_preprocessed.png"
    )
# %%
