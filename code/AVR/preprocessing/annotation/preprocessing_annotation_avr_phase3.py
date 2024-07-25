"""
Script to preprocess annotation data for AVR phase 3.

Inputs: Raw annotation data

Outputs: Preprocessed annotation data as tsv file (both for subjects separately and averaged over all subjects)

Steps:
1. LOAD DATA

2. PREPROCESS DATA
    2a. Cutting data
    2b. Format data
    2c. Save preprocessed data for each participant as tsv file

3. AVERAGE OVER ALL PARTICIPANTS
    3a. Concatenate data of all participants into one dataframe
    3b. Save dataframe with all participants in tsv file ("all_subjects_task-{task}_beh_preprocessed.tsv")
    3c. Calculate averaged data
    3d. Save dataframe with mean arousal data in tsv file ("avg_task-{task}_beh_preprocessed.tsv")
    3e. Plot mean arousal and valence data as sanity check (downsampled to 1 Hz)

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 6 July 2024
Last update: 9 July 2024
"""

# %% Import
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["001", "002", "003"]  # Adjust as needed
task = "AVR"

# Only analyze one subject when debug mode is on
debug = False
if debug:
    subjects = [subjects[0]]

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

# Define if the first and last 2.5 seconds of the data should be cut off
# To avoid any potential artifacts at the beginning and end of the experiment
cut_off_seconds = 2.5

sampling_frequency = 90  # Sampling frequency of the data in Hz

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
        file = directory / f"sub-{subject}_task-{task}_{datatype_name}.tsv.gz"

        # Unzip and read in data
        with gzip.open(file, "rt") as f:
            data = pd.read_csv(f, sep="\t")

        # Load event markers for subject
        event_markers = pd.read_csv(directory / f"sub-{subject}_task-{task}_events.tsv", sep="\t")

        # Load mapping for event markers to real events
        event_mapping = pd.read_csv(data_dir / exp_name / rawdata_name / "events_mapping.tsv", sep="\t")

        # Drop column with trial type
        event_markers = event_markers.drop(columns=["trial_type"])

        # Add column with event names to event markers
        events = pd.concat([event_markers, event_mapping], axis=1)

        # Drop unnecessary columns
        events = events.drop(columns=["duration"])

        # %% STEP 2. PREPROCESS DATA
        # 2a. Cutting data
        # Get start and end time of the experiment
        start_time = events[events["event_name"] == "start_spaceship"].reset_index()["onset"].tolist()[0]
        end_time = events[events["event_name"] == "end_spaceship"].reset_index()["onset"].tolist()[-1]

        # Get events for experiment (from start to end of experiment)
        events_experiment = events[(events["onset"] >= start_time) & (events["onset"] <= end_time)]

        # Cut data to start and end time
        # And remove first and last 2.5 seconds of data (if specified above)
        if cut_off_seconds > 0:
            data = data[
                (data["onset"] >= start_time + cut_off_seconds) & (data["onset"] <= end_time - cut_off_seconds)
            ]
            # Adjust event markers accordingly
            events_experiment["onset"] = events_experiment["onset"] - cut_off_seconds
        else:
            data = data[(data["onset"] >= start_time) & (data["onset"] <= end_time)]

        # 2b. Format data
        # Set time to start at 0
        data["onset"] = data["onset"] - data["onset"].iloc[0]

        # Set event time to start at - cut_off_seconds
        events_experiment["onset"] = events_experiment["onset"] - events_experiment["onset"].iloc[0] - cut_off_seconds

        # Remove unnecessary columns
        data = data.drop(columns=["duration"])

        # Round onset column to 2 decimal places (10 ms accuracy)
        # To account for small differences in onset times between participants
        data["onset"] = data["onset"].round(2)

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
            / f"sub-{subject}_task-{task}_{datatype_name}_preprocessed.tsv",
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
        / f"all_subjects_task-{task}_{datatype_name}_preprocessed.tsv",
        sep="\t",
        index=False,
    )

    # Drop column with subject number
    data_all = data_all.drop(columns=["subject"])

    # 3c. Calculate averaged data
    data_mean = data_all.groupby("onset").mean()
    # Add time column as first column
    data_mean.insert(0, "onset", data_mean.index)

    # 3d. Save dataframe with mean arousal data in tsv file
    data_mean.to_csv(
        Path(data_dir)
        / exp_name
        / derivative_name
        / preprocessed_name
        / averaged_name
        / datatype_name
        / f"avg_task-{task}_{datatype_name}_preprocessed.tsv",
        sep="\t",
        index=False,
    )

    # 3e. Plot mean arousal and valence data
    # Downsampling to 1 Hz for better visualization
    data_mean = data_mean.iloc[::sampling_frequency]

    plt.figure(figsize=(20, 10))
    plt.plot(data_mean["onset"], data_mean["arousal"], label="Arousal")
    plt.plot(data_mean["onset"], data_mean["valence"], label="Valence")
    plt.title(f"Mean ratings for {task} phase 3 (n={len(subjects)})")

    # Add vertical lines for event markers
    # Exclude first and last event markers
    # And only use every second event marker to avoid overlap
    for _, row in events_experiment.iloc[1:-1:2].iterrows():
        plt.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)
        plt.text(row["onset"] - 90, -1.52, row["event_name"], rotation=45, fontsize=10, color="gray")

    # Transform x-axis to minutes
    plt.xticks(
        ticks=range(0, int(data_mean["onset"].max()) + 1, 60),
        labels=[str(int(t / 60)) for t in range(0, int(data_mean["onset"].max()) + 1, 60)],
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Ratings")
    plt.legend(["Arousal", "Valence"])
    plt.ylim(-1.2, 1.2)

    # Save plot
    plt.savefig(Path(results_dir) / exp_name / averaged_name / datatype_name /
    f"avg_task-{task}_{datatype_name}_preprocessed.png")

    plt.show()
# %%
