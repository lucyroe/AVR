"""
Script to read in and calculate descriptive statistics of the participants of AVR phase 3.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 9 August 2024
Last update: 10 August 2024
"""
#%%
def descriptive_statistics(subjects=["001", "002", "003"],  # noqa: PLR0915, B006, C901, PLR0912, PLR0913
            data_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
            results_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
            debug=False):
    """
    Calculate descriptive statistics.

    Inputs:     Pre-survey questionnaire data
                Preprocessed annotation data
                Preprocessed physiological data (EEG, ECG)

    Outputs:    TODO: define

    Steps:
    1. LOAD & FORMAT DATA
        1a. Load & Format Questionnaire Data
        1b. Load & Format Annotation Data
        1c. Load & Format Event Markers
        1d. Load & Format Physiological Data
    2. CALCULATE DESCRIPTIVE STATISTICS
        2a. Demographics
        2b. Annotation Data
        2c. Physiological Data

    """
    # %% Import
    import gzip
    import json
    import sys
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    task = "AVR"

    # Only analyze one subject when debug mode is on
    if debug:
        subjects = [subjects[0]]

    # Specify the data path info (in BIDS format)
    # Change with the directory of data storage
    data_dir = Path(data_dir) / "phase3"
    exp_name = "AVR"
    derivative_name = "derivatives"  # derivates folder
    preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
    features_name = "features"  # features folder (inside preprocessed
    averaged_name = "avg"  # averaged data folder (inside preprocessed)
    results_dir = Path(results_dir) / "phase3"

    # List of different videos
    videos = ["spaceship", "invasion", "asteroids", "underwood"]

    # List of physiological features to be analyzed
    features = ["posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"]

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    pass

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. LOAD & FORMAT DATA

    # 1a. Load Questionnaire Data
    questionnaire_filepath = data_dir / exp_name / "data_avr_phase3_pre_survey_2024-08-08_13-49.csv"
    data_questionnaire = pd.read_csv(questionnaire_filepath, sep="\t", header=1, encoding='utf-16')

    participant_ids = data_questionnaire["ID Number (free text)"]

    # Get age and gender
    age = data_questionnaire["Age.1"]
    gender = data_questionnaire["Gender"]   # 1: female; 2: male; 3: non-binary

    # Add all variablees to a dataframe
    demographics = pd.DataFrame()
    demographics["subject"] = participant_ids
    demographics["age"] = age
    demographics["gender"] = gender

    # Save dataframe as tsv file
    demographics.to_csv(data_dir / exp_name / "demographics.tsv", sep="\t", index=False)

    # 1b. Load Annotation Data
    # Initialize dataframe to store annotation data
    annotation_data = pd.DataFrame()

    annotation_filepath = data_dir / exp_name / derivative_name / preprocessed_name / averaged_name / "beh" / "all_subjects_task-AVR_beh_preprocessed.tsv"
    annotation_data_file = pd.read_csv(annotation_filepath, sep="\t")

    subjects_list = annotation_data_file["subject"]
    timestamps = annotation_data_file["timestamp"]

    # Get valence and arousal ratings
    valence = annotation_data_file["valence"]
    arousal = annotation_data_file["arousal"]

    # Add all variables to a dataframe
    annotation_data["subject"] = subjects_list
    annotation_data["timestamp"] = timestamps
    annotation_data["valence"] = valence
    annotation_data["arousal"] = arousal

    # 1c. Load Event Markers
    event_filepath = data_dir / exp_name / derivative_name / preprocessed_name / "events_experiment.tsv"
    event_data = pd.read_csv(event_filepath, sep="\t")

    timestamps_events = event_data["onset"]

    # Get event markers
    events = event_data["event_name"]

    # Assign the corresponding video name to each period of the data
    list_subjects = []
    for subject in annotation_data["subject"].unique():
        subject_data = annotation_data[annotation_data["subject"] == subject]
        for video in videos:
            counter = 0
            for row in subject_data.iterrows():
                # Get the timestamps of the start and end of the video
                timestamp_start_video = timestamps_events[events == f"start_{video}"].reset_index()["onset"][counter]
                timestamp_stop_video = timestamps_events[events == f"end_{video}"].reset_index()["onset"][counter]
                if row[1]["timestamp"] >= timestamp_start_video and row[1]["timestamp"] <= timestamp_stop_video:
                    subject_data.loc[row[0], "video"] = video
                if video == "spaceship" and row[1]["timestamp"] >= timestamp_stop_video:
                    counter += 1
        list_subjects.append(subject_data)

    # Concatenate all subjects
    annotation_data = pd.concat(list_subjects)

    # Get any rows with nan values in the "video" variable (periods of fade-ins/fade-outs in between the videos)
    nan_rows_annotation = annotation_data[annotation_data.isna().any(axis=1)]
    # Count the number of nan rows
    print(f"Number of nan rows in annotation data (fade-ins/fade-outs of the videos): {len(nan_rows_annotation)}")
    # These rows will later be ignored in calculating the descriptive statistics

    annotation_features_dir = data_dir / exp_name / derivative_name / features_name / averaged_name / "beh"
    # Create directory if it does not exist
    annotation_features_dir.mkdir(parents=True, exist_ok=True)
    # Save annotation data to features folder
    annotation_data.to_csv(annotation_features_dir / "all_subjects_task-AVR_beh_features.tsv", sep="\t", index=False)

    # 1d. Load Physiological Data
    # Initialize dataframes to store physiological data
    list_ecg_data = []
    list_eeg_data = []
    # Loop over all subjects
    for subject in subjects:
        subject_datapath = data_dir / exp_name / derivative_name / features_name / f"sub-{subject}" / "eeg"
        ecg_data_subject = pd.read_csv(subject_datapath / f"sub-{subject}_task-AVR_ecg_features.tsv", sep="\t")
        # Add subject ID to the dataframe as first column
        ecg_data_subject.insert(0, "subject", subject)

        # Initialize dataframe to store feature data
        list_features_subject = pd.DataFrame()
        # Loop over physiological features
        for feature in features:
            if len(feature.split("_")) > 1:
                region = feature.split("_")[0]
                band = feature.split("_")[1]
                # Get the data
                filename = f"sub-{subject}_task-AVR_eeg_features_{region}_power.tsv"
                data = pd.read_csv(subject_datapath / filename, sep="\t")
                feature_data = data[band]
                list_features_subject[feature] = feature_data
            else:
                # Get the data
                filename = f"sub-{subject}_task-AVR_eeg_features_whole-brain_power.tsv"
                data = pd.read_csv(subject_datapath / filename, sep="\t")
                feature_data = data[feature]
                list_features_subject[feature] = feature_data

        # Add timestamp and subject ID to the dataframe
        list_features_subject.insert(0, "subject", subject)
        list_features_subject.insert(1, "timestamp", data["timestamp"])

        # Compare the lengths of the timestamp columns of both dataframes
        if len(ecg_data_subject["timestamp"]) != len(list_features_subject["timestamp"]):
            print(f"The length of the timestamp columns of the ECG and EEG dataframes of subject {subject} do not match.")
            print("The longer dataframe will be truncated to the length of the shorter dataframe.")
            # Truncate the longer dataframe to the length of the shorter dataframe
            min_length = min(len(ecg_data_subject["timestamp"]), len(list_features_subject["timestamp"]))
            ecg_data_subject = ecg_data_subject[:min_length]
            list_features_subject = list_features_subject[:min_length]

        list_ecg_data.append(ecg_data_subject)
        list_eeg_data.append(list_features_subject)

    # Concatenate all subjects
    ecg_data = pd.concat(list_ecg_data)
    eeg_data = pd.concat(list_eeg_data)

    # Concatenate both dataframes horizontally
    physiological_data = pd.concat([ecg_data, eeg_data], axis=1)
    # Remove duplicate columns
    physiological_data = physiological_data.loc[:, ~physiological_data.columns.duplicated()]

    # Assign the corresponding video name to each period of the data
    list_subjects = []
    for subject in physiological_data["subject"].unique():
        subject_data = physiological_data[physiological_data["subject"] == subject]
        for video in videos:
            counter = 0
            for row in subject_data.iterrows():
                # Get the timestamps of the start and end of the video
                timestamp_start_video = timestamps_events[events == f"start_{video}"].reset_index()["onset"][counter]
                timestamp_stop_video = timestamps_events[events == f"end_{video}"].reset_index()["onset"][counter]
                if row[1]["timestamp"] >= timestamp_start_video and row[1]["timestamp"] <= timestamp_stop_video:
                    subject_data.loc[row[0], "video"] = video
                if video == "spaceship" and row[1]["timestamp"] >= timestamp_stop_video:
                    counter += 1
        list_subjects.append(subject_data)

    # Concatenate all subjects
    physiological_data = pd.concat(list_subjects)

    # Get any rows with nan values in the "video" variable (periods of fade-ins/fade-outs in between the videos)
    nan_rows_physiological = physiological_data[physiological_data.isna().any(axis=1)]
    # Count the number of nan rows
    print(f"Number of nan rows in physiological data (fade-ins/fade-outs of the videos): {len(nan_rows_physiological)}")
    # These rows will later be ignored in calculating the descriptive statistics

    physiological_features_dir = data_dir / exp_name / derivative_name / features_name / averaged_name / "eeg"
    # Create directory if it does not exist
    physiological_features_dir.mkdir(parents=True, exist_ok=True)
    # Save physiological data to features folder
    physiological_data.to_csv(physiological_features_dir / "all_subjects_task-AVR_physio_features.tsv", sep="\t", index=False)

    # %% STEP 2. CALCULATE DESCRIPTIVE STATISTICS

    # 2a. Demographics


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    descriptive_statistics()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
