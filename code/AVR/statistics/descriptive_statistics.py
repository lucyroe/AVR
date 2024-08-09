"""
Script to read in and calculate descriptive statistics of the participants of AVR phase 3.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 9 August 2024
Last update: 9 August 2024
"""
#%%
def descriptive_statistics(subjects=["001"],  # noqa: PLR0915, B006, C901, PLR0912, PLR0913
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
    1. LOAD DATA
        1a. Load Questionnaire Data
        1b. Load Annotation Data
        !c. Load Physiological Data
        1c. Load Event Markers
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
    averaged_name = "avg"  # averaged data folder (inside preprocessed)
    results_dir = Path(results_dir) / "phase3"

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    pass

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. LOAD DATA
    # Initialize list to store data of all participants
    data_all = {"Questionnaire": [], "Annotation": [], "ECG": [], "EEG": []}

    # 1a. Load Questionnaire Data
    questionnaire_filepath = data_dir / exp_name / "data_avr_phase3_pre_survey_2024-08-08_13-49.csv"
    data_questionnaire = pd.read_csv(questionnaire_filepath, sep=",")

    # Remove first row (questionnaire header)
    data_questionnaire = data_questionnaire[1:]

    participant_ids = data_questionnaire["ID Number (free text)"]

    # Get age and gender
    age = data_questionnaire["Age"][1]
    gender = data_questionnaire["Gender"]

    # Add all variablees to a dataframe
    demographics = pd.DataFrame()
    demographics["subject"] = participant_ids
    demographics["age"] = age
    demographics["gender"] = gender

    # Add to list
    data_all["Questionnaire"].append(demographics)

    # 1b. Load Annotation Data
    annotation_filepath = data_dir / exp_name / derivative_name / preprocessed_name / averaged_name / "beh" / "all_subjects_task-AVR_beh_preprocessed.tsv"
    annotation_data = pd.read_csv(annotation_filepath, sep="\t")

    timestamps = annotation_data["timestamp"]

    # Get valence and arousal ratings
    valence = annotation_data["valence"]
    arousal = annotation_data["arousal"]

    # Add all variables to a dataframe
    annotations = pd.DataFrame()
    annotations["timestamp"] = timestamps
    annotations["valence"] = valence
    annotations["arousal"] = arousal

    # Add to list
    data_all["Annotation"].append(annotations)

    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print("--------------------------------------------------------------------------------")
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
        print("--------------------------------------------------------------------------------")

        subject_datapath = data_dir / exp_name / derivative_name / preprocessed_name / f"sub-{subject}"


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    descriptive_statistics()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
