# %% Import
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
task = "AVR"

# Specify the data path info (in BIDS format)
# change with the directory of data storage
data_dir = "E:/AffectiveVR/Phase_3/Data/"
results_dir = "E:/AffectiveVR/Phase_3/Results/"

# Specify the data path info (in BIDS format)
# Change with the directory of data storage
data_dir = Path(data_dir)
exp_name = "AVR"
rawdata_name = "rawdata"  # rawdata folder
derivative_name = "derivatives"  # derivates folder
preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
averaged_name = "avg"  # averaged data folder (inside preprocessed)
datatype_name = "eeg"  # data type specification
results_dir = Path(results_dir)

# Define the threshold for bad epochs
artifact_threshold = 100  # in µV
# Define the percentage of bad epochs that should be tolerated
bad_epochs_threshold = 30  # in percent

# Define if plots should be shown
show_plots = False

# subjects = [
#             "005"
#             ]
subjects = [
            "001", "002", "003","004", "005", "006", "007", "009",
            "011", "012", "014", "015", "016", "017", "018", "019",
            "020", "021", "022", "024", "025", "026", "027", "028", 
            "029", "030", "031", "032", "033", "034", "035", "036",
            "037", "038", "039", "040", "041", "042", "043", "045",
            "046", "047"
            ]

#%% Load data
excluded_participants = []

for subject_index, subject in enumerate(subjects):
    print("--------------------------------------------------------------------------------")
    print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
    print("--------------------------------------------------------------------------------")

    # Define the path to the data
    subject_data_path = data_dir / exp_name / rawdata_name / f"sub-{subject}" / datatype_name

    # Define the path to the preprocessed data
    subject_preprocessed_folder = (
        data_dir / exp_name / derivative_name / preprocessed_name / f"sub-{subject}" / datatype_name
    )

    # Define the path to the results
    subject_results_folder = results_dir / exp_name / f"sub-{subject}" / datatype_name

    print("********** Loading data **********\n")

    # Get the info json files
    info_channels_path = subject_data_path / f"sub-{subject}_task-{task}_channels.tsv"
    info_channels = pd.read_csv(info_channels_path, sep="\t")

    # Get the EOG channels
    eog_channels = []
    for channel in info_channels.iterrows():
        if "EOG" in channel[1]["type"]:
            eog_channels.append(channel[1]["name"])

    # Read in EEG data
    cleaned_eeg_data = mne.io.read_raw_fif(
        subject_preprocessed_folder / f"sub-{subject}_task-{task}_eeg_preprocessed_filtered_0.1-45_after_ica.fif", preload=True
    )

    # Final Check: Segment cleaned data into epochs again and check in how many epochs
    # there are still artifacts of more than 100 µV
    print(
        f"Checking for any remaining bad epochs (max. value above {artifact_threshold} µV) "
        "in the cleaned data..."
    )
    tstep = 10  # in seconds
    events = mne.make_fixed_length_events(cleaned_eeg_data, duration=tstep)
    epochs = mne.Epochs(cleaned_eeg_data, events, tmin=0, tmax=tstep, baseline=None, preload=True)
    # Pick only EEG channels
    epochs.pick_types(eeg=True, eog=False, ecg=False)

    # if show_plots:
    #     epochs.plot(scalings={"eeg": artifact_threshold*1e-6})
    #     input("Press the Enter key to continue: ") 

    # Check for bad epochs
    remaining_bad_epochs = []
    for i, epoch in enumerate(epochs):
        if np.abs(np.max(epoch.data)) > (artifact_threshold*1e-6):
            remaining_bad_epochs.append(i)

    # Calculate the percentage of bad epochs
    percentage_bad_epochs = len(remaining_bad_epochs) / len(epochs) * 100

    # Print the number of remaining bad epochs
    print(f"Number of remaining bad epochs: {len(remaining_bad_epochs)} ({percentage_bad_epochs:.2f}%).")

    # Check if the percentage of bad epochs is above the threshold
    if percentage_bad_epochs > bad_epochs_threshold:
        print("The percentage of bad epochs is above the threshold. Participant should be excluded.")

    if show_plots:
        cleaned_eeg_data.plot(scalings={"eeg": artifact_threshold*1e-6})

    answer = input("Do you want to exclude this participant? (Y/n): ")
    if answer == "Y":
        excluded_participants.append(subject)
        print("Participant added to the list of excluded participants.")
    else:
        print("Participant not excluded.")

# %%
# Save the list with excluded participants by adding to the existing file
# Create the file if it does not exist
if not (data_dir / exp_name / derivative_name / preprocessed_name / "excluded_participants.tsv").exists():
    with (data_dir / exp_name / derivative_name / preprocessed_name / "excluded_participants.tsv").open(
        "w"
    ) as f:
        f.write(str(excluded_participants) + "\n")
else:
    with (data_dir / exp_name / derivative_name / preprocessed_name / "excluded_participants.tsv").open(
        "a"
    ) as f:
        f.write(str(excluded_participants) + "\n")