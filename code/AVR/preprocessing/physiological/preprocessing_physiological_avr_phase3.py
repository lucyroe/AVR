"""
Script to preprocess physiological data (EEG, ECG, PPG) for AVR phase 3.

Inputs: Raw EEG data in .edf files, ECG and PPG data in tsv.gz files

Outputs: Preprocessed data (EEG, ECG, PPG) in tsv files

Functions:
    plot_ecgpeaks(ecg_clean, rpeaks_info, min_time, max_time, plot_title, ecg_sampling_rate):
                        Plot ECG signal with R-peaks

Steps:
1. LOAD DATA TODO: update steps
    1a.
2. PREPROCESS DATA
    2a.
3. AVERAGE OVER ALL PARTICIPANTS
    3a.
Required packages: mne, neurokit

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 6 July 2024
Last update: 11 July 2024
"""
# %% Import
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["001", "002", "003"]  # Adjust as needed
task = "AVR"

# Only analyze one subject when debug mode is on
debug = True
if debug:
    subjects = [subjects[0]]

# Specify the data path info (in BIDS format)
# Change with the directory of data storage
data_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/")
exp_name = "AVR"
rawdata_name = "rawdata"  # rawdata folder
derivative_name = "derivatives"  # derivates folder
preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
averaged_name = "avg"  # averaged data folder (inside preprocessed)
datatype_name = "eeg"  # data type specification
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

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def plot_ecgpeaks(ecg_clean, rpeaks_info, min_time, max_time, plot_title, ecg_sampling_rate):
    """
    Plot ECG signal with R-peaks.

    Arguments:
    ---------
    - ecg_clean = dict or ndarray
    - rpeaks_info = dict obtained from nk.ecg_peaks() function
    - min_time = starting time of the to-be-plotted interval
    - max_time = final time of the to-be-plotted interval
    - plot_title = general title of the plot (optional)
    - ecg_sampling_rate = sampling rate of the ECG signal

    """
    # Transform min_time and max_time to samples
    min_sample = int(min_time * ecg_sampling_rate)
    max_sample = int(max_time * ecg_sampling_rate)
    # Create a time vector (samples)
    time = np.arange(min_sample, max_sample)
    fig, axs = plt.subplots(figsize=(10, 5))

    # Select ECG signal and R-Peaks interval to plot
    ecg_select = ecg_clean[min_sample:max_sample]
    rpeaks = rpeaks_info["ECG_R_Peaks"]
    rpeaks_select = rpeaks[(rpeaks < max_sample) & (rpeaks >= min_sample)]

    axs.plot(time, ecg_select, linewidth=1, label="ECG signal")
    axs.scatter(rpeaks_select, ecg_select[rpeaks_select - min_sample], color="gray", edgecolor="k", alpha=0.6)
    axs.set_ylabel("ECG (mV)")
    axs.set_ylim(-0.015, 0.02)  # set y-axis range in V
    # transform y-axis ticks from V to mV
    axs.set_yticklabels([f"{round(x,3)*100}" for x in axs.get_yticks()])
    axs.set_xlabel("Time (s)")
    # transform x-axis to seconds
    axs.set_xticklabels([f"{x/ecg_sampling_rate}" for x in axs.get_xticks()])
    if plot_title:
        axs.set_title(plot_title)

    sns.despine()
    plt.show()

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. LOAD DATA
if __name__ == "__main__":
    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")

        # Define the path to the data
        subject_data_path = data_dir / exp_name / rawdata_name / f"sub-{subject}" / datatype_name

        # Get the info json files
        info_eeg_file = open(subject_data_path / f"sub-{subject}_task-{task}_eeg.json")
        info_eeg = json.load(info_eeg_file)
        info_channels = pd.read_csv(subject_data_path / f"sub-{subject}_task-{task}_channels.tsv", sep="\t")
        info_physio_file = open(subject_data_path / f"sub-{subject}_task-{task}_physio.json")
        info_physio = json.load(info_physio_file)

        # Get the EOG channels
        eog_channels = []
        for channel in info_channels.iterrows():
            if "EOG" in channel[1]["type"]:
                eog_channels.append(channel[1]["name"])

        # Read in EEG data
        raw_eeg_data = mne.io.read_raw_edf(
            subject_data_path / f"sub-{subject}_task-{task}_eeg.edf",
            eog=eog_channels,
            preload=True
        )

        # Unzip and read in other physiological data (ECG, PPG)
        file = subject_data_path / f"sub-{subject}_task-{task}_physio.tsv.gz"
        with gzip.open(file, "rt") as f:
            raw_physio_data = pd.read_csv(f, sep="\t")
        # Separate ECG and PPG data
        raw_ecg_data = pd.DataFrame(data = raw_physio_data, columns=["timestamp", "cardiac"])
        raw_ppg_data = pd.DataFrame(data = raw_physio_data, columns=["timestamp", "ppg"])

        # Get the sampling rates of the data from info json files
        sampling_rates = {"eeg": info_eeg["SamplingFrequency"],
                            "ecg": info_physio["SamplingFrequency"],
                            "ppg": info_physio["SamplingFrequency"]}

        # Load event markers for subject
        event_markers = pd.read_csv(
            data_dir / exp_name / rawdata_name / f"sub-{subject}" / "beh" /
            f"sub-{subject}_task-{task}_events.tsv", sep="\t"
        )

        # Load mapping for event markers to real events
        event_mapping = pd.read_csv(data_dir / exp_name / rawdata_name / "events_mapping.tsv", sep="\t")

        # Drop column with trial type
        event_mapping = event_mapping.drop(columns=["trial_type"])

        # Add column with event names to event markers
        events = pd.concat([event_markers, event_mapping], axis=1)

        # Drop unnecessary columns
        events = events.drop(columns=["duration"])

        # Set event time to start at delay of first event after beginning of recording
        events["onset"] = events["onset"] - raw_ecg_data["timestamp"].iloc[0]

        # Set time to start at 0
        # EEG data already starts at 0
        raw_ecg_data["timestamp"] = raw_ecg_data["timestamp"] - raw_ecg_data["timestamp"].iloc[0]
        raw_ppg_data["timestamp"] = raw_ppg_data["timestamp"] - raw_ppg_data["timestamp"].iloc[0]

        # %% STEP 2. PREPROCESS DATA
        # 2a. Cutting data
        # Get start and end time of the experiment
        start_time = events[events["event_name"] == "start_experiment"]["onset"].iloc[0]
        end_time = events[events["event_name"] == "end_experiment"]["onset"].iloc[0]

        # Get events for experiment (from start to end of experiment)
        events_experiment = events[(events["onset"] >= start_time) & (events["onset"] <= end_time)]

        # Cut data to start and end time
        # And remove first and last 2.5 seconds of data (if specified above)
        if cut_off_seconds > 0:
            cropped_eeg_data = raw_eeg_data.copy().crop(tmin=(start_time + cut_off_seconds), tmax=(end_time - cut_off_seconds))
            cropped_ecg_data = raw_ecg_data[(raw_ecg_data["timestamp"] >= (start_time + cut_off_seconds)) & (raw_ecg_data["timestamp"] <= (end_time - cut_off_seconds))]
            cropped_ppg_data = raw_ppg_data[(raw_ppg_data["timestamp"] >= (start_time + cut_off_seconds)) & (raw_ppg_data["timestamp"] <= (end_time - cut_off_seconds))]
        else:
            cropped_eeg_data = raw_eeg_data.copy().crop(tmin=(start_time), tmax=(end_time))
            cropped_ecg_data = raw_ecg_data[(raw_ecg_data["timestamp"] >= (start_time)) & (raw_ecg_data["timestamp"] <= (end_time))]
            cropped_ppg_data = raw_ppg_data[(raw_ppg_data["timestamp"] >= (start_time)) & (raw_ppg_data["timestamp"] <= (end_time))]

        # %%
        # 2b. Format data
        # Set time to start at 0
        # EEG data already starts at 0
        cropped_ecg_data["timestamp"] = cropped_ecg_data["timestamp"] - cropped_ecg_data["timestamp"].iloc[0]
        cropped_ppg_data["timestamp"] = cropped_ppg_data["timestamp"] - cropped_ppg_data["timestamp"].iloc[0]

        # Adjust event time so first marker starts at - cut_off_seconds
        
        # Round onset column to 3 decimal places (1 ms accuracy)
        # To account for small differences in onset times between participants
        cropped_ecg_data.times = cropped_eeg_data.times.round(2)
        cropped_ecg_data["timestamp"] = cropped_ecg_data["timestamp"].round(3)
        cropped_ppg_data["timestamp"] = cropped_ppg_data["timestamp"].round(3)

        # Reset index
        cropped_ecg_data = cropped_ecg_data.reset_index(drop=True)
        cropped_ppg_data = cropped_ppg_data.reset_index(drop=True)

        # %%
                # plot ECG data for manual inspection
                # data_section_ecg.plot()  # noqa: ERA001

                # plot EEG data for manual inspection
                # data_section_eeg.plot()  # noqa: ERA001

                # ------------------------ ECG ------------------------
                # get data as numpy array
                ecg_data = data_section_ecg.get_data()[0]
                # get sampling rate
                ecg_sampling_rate = data_section_ecg.info["sfreq"]

                # flip ECG signal (as it is inverted)
                ecg_data_flipped = nk.ecg_invert(ecg_data, sampling_rate=ecg_sampling_rate, force=True)[0]

                # Data Cleaning using NeuroKit
                # A 50 Hz powerline filter and
                # 4th-order Butterworth filters (0.5 Hz high-pass, 30 Hz low-pass)
                # are applied to the ECG signal.
                ecg_cleaned = nk.signal_filter(
                    ecg_data_flipped,
                    sampling_rate=ecg_sampling_rate,
                    lowcut=0.5,
                    highcut=30,
                    method="butterworth",
                    order=4,
                    powerline=50,
                    show=False,
                )

                # R-peaks detection using NeuroKit
                r_peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)

                # Plot cleaned ECG data and R-peaks for the first 10s
                # plot_ecgpeaks(ecg_clean=ecg_cleaned, rpeaks_info=info, min_time=0, max_time=10,
                # plot_title="Cleaned ECG signal with R-peaks", ecg_sampling_rate=ecg_sampling_rate)

                # TODO: manually check R-peaks and adjust if necessary  # noqa: FIX002

                # IBI Calculation
                # Calculate inter-beat-intervals (IBI) from R-peaks
                r_peaks_indices = info["ECG_R_Peaks"]
                ibi = nk.signal_period(peaks=r_peaks_indices, sampling_rate=ecg_sampling_rate)

                # Calculate heart rate (HR) from R-peaks
                heart_rate = nk.ecg_rate(peaks=r_peaks_indices, sampling_rate=ecg_sampling_rate)

                # TODO: exclude participants with 40 < HR < 90 ? (as resting state)  # noqa: FIX002
                # TODO: relate HR to resting HR ?  # noqa: FIX002

                # plot IBI
                plt.plot(ibi)

                # plot HR
                plt.plot(heart_rate)

                # create dataframe with cleaned ECG data, R-peaks, IBI, and HR
                ecg_data_df = pd.DataFrame({"ECG": ecg_cleaned})
                ecg_data_df["R-peaks"] = pd.Series(r_peaks_indices)
                ecg_data_df["IBI"] = pd.Series(ibi)
                ecg_data_df["HR"] = pd.Series(heart_rate)
                ecg_data_df["sampling_rate"] = pd.Series(ecg_sampling_rate)
                # create array with subject id that has the same length as the other series
                subject_array = [subject] * len(r_peaks_indices)
                ecg_data_df["sj_id"] = pd.Series(subject_array)

                # save ECG data to tsv file
                ecg_data_df.to_csv(
                    preprocessed_path + f"sub_{subject}_{condition}_{section}_ECG_preprocessed.tsv", sep="\t"
                )
                # ------------------------ EEG ------------------------
                # PREP Pipeline (MATLAB) #TODO  # noqa: FIX002, TD004

    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS
    # TODO: this does not make sense atm  # noqa: FIX002
    # Loop over conditions
    for condition in conditions:
        # Loop over sections
        for section in sections:
            # List all files in the path
            file_list = os.listdir(preprocessed_path)

            # Average over all participants' ECG & save to .tsv file
            # Get files corresponding to the current condition and section
            file_list_section = [file for file in file_list if f"{condition}_{section}_ECG_preprocessed.tsv" in file]
            data_all = []

            # Loop over all subjects
            for file in file_list_section:
                # Read in ECG data
                ecg_data = pd.read_csv(preprocessed_path + file, delimiter="\t")
                # Add data to list
                data_all.append(ecg_data)

            # Concatenate all dataframes
            all_data_df = pd.concat(data_all)

            # Average over all participants (grouped by the index = timepoint)
            data_avg = all_data_df.groupby(level=0).mean()["ECG"]

            # R-peaks detection using NeuroKit
            r_peaks, info = nk.ecg_peaks(data_avg, sampling_rate=ecg_data["sampling_rate"][0])

            # Plot cleaned ECG data and R-peaks for the first 10s
            plot_ecgpeaks(
                ecg_clean=data_avg,
                rpeaks_info=info,
                min_time=0,
                max_time=10,
                plot_title="Cleaned ECG signal with R-peaks",
                ecg_sampling_rate=ecg_data["sampling_rate"][0],
            )

            # TODO: manually check R-peaks and adjust if necessary  # noqa: FIX002

            # IBI Calculation
            # Calculate inter-beat-intervals (IBI) from R-peaks
            r_peaks_indices = info["ECG_R_Peaks"]
            ibi = nk.signal_period(peaks=r_peaks_indices, sampling_rate=ecg_data["sampling_rate"][0])

            # Calculate heart rate (HR) from R-peaks
            heart_rate = nk.ecg_rate(peaks=r_peaks_indices, sampling_rate=ecg_data["sampling_rate"][0])

            # Create dataframe with cleaned ECG data, R-peaks, IBI, and HR
            ecg_data_df = pd.DataFrame({"ECG": data_avg})
            ecg_data_df["R-peaks"] = pd.Series(r_peaks_indices)
            ecg_data_df["IBI"] = pd.Series(ibi)
            ecg_data_df["HR"] = pd.Series(heart_rate)
            ecg_data_df["sampling_rate"] = pd.Series(ecg_data["sampling_rate"][0])

            # Save ECG data to tsv file
            ecg_data_df.to_csv(preprocessed_path + f"avg_{condition}_{section}_ECG_preprocessed.tsv", sep="\t")
            # Average over all participants' EEG & save to .tsv file
            # TODO  # noqa: FIX002, TD004

# %%
