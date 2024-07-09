"""
Script to preprocess physiological data (EEG, ECG, PPG) for AVR phase 3.

Inputs: Raw EEG data in .edf files, ECG and PPG data in tsv.gz files

Outputs: Preprocessed data (EEG, ECG, PPG) in tsv files

Functions:
    crop_data(raw_data, markers, sampling_rate) -> mne.io.Raw:
                        Crops the raw data to the given markers.
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
Created on: July 8th, 2024
Last update: July 6th, 2024
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

subjects = ["pilot003"]  # Adjust as needed
# "pilot001", "pilot002"
subject_task_mapping = {subject: "AVRnomov" if subject == "pilot001" else "AVR" for subject in subjects}
# pilot subject 001 and 002 were the same person but once without movement and once with movement

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

# Only analyze one subject when debug mode is on
debug = True
if debug:
    subjects = [subjects[0]]

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def crop_data(raw_data, markers, sampling_rate) -> mne.io.Raw:
    """Crops the raw data to the given markers."""
    # Get events from annotations
    events = mne.events_from_annotations(raw_data)

    # Get index of the markers in the events
    markers_indeces = [events[1][marker] for marker in markers]

    # Get corresponding times for the markers and transform them to seconds
    start_time = events[0][markers_indeces[0]][0] / sampling_rate
    end_time = events[0][markers_indeces[1]][0] / sampling_rate

    # Get the data for the current condition
    return raw_data.copy().crop(tmin=start_time, tmax=end_time)


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
        info_eeg_file = open(subject_data_path / f"sub-{subject}_task-{subject_task_mapping[subject]}_eeg.json")
        info_eeg = json.load(info_eeg_file)
        info_channels = pd.read_csv(subject_data_path / f"sub-{subject}_task-{subject_task_mapping[subject]}_channels.tsv", sep="\t")
        info_physio_file = open(subject_data_path / f"sub-{subject}_task-{subject_task_mapping[subject]}_physio.json")
        info_physio = json.load(info_physio_file)

        # Get the EOG channels
        eog_channels = info_channels[info_channels["type"] != "EEG"]["name"].to_list()

        # Read in EEG data
        raw_eeg_data = mne.io.read_raw_edf(
            subject_data_path / f"sub-{subject}_task-{subject_task_mapping[subject]}_eeg.edf",
            eog=eog_channels,
            preload=True
        )

        # Unzip and read in other physiological data (ECG, PPG)
        file = subject_data_path / f"sub-{subject}_task-{subject_task_mapping[subject]}_physio.tsv.gz"
        with gzip.open(file, "rt") as f:
            raw_physio_data = pd.read_csv(f, sep="\t")
        # Separate ECG and PPG data
        raw_ecg_data = raw_physio_data["timestamp"] + raw_physio_data["cardiac"]
        raw_ppg_data = raw_physio_data["timestamp"] + raw_physio_data["ppg"]

        # Get the sampling rates of the data from info json files
        sampling_rates = {"eeg": info_eeg["SamplingFrequency"],
                            "ecg": info_physio["SamplingFrequency"],
                            "ppg": info_physio["SamplingFrequency"]}

        # Load event markers for subject
        event_markers = pd.read_csv(
            data_dir / exp_name / rawdata_name / f"sub-{subject}" / "beh" /
            f"sub-{subject}_task-{subject_task_mapping[subject]}_events.tsv", sep="\t"
        )

        # Load mapping for event markers to real events
        event_mapping = pd.read_csv(data_dir / exp_name / rawdata_name / "events_mapping.tsv", sep="\t")
        # TODO: update event_mapping.tsv with correct mapping when markers are finalized  # noqa: FIX002

        # %% STEP 2. PREPROCESS DATA
        # 2a. Cutting data
        # Get start and end time of the experiment
        start_marker = event_mapping[event_mapping["event_name"] == "start_experiment"]["trial_type"].iloc[0]
        start_time = event_markers[event_markers["trial_type"] == start_marker]["onset"].iloc[1]
        end_marker = event_mapping[event_mapping["event_name"] == "end_experiment"]["trial_type"].iloc[0]
        end_time = event_markers[event_markers["trial_type"] == end_marker]["onset"].iloc[0]

        # Cut data to start and end time
        # And remove first and last 2.5 seconds of data (if specified above)

            # Crop data to the current condition
            data_condition = crop_data(raw_data, markers_condition, sampling_rates["EEG"])

            # Loop over sections
            for section in sections:
                # Get markers for the current section
                markers_section = markers_sections[condition][section]

                # Crop data to the current section
                data_section = crop_data(raw_data, markers_section, sampling_rates["EEG"])

                # TODO: problem: dataframes have different lengths for each participant  # noqa: FIX002
                # but should be 74.000 samples = 148s
                # exclude participants with different lengths for now
                # but sth is wrong with the markers (different markers for different participants?)
                # so cropping data for the current section results in different lengths

                # Check if the section differentiates from the defined length
                if round(data_section.times[-1]) - round(data_section.times[0]) != section_lengths[section]:
                    print(
                        "Section "
                        + section
                        + " of Participant "
                        + subject
                        + " is too short or too long. Participant will be excluded."
                    )
                    continue
                else:
                    print("Section " + section + " of Participant " + subject + " is the correct length.")

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

                # %% STEP 2. PREPROCESS DATA

                # Downsample data if downsample is True
                if downsample:
                    data_section_eeg.resample(sfreq=downsample_rate)
                    data_section_ecg.resample(sfreq=downsample_rate)

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
