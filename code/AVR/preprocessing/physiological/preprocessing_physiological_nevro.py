"""
Script to preprocess physiological data for a given dataset.

Inputs: Raw EEG data in .set and .fdt files

Outputs: Preprocessed data (EEG, ECG) in tsv files

Functions:
    crop_data(raw_data, markers, sampling_rate) -> mne.io.Raw:
                        Crops the raw data to the given markers.
    plot_ecgpeaks(ecg_clean, rpeaks_info, min_time, max_time, plot_title, ecg_sampling_rate):
                        Plot ECG signal with R-peaks

Steps:
1. GET FILES AND FORMAT DATA
    1a. Read in .set files
    1b. Get event markers (for conditions (mov/nomov) and sections (Space/Break/Anden)) from annotations
    1c. Crop data to the given markers for a certain consition and section
    1d. Trim data to remove artifacts from the beginning and end of the rollercoaster ride
    1e. Separate EEG and ECG data
2. PREPROCESS DATA
    2a. Downsample data (optional)
    2b. ECG
        2b.1. Flip ECG signal
        2b.2. Data Cleaning using NeuroKit
        2b.3. R-peaks detection using NeuroKit
        2b.4. IBI Calculation
        2b.5. Calculate heart rate (HR) from R-peaks
        2b.6. Save data to .tsv file
    2c. EEG
        2c.1. PREP Pipeline (MATLAB) #TODO
3. AVERAGE OVER ALL PARTICIPANTS
    3a. Average over all participants' ECG & save to .tsv file
    3b. Average over all participants' EEG & save to .tsv file

Required packages: mne, neurokit

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: April 25th, 2024
"""

# %% Import
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns

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

# define whether data should be downsampled (in Hz)
downsample = False
downsample_rate = 250

# only analyze one subject when debug mode is on
debug = False


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


def plot_ecgpeaks(ecg_clean, rpeaks_info, min_time, max_time, plot_title, ecg_sampling_rate):  # noqa: D417, PLR0913
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

# %% STEP 1. GET FILES AND FORMAT DATA
if __name__ == "__main__":
    # get the file path for each file of the dataset
    file_list = os.listdir(datapath + "full_SETs")

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
    for subject_index, subject in enumerate(subject_list):
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

        # Get .set file for the current subject
        subject_file = (
            datapath + "full_SETs/" + next(file for file in file_list if subject in file and file.endswith(".set"))
        )

        # Read in data
        raw_data = mne.io.read_raw_eeglab(subject_file, preload=True)
        # print(raw_data.info)  # noqa: ERA001

        # Loop over conditions
        for condition_index, condition in enumerate(conditions):
            print(
                "Processing condition "
                + condition
                + " (condition "
                + str(condition_index + 1)
                + " out of "
                + str(len(conditions))
                + ")..."
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

                # TODO: problem: dataframes have different lengths for each participant
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
                    print("Section " + section + " of Participant "
                    + subject + " is the correct length.")

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

                # TODO: manually check R-peaks and adjust if necessary

                # IBI Calculation
                # Calculate inter-beat-intervals (IBI) from R-peaks
                r_peaks_indices = info["ECG_R_Peaks"]
                ibi = nk.signal_period(peaks=r_peaks_indices, sampling_rate=ecg_sampling_rate)

                # Calculate heart rate (HR) from R-peaks
                heart_rate = nk.ecg_rate(peaks=r_peaks_indices, sampling_rate=ecg_sampling_rate)

                # TODO: exclude participants with 40 < HR < 90 ? (as resting state)
                # TODO: relate HR to resting HR ?

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
                # ------------------------ ECG ------------------------
                # PREP Pipeline (MATLAB) #TODO  # noqa: FIX002, TD004
    
    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS
    # TODO: this does not make sense atm
    # all participants have different 
    # Loop over conditions
    for condition in conditions:
        # Loop over sections
        for section in sections:
            # List all files in the path
            file_list = os.listdir(preprocessed_path)

            # Average over all participants' ECG & save to .tsv file
            # Get files corresponding to the current condition and section
            file_list_section = [
                file
                for file in file_list
                if f"{condition}_{section}_ECG_preprocessed.tsv" in file
            ]
            data_all = []

            # Loop over all subjects
            for file in file_list_section:
                # Read in ECG data
                ecg_data = pd.read_csv(
                    preprocessed_path + file, delimiter="\t"
                )
                # Add data to list
                data_all.append(ecg_data)
            
            # Concatenate all dataframes
            all_data_df = pd.concat(data_all)
            
            # Average over all participants (grouped by the index = timepoint)
            data_avg = all_data_df.groupby(level=0).mean()["ECG"]

            # R-peaks detection using NeuroKit
            r_peaks, info = nk.ecg_peaks(data_avg, sampling_rate=ecg_data["sampling_rate"][0])

            # Plot cleaned ECG data and R-peaks for the first 10s
            plot_ecgpeaks(ecg_clean=data_avg, rpeaks_info=info, min_time=0, max_time=10,
                plot_title="Cleaned ECG signal with R-peaks", ecg_sampling_rate=ecg_data["sampling_rate"][0])

            # TODO: manually check R-peaks and adjust if necessary

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
            ecg_data_df.to_csv(
                preprocessed_path + f"avg_{condition}_{section}_ECG_preprocessed.tsv", sep="\t"
            )
            # Average over all participants' EEG & save to .tsv file
            # TODO

# %%
