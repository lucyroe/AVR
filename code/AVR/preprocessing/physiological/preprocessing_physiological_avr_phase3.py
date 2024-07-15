"""
Script to preprocess physiological data (EEG, ECG, PPG) for AVR phase 3.

Inputs: Raw EEG data in .edf files, ECG and PPG data in tsv.gz files

Outputs: Preprocessed data (EEG, ECG, PPG) in tsv files

Functions:
    plot_peaks(cleaned_signal, rpeaks_info, min_time, max_time, plot_title, sampling_rate):
                        Plot ECG signal with R-peaks

Steps:
1. LOAD DATA
    1a. Load EEG data
    1b. Load ECG and PPG data
    1c. Load event markers
    1d. Load event mapping
2. PREPROCESS DATA
    2a. Cutting data
    2b. Format data
    2c. Preprocess ECG and PPG data
    2d. Preprocess EEG data
3. AVERAGE OVER ALL PARTICIPANTS
    3a.
Required packages: mne, neurokit, systole

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 6 July 2024
Last update: 15 July 2024
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
from systole.interact import Editor
from IPython.display import display
from bokeh.plotting import figure, show, output_notebook

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["001", "002", "003"]  # Adjust as needed
task = "AVR"

# Only analyze one subject when debug mode is on
debug = True
if debug:
    subjects = [subjects[0]]

# Define if plots for sanity checks should be shown
show_plots = True

# Define whether manual cleaning of R-peaks should be done
manual_cleaning = False

# Define whether scaling of the data should be done
scaling = True
scale_factor = 0.01

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

# Get rid of the sometimes excessive logging of MNE
mne.set_log_level('error')

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def plot_peaks(cleaned_signal, peaks, min_time, max_time, plot_title, sampling_rate):
    """
    Plot ECG or PPG signal with peaks.

    Arguments:
    ---------
    - cleaned_signal = dict or ndarray
    - peaks = peaks
    - min_time = starting time of the to-be-plotted interval
    - max_time = final time of the to-be-plotted interval
    - plot_title = general title of the plot (optional)
    - sampling_rate = sampling rate of the signal

    """
    # Transform min_time and max_time to samples
    min_sample = int(min_time * sampling_rate)
    max_sample = int(max_time * sampling_rate)
    # Create a time vector (samples)
    time = np.arange(min_sample, max_sample)
    fig, axs = plt.subplots(figsize=(10, 5))

    # Select signal and peaks interval to plot
    selected_signal = cleaned_signal[min_sample:max_sample]
    selected_peaks = peaks[(peaks < max_sample) & (peaks >= min_sample)]

    # Transform data from mV to V
    selected_signal = selected_signal / 1000

    axs.plot(time, selected_signal, linewidth=1, label="Signal")
    axs.scatter(selected_peaks, selected_signal[selected_peaks - min_sample], color="gray", edgecolor="k", alpha=0.6)
    axs.set_ylabel("ECG" if "ECG" in plot_title else "PPG")
    axs.set_xlabel("Time (s)")
    # transform x-axis to seconds
    axs.set_xticklabels([f"{x/sampling_rate}" for x in axs.get_xticks()])
    if plot_title:
        axs.set_title(plot_title)

    sns.despine()
    plt.show()

def preprocess_eeg(raw_data: Raw,
    low_frequency: float,
    high_frequency: int,
    resample_rate: float,
    sampling_frequency: int,
    detrend: bool,
    ransac: bool,
    autoreject: bool):
    """
    Preprocess EEG data using the MNE toolbox.

    As implemented in MNE, add channel locations according to the 10/10 system,
    inspect data for bad channels using RANSAC from the autoreject package. To
    ensure the same amount of channels for all subjects, we interpolate bad channels.

    Args:
    ----
    low_frequency: float
        Low cut-off frequency in Hz.
    high_frequency: int
        High cut-off frequency in Hz.
    resample_rate: float
        New sampling frequency in Hz.
    sampling_frequency: int
        Sampling frequency of the raw data in Hz.
    detrend: bool
        If True, the data is detrended (linear detrending)
    ransac: bool
        If True, RANSAC is used to detect bad channels.
    autoreject: bool
        If True, autoreject is used to detect bad epochs.

    Returns:
    -------
    preprocessed_eeg: TODO: specify
    channels_interpolated: list
    rejected_epochs: TODO: specify
        List with the number of interpolated channels per participant.

    """
    # Set EEG channel layout for topo plots
    montage_filename = datadir / exp_name / rawdata_name / "CACS-64_REF.bvef"
    montage = mne.channels.read_custom_montage(montage_filename)

    # Store how many channels were interpolated per participant
    channels_interpolated = []

    # Setting montage from Brainvision montage file
    raw_data.set_montage(montage)

    # Bandpass filter the data
    filtered_data = raw_data.filter(l_frew=low_frequency, h_freq=high_frequency, method="iir")

    # Resample the data
    resampled_data = filtered_data.resample(resample_rate)

    # Pick only EEG channels for Ransac bad channel detection
    picks = mne.pick_types(resampled_data.info, eeg=True, eog=False)

    # Use RANSAC to detect bad channels
    # (autoreject interpolates bad channels, takes quite long and removes a lot of epochs due to blink artifacts)
    if ransac:
        ransac = Ransac(verbose="progressbar", picks=picks, n_jobs=3)

        preprocessed_data = ransac.fit_transform(resampled_data)
        print("\n".join(ransac.bad_chs_))
        channels_interpolated.append(ransac.bad_chs_)

        # Detect bad epochs
        # Now feed the clean channels into Autoreject to detect bad trials
        if autoreject:
            ar = AutoReject()
            cleaned_data, rejected_epochs = ar.fit_transform(ransac_data)

    print("Done with preprocessing and creating clean epochs.")

    return preprocessed_data, channels_interpolated, rejected_epochs

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
        # ---------------------- 2a. Cutting data ----------------------
        # Get start and end time of the experiment
        start_time = events[events["event_name"] == "start_spaceship"]["onset"].iloc[0]
        end_time = events[events["event_name"] == "end_spaceship"]["onset"].iloc[-1]

        # Get events for experiment (from start to end of experiment)
        events_experiment = events[(events["onset"] >= start_time) & (events["onset"] <= end_time)]
        # Delete unnecessary column trial_type
        events_experiment = events_experiment.drop(columns=["trial_type"])

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

        # ---------------------- 2b. Format data ----------------------
        # Set time to start at 0
        # EEG data already starts at 0
        cropped_ecg_data["timestamp"] = cropped_ecg_data["timestamp"] - cropped_ecg_data["timestamp"].iloc[0]
        cropped_ppg_data["timestamp"] = cropped_ppg_data["timestamp"] - cropped_ppg_data["timestamp"].iloc[0]

        # Adjust event time so first marker starts not at 0 but at - cut_off_seconds
        events_experiment["onset"] = events_experiment["onset"] - start_time - cut_off_seconds

        # Round onset column to 3 decimal places (1 ms accuracy)
        # To account for small differences in onset times between participants
        cropped_ecg_data.times = cropped_eeg_data.times.round(2)
        cropped_ecg_data["timestamp"] = cropped_ecg_data["timestamp"].round(3)
        cropped_ppg_data["timestamp"] = cropped_ppg_data["timestamp"].round(3)
        events_experiment["onset"] = events_experiment["onset"].round(3)

        # Reset index
        events_experiment = events_experiment.reset_index(drop=True)
        cropped_ecg_data = cropped_ecg_data.reset_index(drop=True)
        cropped_ppg_data = cropped_ppg_data.reset_index(drop=True)

        # Scale ECG and PPG data
        if scaling:
            cropped_ecg_data["cardiac"] = cropped_ecg_data["cardiac"] * scale_factor
            cropped_ppg_data["ppg"] = cropped_ppg_data["ppg"] * scale_factor

        # ---------------------- 2c. Preprocess ECG and PPG data ----------------------
        # Flip ECG signal (as it is inverted)
        ecg_data_flipped = nk.ecg_invert(cropped_ecg_data["cardiac"], sampling_rate=sampling_rates["ecg"], force=True)[0]

        # Data Cleaning using NeuroKit for ECG data
        # A 50 Hz powerline filter and
        # 4th-order Butterworth filters (0.5 Hz high-pass, 30 Hz low-pass)
        # are applied to the ECG signal.
        cleaned_ecg = nk.signal_filter(
            ecg_data_flipped,
            sampling_rate=sampling_rates["ecg"],
            lowcut=0.5,
            highcut=30,
            method="butterworth",
            order=4,
            powerline=50,
            show=False,
            )
        # R-peaks detection using NeuroKit for ECG data
        r_peaks_ecg, info_ecg = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_rates["ecg"])

        # Data Cleaning using NeuroKit for PPG data
        # Uses the preprocessing pipeline "elgendi" and "templatematch" to asses quality of method
        # R-peaks detection using NeuroKit for PPG data
        signals_ppg, info_ppg = nk.ppg_process(cropped_ppg_data["ppg"], sampling_rate=sampling_rates["ppg"], method="elgendi",
            method_quality="templatematch")

        # Plot cleaned ECG data and R-peaks for the first 10s
        if show_plots:
            plot_peaks(cleaned_signal=cleaned_ecg, peaks=info_ecg["ECG_R_Peaks"], min_time=0, max_time=10,
                plot_title="Cleaned ECG signal with R-peaks", sampling_rate=sampling_rates["ecg"])

        # Plot PPG data and PPG-peaks for the first 10s
        if show_plots:
            plot_peaks(cleaned_signal=signals_ppg["PPG_Clean"], peaks=info_ppg["PPG_Peaks"], min_time=0, max_time=10,
                plot_title="Cleaned PPG signal with PPG-peaks", sampling_rate=sampling_rates["ppg"])

        # Perform manual cleaning of peaks if specified
        if manual_cleaning:
            # Manual correction of R-peaks
            # Save JSON file with corrected R-peaks and bad segments indices
            ecg_corr_fname = f'sub-{subject}_task-{exp_name}_rpeaks-corrected.json'
            ecg_corr_fpath = Path(subject_preprocessed_folder) / ecg_corr_fname

            # Transform array of R-peaks marked as 1s in a list of 0s to a boolean array
            r_peaks_ecg_boolean = r_peaks_ecg["ECG_R_Peaks"].astype(bool)

            # Display interactive plot
            # TODO: make this better by scaling it to 10 seconds for each window and then clicking through them
            # Also, how do I actually correct anything?!
            %matplotlib ipympl

            editor_ecg = Editor(signal=cleaned_ecg,
                        corrected_json=ecg_corr_fpath,
                        sfreq=sampling_rates["ecg"], corrected_peaks=r_peaks_ecg_boolean,
                        signal_type="ECG", figsize=(10, 6))

            display(editor_ecg.commands_box)

            # Manual correction of PPG-peaks
            # Save JSON file with corrected PPG-peaks and bad segments indices
            ppg_corr_fname = f'sub-{subject}_task-{exp_name}_ppg-peaks-corrected.json'
            ppg_corr_fpath = Path(subject_preprocessed_folder) / ppg_corr_fname

            # Transform array of PPG-peaks marked as 1s in a list of 0s to a boolean array
            ppg_peaks_boolean = signals_ppg["PPG_Peaks"].astype(bool)

            editor_ppg = Editor(signal=signals_ppg["PPG_Clean"],
                        corrected_json=ppg_corr_fpath,
                        sfreq=sampling_rates["ppg"], corrected_peaks=ppg_peaks_boolean,
                        signal_type="PPG", figsize=(10, 6))

            display(editor_ppg.commands_box)

        # Execute only when manual peak correction is done
        if manual_cleaning:
            editor_ecg.save()
            editor_ppg.save()
        
        # Load corrected R-peaks and PPG-peaks if manual cleaning was done
        if manual_cleaning:
            # Load corrected R-peaks
            with open(ecg_corr_fpath, "r") as f:
                corrected_rpeaks = json.load(f)
            # Load corrected PPG-peaks
            with open(ppg_corr_fpath, "r") as f:
                corrected_ppg_peaks = json.load(f)

        # Calculate inter-beat-intervals (IBI) from peaks
        r_peaks_indices = corrected_rpeaks["ecg"]["corrected_peaks"] if manual_cleaning else info_ecg["ECG_R_Peaks"]
        ibi_ecg = nk.signal_period(peaks=r_peaks_indices, sampling_rate=sampling_rates["ecg"])

        ppg_peaks_indices = corrected_ppg_peaks["ppg"]["corrected_peaks"] if manual_cleaning else info_ppg["PPG_Peaks"]
        ibi_ppg = nk.signal_period(peaks=ppg_peaks_indices, sampling_rate=sampling_rates["ppg"])

        # Calculate heart rate (HR) from peaks
        heart_rate_ecg = nk.ecg_rate(peaks=r_peaks_indices, sampling_rate=sampling_rates["ecg"])
        heart_rate_ppg = nk.ppg_rate(peaks=ppg_peaks_indices, sampling_rate=sampling_rates["ppg"])

        # Plot IBI and HR for ECG and PPG data
        if show_plots:
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            axs[0, 0].plot(ibi_ecg)
            axs[0, 0].set_title("IBI from ECG")
            axs[0, 1].plot(heart_rate_ecg)
            axs[0, 1].set_title("HR from ECG")
            axs[1, 0].plot(ibi_ppg)
            axs[1, 0].set_title("IBI from PPG")
            axs[1, 1].plot(heart_rate_ppg)
            axs[1, 1].set_title("HR from PPG")
            plt.show()
        
        # Create dataframe with cleaned ECG data, R-peaks, IBI, and HR
        ecg_data_df = pd.DataFrame({"ECG": cleaned_ecg})
        ecg_data_df["R-peaks"] = pd.Series(r_peaks_indices)
        ecg_data_df["IBI"] = pd.Series(ibi_ecg)
        ecg_data_df["HR"] = pd.Series(heart_rate_ecg)
        # Create array with subject id that has the same length as the other series
        subject_array = [subject] * len(cleaned_ecg)
        ecg_data_df["subject"] = pd.Series(subject_array)
        # Make the subject column the first column
        ecg_data_df = ecg_data_df[["subject", "ECG", "R-peaks", "IBI", "HR"]]

        # Save ECG data to tsv file
        ecg_data_df.to_csv(
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_ecg_preprocessed.tsv", sep="\t", index=False
        )

        # Create dataframe with cleaned PPG data, PPG-peaks, IBI, and HR
        ppg_data_df = pd.DataFrame({"PPG": signals_ppg["PPG_Clean"]})
        ppg_data_df["PPG-peaks"] = pd.Series(ppg_peaks_indices)
        ppg_data_df["IBI"] = pd.Series(ibi_ppg)
        ppg_data_df["HR"] = pd.Series(heart_rate_ppg)
        # Create array with subject id that has the same length as the other series
        subject_array = [subject] * len(signals_ppg["PPG_Clean"])
        ppg_data_df["subject"] = pd.Series(subject_array)
        # Make the subject column the first column
        ppg_data_df = ppg_data_df[["subject", "PPG", "PPG-peaks", "IBI", "HR"]]

        # Save PPG data to tsv file
        ppg_data_df.to_csv(
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_ppg_preprocessed.tsv", sep="\t", index=False
        )

        # ---------------------- 2d. Preprocess EEG data ----------------------
        # High-pass filter EEG data (0.5 Hz)

        # Import channel locations

        # Rereference EOG

        # Subtract pre-stimulus baseline

        # Mark bad electrodes

        # Average reference EEG channels

        # Run ICA to clean data

        # Artifact rejection


    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS

    # TODO: THIS IS FROM OLD SCRIPT, NEEDS TO BE ADAPTED TO NEW SCRIPT  # noqa: FIX002

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
            plot_peaks(
                cleaned_signal
            =data_avg,
                rpeaks_info=info,
                min_time=0,
                max_time=10,
                plot_title="Cleaned ECG signal with R-peaks",
                sampling_rate=ecg_data["sampling_rate"][0])

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

# %%
