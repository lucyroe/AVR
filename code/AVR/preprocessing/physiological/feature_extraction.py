"""
Script to extract features from physiological data (ECG, EEG) for AVR phase 3 to be later used in statistical analysis.

Inputs:     Preprocessed data (EEG, ECG) in tsv files (ECG) and fif files (EEG, after ICA)

Outputs:    Features extracted from the data in tsv files

Functions:  TODO: define

Steps:      TODO: define

Required packages:  mne, neurokit2, scipy, fooof

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 23 July 2024
Last update: 25 July 2024
"""

# %% Import
import gzip
import json
import sys
import time
from pathlib import Path

import fooof
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
import scipy

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
subjects = ["001", "002", "003"]  # Adjust as needed
task = "AVR"

# Only analyze one subject when debug mode is on
debug = True
if debug:
    subjects = [subjects[0]]

# Define if plots should be shown
show_plots = True

# Specify the data path info (in BIDS format)
# Change with the directory of data storage
data_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/")
exp_name = "AVR"
derivative_name = "derivatives"  # derivates folder
preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
averaged_name = "avg"  # averaged data folder (inside preprocessed)
feature_name = "features"  # feature extraction folder (inside derivatives)
datatype_name = "eeg"  # data type specification
results_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/phase3/")

# Create the features data folder if it does not exist
for subject in subjects:
    subject_features_folder = (
        data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name
    )
    subject_features_folder.mkdir(parents=True, exist_ok=True)
avg_features_folder = data_dir / exp_name / derivative_name / feature_name / averaged_name / datatype_name
avg_features_folder.mkdir(parents=True, exist_ok=True)

# Define parameters for time-frequency analysis of ECG data
sampling_frequency = 1  # Hz
tfr_method = "cwt"  # Method for TFR (stft or cwt or wvd or pwvd)
mirror_length = 80*sampling_frequency   # Length of the mirror extension for symmetric padding
window_length = 2  # 2s window  # Length of the window for smoothing
overlap = 0.5  # 50% overlap    # Overlap of the windows for smoothing

# Define low and high frequency bands for HRV analysis
lf_band = [0.04, 0.15]
hf_band = [0.15, 0.4]

# Define frequency bands of interest
# Berger, 1929: delta (δ; 0.3-4 Hz), theta (θ; 4-8 Hz), alpha (α; 8-12 Hz), beta (β; 12-30 Hz) and gamma (γ; 30-45 Hz)
bands = fooof.bands.Bands({"delta" : [0.3, 4],
                           "theta" : [4, 8],
                           "alpha" : [8, 13],
                           "beta" : [13, 30],
                           "gamma": [30, 45]})

# Define channels for the ROIs of interest
rois = {"whole-brain": ["Fz", "F3", "F7", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
                        "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF7",
                        "AF3", "AFz", "F1", "F5", "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3",
                        "POz", "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2",
                        "AF8", "AF4", "Iz"],
        "left-hemisphere": ["F3", "F7", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "P3", "P7", "O1", "AF7",
                            "AF3", "F1", "F5", "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3"],
        "right-hemisphere": ["O2", "P4", "P8", "TP10", "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "PO4",
                            "PO8", "P6", "P2", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2", "AF8", "AF4"],
        "frontal": ["AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
        "left-frontal": ["AF7", "AF3", "F7", "F5", "F3", "F1"],
        "fronto-central": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"],
        "central": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
        "centro-parietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
        "parietal": ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
        "parieto-occipital": ["PO7", "PO3", "POz", "PO4", "PO8"],
        "occipital": ["O1", "Oz", "02"]}

# Create color palette for plots
colors = {"ECG": {"IBI": "#FOE442", "HR": "#D55E00", "LF-HRV": "#E69F00", "HF-HRV": "#CC79A7"},
                    # yellow, dark orange, light orange, pink
            "EEG": {"delta": "#E69F00", "theta": "#D55E00", "alpha": "#CC79A7", "beta": "#56B4E9", "gamma": "#009E73"}
                    # light orange, dark orange, pink, light blue, green
            }

# Get rid of the sometimes excessive logging of MNE
mne.set_log_level("error")

# Enable interactive plots (only works when running in interactive mode)
# %matplotlib qt


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def calculate_hrv(ibi, sampling_frequency, method, low_frequency, high_frequency, mirror_length, window_length, overlap, show_plots):
    """
    Calculate heart rate variability (HRV) features from inter-beat intervals (IBI) data.

    Computes low-frequency (LF) and high-frequency (HF) HRV power using Time-Frequency analysis.
    Using the NeuroKit2 library for Time-Frequency analysis.

    Parameters
    ----------
    ibi : array
        Inter-beat intervals (IBI) data.
    sampling_frequency : int
        Sampling frequency of the IBI data.
    method : str
        Method for Time-Frequency analysis.
    low_frequency : list[float]
        Lower frequency bounds for HRV analysis.
    high_frequency : list[float]
        Higher frequency bounds for HRV analysis.
    mirror_length : float
        Length of the mirror extension.
    window_length : float
        Length of the window for smoothing.
    overlap : float
        Overlap of the windows for smoothing.
    show_plots : bool
        Show PSD plots of the Time-Frequency analysis.

    Returns
    -------
    lf_power_mirrored : array
        Low-frequency (LF) HRV power.
    hf_power_mirrored : array
        High-frequency (HF) HRV power.
    hrv_power_mirrored : array
        Total HRV power.
    times_mirrored : array
        Time points of the HRV power.
    ibi_mirrored : array
        Inter-beat intervals (IBI) data without mirrored extension.
    """
    # Check if IBI data is sampled at 1 Hz
    if sampling_frequency != 1:
        raise ValueError("IBI data must be sampled at 1 Hz.")

    # Resample IBI data to 4 Hz for Time-Frequency analysis
    new_frequency = 4
    ibi_resampled = nk.signal_resample(ibi, desired_length=len(ibi)*new_frequency, sampling_rate=1, desired_sampling_rate=new_frequency, method="interpolation")

    # Mirror the IBI at the beginning and end to avoid edge effects (symmetric padding)
    symmetric_padding_end = ibi_resampled[-mirror_length * new_frequency:][::1]
    symmetric_padding_beginning = ibi_resampled[:mirror_length * new_frequency][::-1]
    ibi_mirrored = np.concatenate([symmetric_padding_beginning, ibi_resampled, symmetric_padding_end])

    # Perform Time-Frequency analysis
    if method == "stft":
        # Short-Time Fourier Transform (STFT) on mirrored and resampled IBI data
        frequencies, times, tfr = nk.signal_timefrequency(ibi_mirrored, sampling_rate=new_frequency, method="stft",
                    window_type="hann", mode="psd", min_frequency=low_frequency[0], max_frequency=high_frequency[1],
                    show=False)
    elif method == "cwt":
        # Continuous Wavelet Transform (CWT) on mirrored and resampled IBI data
        frequencies, times, tfr = nk.signal_timefrequency(ibi_mirrored, sampling_rate=new_frequency, method="cwt",
                    nfreqbin=50, min_frequency=low_frequency[0], max_frequency=high_frequency[1],
                    show=False)
    elif method == "wvd":
        # Wigner-Ville Distribution (WVD)
        frequencies, times, tfr = nk.signal_timefrequency(ibi_mirrored, sampling_rate=new_frequency, method="wvd",
                    show=False)
    elif method == "pwvd":
        # Pseudo Wigner-Ville Distribution (PWVD)
        frequencies, times, tfr = nk.signal_timefrequency(ibi_mirrored, sampling_rate=new_frequency, method="pwvd",
                    show=False)
    else:
        raise ValueError("Invalid method for Time-Frequency analysis.")
    
    # Plot the Time-Frequency representation
    if show_plots:
        plt.pcolormesh(times, frequencies, tfr, shading='gouraud')
        plt.colorbar()
        plt.hlines(low_frequency[1], xmin=times[0], xmax=times[-1], colors='white', linestyles='dotted', linewidth=1)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('TFR smoothed HRV (' + method + ')')
        plt.show()

    # Smooth the Time-Frequency representation
    # Average over 2s windows with 50% overlap
    tfr_smooth = np.empty((tfr.shape[0], int(tfr.shape[1] / new_frequency) - 1))
    new_window_length = window_length * new_frequency
    individual_window_length = int(new_window_length * overlap)
    for window in range(0, tfr.shape[0]):
        averaged_window = [np.mean(tfr[window, i:i + new_window_length]) for i in range(0, len(tfr[window]),
                            individual_window_length) if i + new_window_length <= len(tfr[window])]
        tfr_smooth[window] = np.asarray(averaged_window)
    times = np.arange(0, tfr_smooth.shape[1])

    # Get the LF and HF HRV indices and frequencies
    lf_index = np.logical_and(frequencies >= low_frequency[0], frequencies <= low_frequency[1])
    hf_index = np.logical_and(frequencies >= high_frequency[0], frequencies <= high_frequency[1])
    hrv_index = np.logical_and(frequencies >= low_frequency[0], frequencies <= high_frequency[1])
    lf_frequencies = frequencies[lf_index]
    hf_frequencies = frequencies[hf_index]
    hrv_frequencies = frequencies[hrv_index]
    # Get the LF and HF HRV power
    lf_psd = tfr_smooth[lf_index, :]
    hf_psd = tfr_smooth[hf_index, :]
    hrv_psd = tfr_smooth[hrv_index, :]
    # Integrate over frequency bands to get the power
    lf_power = scipy.integrate.trapezoid(lf_psd.transpose(), lf_frequencies)
    hf_power = scipy.integrate.trapezoid(hf_psd.transpose(), hf_frequencies)
    hrv_power = scipy.integrate.trapezoid(hrv_psd.transpose(), hrv_frequencies)

    # Add one NaN at the end, because only N-1 values because of smoothing
    lf_power = np.append(lf_power, np.nan)
    hf_power = np.append(hf_power, np.nan)
    hrv_power = np.append(hrv_power, np.nan)
    times = np.append(times, times[-1]+1)

    # Cut the mirrored part of the power timeseries from the data
    lf_power_mirrored = lf_power[(mirror_length-1)*sampling_frequency:(len(ibi)+mirror_length-1)*sampling_frequency]
    hf_power_mirrored = hf_power[(mirror_length-1)*sampling_frequency:(len(ibi)+mirror_length-1)*sampling_frequency]
    hrv_power_mirrored = hrv_power[(mirror_length-1)*sampling_frequency:(len(ibi)+mirror_length-1)*sampling_frequency]
    times_mirrored = times[(mirror_length-1)*sampling_frequency:(len(ibi)+mirror_length-1)*sampling_frequency]

    # Reset times to start at 0
    times_mirrored = times_mirrored - times_mirrored[0]

    # Mirror the IBI data to the same length as the power timeseries
    # Remove last dummy value
    ibi_sl = ibi[:-1]
    symmetric_padding_end = ibi_sl[-mirror_length * sampling_frequency:][::1]
    symmetric_padding_beginning = ibi_sl[:mirror_length * sampling_frequency][::-1]
    ibi_mirrored_1Hz = np.concatenate([symmetric_padding_beginning, ibi_sl, symmetric_padding_end])

    # Cut the mirrored part of the IBI timeseries from the data
    ibi_mirrored_final = ibi_mirrored_1Hz[(mirror_length-1)*sampling_frequency:(len(ibi)+mirror_length-1)*sampling_frequency]

    return lf_power_mirrored, hf_power_mirrored, hrv_power_mirrored, times_mirrored, ibi_mirrored_final


def compute_tfr():
    pass

def compute_power():
    pass



# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. LOAD DATA
if __name__ == "__main__":
    # Loop over subjects
    for subject_index, subject in enumerate(subjects):
        print("--------------------------------------------------------------------------------")
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
        print("--------------------------------------------------------------------------------")

        # Read in file with excluded participants
        with open(data_dir / exp_name / derivative_name / preprocessed_name / "excluded_participants.tsv", "r") as file:
            excluded_participants = file.read().splitlines()

        # Skip excluded participants
        if subject in excluded_participants:
            print(f"Subject {subject} was excluded from the analysis. Skipping subject...")
            continue

        # Define the path to the data
        subject_data_path = data_dir / exp_name / derivative_name / preprocessed_name / f"sub-{subject}" / datatype_name

        # Define the path to the extracted features
        subject_features_folder = (
            data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name
        )

        # Define the path to the results
        subject_results_folder = results_dir / exp_name / f"sub-{subject}" / datatype_name

        print("********** Loading data **********\n")

        # Get the metadata json file
        metadata_file = subject_data_path / f"sub-{subject}_task-{task}_physio_preprocessing_metadata.json"
        with open(metadata_file, "r") as file:
            subject_metadata = json.load(file)

        # Read the preprocessed ECG data
        ecg_file = subject_data_path / f"sub-{subject}_task-{task}_physio_ecg_preprocessed_scaled.tsv"
        ecg_data = pd.read_csv(ecg_file, sep="\t")

        # Read the preprocessed EEG data
        eeg_file = subject_data_path / f"sub-{subject}_task-{task}_eeg_preprocessed_filtered_0.1-45_after_ica.fif"
        eeg_data = mne.io.read_raw_fif(eeg_file, preload=True)

        # %% STEP 2. FEATURE EXTRACTION ECG
        print("********** Extracting features from ECG data **********\n")

        # Get IBI and heart rate and drop NaN values
        # IBI and HR are already sampled at 1 Hz from the preprocessing
        ibi = ecg_data["IBI"].dropna()
        heart_rate = ecg_data["HR"].dropna()

        # Compute HRV features
        lf_power, hf_power, hrv_power, times, ibi_mirrored = calculate_hrv(ibi, sampling_frequency, tfr_method, lf_band, hf_band, mirror_length, window_length, overlap, show_plots)

        # Plot the HRV features
        if show_plots:
            plt.plot(times, ibi_mirrored, c='orange', linewidth=1)
            plt.title('clean IBI')
            plt.xlim([times[0], times[-1]])
            plt.show()
            for i, hrv in enumerate(['lf_power', 'hf_power', 'hrv_power']):
                if hrv == 'lf_power':
                    plt.plot(times, lf_power, c='blue', linewidth=1)
                    plt.title('clean Low Frequency HRV')
                elif hrv == 'hf_power':
                    plt.plot(times, hf_power, c='green', linewidth=1)
                    plt.title('clean High Frequency HRV')
                elif hrv == 'hrv_power':
                    plt.plot(times, hrv_power, c='red', linewidth=1)
                    plt.title('clean Total HRV')
                plt.xlim([times[0], times[-1]])
                plt.ylim([0, 0.14])
                plt.ylabel('Power')
                plt.xlabel('Time (s)')
                plt.show()

        # %% STEP 3. FEATURE EXTRACTION EEG
        print("********** Extracting features from EEG data **********\n")


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
        lf_power, hf_power, hrv_power, times, ibi_mirrored = calculate_hrv(ibi, sampling_frequency, tfr_method, lf_band, hf_band, mirror_length, window_length, overlap, show_plots)
