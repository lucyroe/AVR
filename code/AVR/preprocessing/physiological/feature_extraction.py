"""
Script to extract features from physiological data (ECG, EEG) for AVR phase 3 to be later used in statistical analysis.

Inputs:     Preprocessed data (EEG, ECG) in tsv files (ECG) and fif files (EEG, after ICA)

Outputs:    Features extracted from the data in tsv files

Functions:
            calculate_hrv(): Compute heart rate variability (HRV) features from inter-beat intervals (IBI) data.

Steps:
1. LOAD DATA
2. FEATURE EXTRACTION ECG
3. FEATURE EXTRACTION EEG

Required packages:  mne, neurokit2, fcwt, scipy, fooof

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 23 July 2024
Last update: 26 July 2024
"""

# %% Import
import json
from pathlib import Path

import fcwt
import fooof
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
import time

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
subjects = ["001", "002", "003"]  # Adjust as needed
task = "AVR"

# Only analyze one subject when debug mode is on
debug = True
if debug:
    subjects = [subjects[0]]

# Define if plots should be shown
show_plots = True

# Define which steps to run
steps = ["Load Data", "Feature Extraction EEG"]
# "Feateure Extraction ECG"

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
    subject_features_folder = data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name
    subject_features_folder.mkdir(parents=True, exist_ok=True)
avg_features_folder = data_dir / exp_name / derivative_name / feature_name / averaged_name / datatype_name
avg_features_folder.mkdir(parents=True, exist_ok=True)

# Define parameters for time-frequency analysis of both ECG and EEG data
window_length = 2  # 2s window  # Length of the window for smoothing
overlap = 0.5  # 50% overlap    # Overlap of the windows for smoothing
mirror_length = 80  # Length of the mirror extension for symmetric padding

# Define parameters for time-frequency analysis of ECG data
sampling_frequency_ibi = 1  # Hz    # IBI and HR data are already sampled at 1 Hz from the preprocessing

# Define low and high frequency bands for HRV analysis
lf_band = [0.04, 0.15]
hf_band = [0.15, 0.4]

# Define parameters for time-frequency analysis of EEG data
sampling_frequency_eeg = 250  # Hz    # EEG data will be downsampled to 250 Hz
frequencies = np.arange(0.5, 125.5, 0.5)  # Resolution 0.5 Hz

# Define frequency bands of interest
# Berger, 1929: delta (δ; 0.3-4 Hz), theta (θ; 4-8 Hz),
# alpha (α; 8-12 Hz), beta (β; 12-30 Hz) and gamma (γ; 30-45 Hz)  # noqa: RUF003
bands = fooof.bands.Bands({"delta": [0.3, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]})

# Define channels for the ROIs of interest
rois = {
    "whole-brain": [
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
        "AF7",
        "AF3",
        "AFz",
        "F1",
        "F5",
        "FT7",
        "FC3",
        "C1",
        "C5",
        "TP7",
        "CP3",
        "P1",
        "P5",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "P6",
        "P2",
        "CPz",
        "CP4",
        "TP8",
        "C6",
        "C2",
        "FC4",
        "FT8",
        "F6",
        "F2",
        "AF8",
        "AF4",
        "Iz",
    ],
    "left-hemisphere": [
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "TP9",
        "CP5",
        "CP1",
        "P3",
        "P7",
        "O1",
        "AF7",
        "AF3",
        "F1",
        "F5",
        "FT7",
        "FC3",
        "C1",
        "C5",
        "TP7",
        "CP3",
        "P1",
        "P5",
        "PO7",
        "PO3",
    ],
    "right-hemisphere": [
        "O2",
        "P4",
        "P8",
        "TP10",
        "CP6",
        "CP2",
        "C4",
        "T8",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "PO4",
        "PO8",
        "P6",
        "P2",
        "CP4",
        "TP8",
        "C6",
        "C2",
        "FC4",
        "FT8",
        "F6",
        "F2",
        "AF8",
        "AF4",
    ],
    "frontal": ["AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
    "left-frontal": ["AF7", "AF3", "F7", "F5", "F3", "F1"],
    "fronto-central": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"],
    "central": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
    "centro-parietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "parietal": ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
    "parieto-occipital": ["PO7", "PO3", "POz", "PO4", "PO8"],
    "occipital": ["O1", "Oz", "02"],
}

# Create color palette for plots
colors = {
    "ECG": {"IBI": "#F0E442", "HRV": "#CC79A7", "LF-HRV": "#E69F00", "HF-HRV": "#D55E00"},
    # yellow, pink, light orange, dark orange
    "EEG": {"delta": "#E69F00", "theta": "#D55E00", "alpha": "#CC79A7", "beta": "#56B4E9", "gamma": "#009E73"},
    # light orange, dark orange, pink, light blue, green
}

# Get rid of the sometimes excessive logging of MNE
mne.set_log_level("error")

# Enable interactive plots (only works when running in interactive mode)
# %matplotlib qt


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def calculate_hrv(  # noqa: PLR0915, PLR0913
    ibi, sampling_frequency, low_frequency, high_frequency, mirror_length, window_length, overlap, show_plots
):
    """
    Calculate heart rate variability (HRV) features from inter-beat intervals (IBI) data.

    Computes low-frequency (LF) and high-frequency (HF) HRV power using Time-Frequency analysis.
    Using the fCWT package for Time-Frequency analysis (continuous wavelet transform).
    https://github.com/fastlib/fCWT

    Parameters
    ----------
    ibi : array
        Inter-beat intervals (IBI) data.
    sampling_frequency : int
        Sampling frequency of the IBI data.
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
        Show PSDs plots of the Time-Frequency analysis.

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
    fig : figure
        Figure of the Time-Frequency representation.
    """
    # Check if IBI data is sampled at 1 Hz
    if sampling_frequency != 1:
        raise ValueError("IBI data must be sampled at 1 Hz.")

    # Resample IBI data to 4 Hz for Time-Frequency analysis
    new_frequency = 4
    ibi_resampled = nk.signal_resample(
        ibi,
        desired_length=len(ibi) * new_frequency,
        sampling_rate=1,
        desired_sampling_rate=new_frequency,
        method="interpolation",
    )

    # Mirror the IBI at the beginning and end to avoid edge effects (symmetric padding)
    symmetric_padding_end = ibi_resampled[-mirror_length * new_frequency :][::1]
    symmetric_padding_beginning = ibi_resampled[: mirror_length * new_frequency][::-1]
    ibi_mirrored = np.concatenate([symmetric_padding_beginning, ibi_resampled, symmetric_padding_end])

    # Perform Time-Frequency analysis
    # Continuous Wavelet Transform (CWT) on mirrored and resampled IBI data
    # Using the fCWT package instead of Neurokit2 because of better control over the frequency bins
    # (nfreqbin not implemented in Neurokit2's CWT anymore)
    frequencies, tfr = fcwt.cwt(ibi_mirrored, new_frequency, low_frequency[0], high_frequency[1], 50)

    # Create times array with the same length as the TFR and the same sampling rate as the IBI data
    times = np.arange(0, len(ibi_mirrored) / new_frequency, 1 / new_frequency)
    # Convert complex TFR values to real values
    tfr = np.abs(tfr)
    # Convert frequencies to be increasing from low to high
    frequencies = np.flip(frequencies)
    # Also flip the TFR matrix to match the frequencies
    tfr = np.flip(tfr, axis=0)

    # Smooth the Time-Frequency representation
    # Average over 2s windows with 50% overlap
    tfr_smooth = np.empty((tfr.shape[0], int(tfr.shape[1] / new_frequency) - 1))
    new_window_length = window_length * new_frequency
    individual_window_length = int(new_window_length * overlap)
    for window in range(0, tfr.shape[0]):
        averaged_window = [
            np.mean(tfr[window, i : i + new_window_length])
            for i in range(0, len(tfr[window]), individual_window_length)
            if i + new_window_length <= len(tfr[window])
        ]
        tfr_smooth[window] = np.asarray(averaged_window)
    times = np.arange(0, tfr_smooth.shape[1])

    # Plot the Time-Frequency representation
    fig, ax = plt.subplots()
    plt.pcolormesh(times, frequencies, tfr_smooth, shading="gouraud")
    plt.colorbar()
    plt.hlines(low_frequency[1], xmin=times[0], xmax=times[-1], colors="white", linestyles="dotted", linewidth=1)
    plt.ylabel("Frequency (Hz)")
    # Convert x-axis labels to minutes
    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
    plt.xlabel("Time (min)")
    plt.title(f"TFR smoothed HRV (CWT) for subject {subject}")
    plt.xlim([times[0], times[-1]])

    if show_plots:
        plt.show()

    plt.close()

    # Get the LF and HF HRV indices and frequencies
    lf_index = np.logical_and(frequencies >= low_frequency[0], frequencies <= low_frequency[1])
    hf_index = np.logical_and(frequencies >= high_frequency[0], frequencies <= high_frequency[1])
    hrv_index = np.logical_and(frequencies >= low_frequency[0], frequencies <= high_frequency[1])
    lf_frequencies = frequencies[lf_index]
    hf_frequencies = frequencies[hf_index]
    hrv_frequencies = frequencies[hrv_index]
    # Get the LF and HF HRV power
    lf_PSDs = tfr_smooth[lf_index, :]
    hf_PSDs = tfr_smooth[hf_index, :]
    hrv_PSDs = tfr_smooth[hrv_index, :]
    # Integrate over frequency bands to get the power
    lf_power = scipy.integrate.trapezoid(lf_PSDs.transpose(), lf_frequencies)
    hf_power = scipy.integrate.trapezoid(hf_PSDs.transpose(), hf_frequencies)
    hrv_power = scipy.integrate.trapezoid(hrv_PSDs.transpose(), hrv_frequencies)

    # Add one NaN at the end, because only N-1 values because of smoothing
    lf_power = np.append(lf_power, np.nan)
    hf_power = np.append(hf_power, np.nan)
    hrv_power = np.append(hrv_power, np.nan)
    times = np.append(times, times[-1] + 1)

    # Cut the mirrored part of the power timeseries from the data
    lf_power_mirrored = lf_power[
        (mirror_length - 1) * sampling_frequency : (len(ibi) + mirror_length - 1) * sampling_frequency
    ]
    hf_power_mirrored = hf_power[
        (mirror_length - 1) * sampling_frequency : (len(ibi) + mirror_length - 1) * sampling_frequency
    ]
    hrv_power_mirrored = hrv_power[
        (mirror_length - 1) * sampling_frequency : (len(ibi) + mirror_length - 1) * sampling_frequency
    ]
    times_mirrored = times[
        (mirror_length - 1) * sampling_frequency : (len(ibi) + mirror_length - 1) * sampling_frequency
    ]

    # Reset times to start at 0
    times_mirrored = times_mirrored - times_mirrored[0]

    # Mirror the IBI data to the same length as the power timeseries
    # Remove last dummy value
    ibi_sl = ibi[:-1]
    symmetric_padding_end = ibi_sl[-mirror_length * sampling_frequency :][::1]
    symmetric_padding_beginning = ibi_sl[: mirror_length * sampling_frequency][::-1]
    ibi_mirrored_original = np.concatenate([symmetric_padding_beginning, ibi_sl, symmetric_padding_end])

    # Cut the mirrored part of the IBI timeseries from the data
    ibi_mirrored_final = ibi_mirrored_original[
        (mirror_length - 1) * sampling_frequency : (len(ibi) + mirror_length - 1) * sampling_frequency
    ]

    return lf_power_mirrored, hf_power_mirrored, hrv_power_mirrored, times_mirrored, ibi_mirrored_final, fig

def calculate_power(
    eeg, sampling_frequency, bands, frequencies, mirror_length, window_length, overlap
):
    """
    Calculate the power of all EEG frequency bands.

    Computes power of all frequency bands from EEG data using Time-Frequency analysis.
    Using MNE's Time-Frequency analysis functions with Continuous Wavelet Transform (CWT).

    Parameters
    ----------
    eeg : mne.io.Raw
        Cleaned EEG data.
    sampling_frequency : int
        Sampling frequency of the EEG data.
    bands : fooof.bands.Bands
        Frequency bands of interest.
    frequencies : array
        Frequencies of the power spectral density (PSD) (accuracy 0.5 Hz).
    mirror_length : float
        Length of the mirror extension.
    window_length : float
        Length of the window for smoothing.
    overlap : float
        Overlap of the windows for smoothing.

    Returns
    -------
    power : dict
        Power of all EEG frequency bands.
    """
    print("Downsampling EEG data to 250 Hz...")
    # Downsampling the EEG to 250 Hz
    downsampled_eeg = eeg.copy().resample(sampling_frequency)
    
    # Load the EEG data
    data = downsampled_eeg.get_data()

    # Set TFR parameters
    window = window_length * sampling_frequency
    noverlap = int(window * overlap)

    print("Mirroring EEG data (symmetric padding)...")
    # Mirror the data at the beginning and end to avoid edge effects (symmetric padding)
    eeg_data_mirrored = np.empty((data.shape[0], data.shape[1] + 2 * mirror_length * sampling_frequency))
    for e in range(0, data.shape[0]):
        eeg_data_mirrored[e, :] = np.hstack(
            (np.flip(data[e])[-mirror_length * sampling_frequency:], data[e], np.flip(data[e][-mirror_length * sampling_frequency:])))

    print("Performing Continuous Wavelet Transform (CWT) for TFR on EEG data...")
    # Get the current time
    start_time = time.ctime()
    print(f"Start time: {start_time}")
    # Perform Time-Frequency analysis
    # Continuous Wavelet Transform (CWT) on mirrored EEG data

    # Create the wavelet for the CWT
    wavelet = mne.time_frequency.morlet(sampling_frequency, frequencies, n_cycles=7.0, sigma=None, zero_mean=False)

    # Compute the TFR for the whole EEG data
    tfr = mne.time_frequency.tfr.cwt(eeg_data_mirrored, wavelet, use_fft=True, mode='same', decim=1)
    tfr_power = (np.abs(tfr)) ** 2

    # Get the current time
    end_time = time.ctime()
    print(f"End time: {end_time}")
    print("CWT for TFR on EEG data completed. Total time elapsed: ", str(end_time - start_time))

    # Cut the power timeseries
    tfr_power_mirrored = tfr_power[:, :, 0: mirror_length * sampling_frequency + data.shape[1]]

    # Smooth the Time-Frequency representation
    # Average over 2s windows with 50% overlap
    # This down-samples the power timeseries to 1 Hz
    PSDs = np.empty((tfr_power_mirrored.shape[0], tfr_power_mirrored.shape[1], int(tfr_power_mirrored.shape[2] / sampling_frequency) - 1))
    for e in range(0, tfr_power_mirrored.shape[0]):
        for f in range(0, tfr_power_mirrored.shape[1]):
            window_avg = [np.mean(tfr_power_mirrored[e, f, i:i + window]) for i in range(0, len(tfr_power_mirrored[e, f]), noverlap) if i + window <= len(tfr_power_mirrored[e, f])]
            PSDs[e, f] = np.asarray(window_avg)
    
    # Separate the power into frequency bands
    # Initialize arrays for the power of each frequency band
    power_delta = np.empty((PSDs.shape[2], PSDs.shape[0],))
    power_theta = np.empty((PSDs.shape[2], PSDs.shape[0],))
    power_alpha = np.empty((PSDs.shape[2], PSDs.shape[0],))
    power_beta = np.empty((PSDs.shape[2], PSDs.shape[0],))
    power_gamma = np.empty((PSDs.shape[2], PSDs.shape[0],))
    # Integrate the power over the frequency bands
    for t in range(0, PSDs.shape[2]):
        PSD = PSDs[:, :, t]
        # Integrate full PSDs over the different frequency bands using the integrate_power() function
        power_delta[t] = integrate_power(frequencies, PSD, bands.delta, "trapezoid")
        power_theta[t] = integrate_power(frequencies, PSD, bands.theta, "trapezoid")
        power_alpha[t] = integrate_power(frequencies, PSD, bands.alpha, "trapezoid")
        power_beta[t] = integrate_power(frequencies, PSD, bands.beta, "trapezoid")
        power_gamma[t] = integrate_power(frequencies, PSD, bands.gamma, "trapezoid")

    # Cut the mirrored part of the power timeseries from the data (data w/o artifacts)
    power_delta_mirrored = power_delta[mirror_length - 1:len(data) + mirror_length - 1]
    power_theta_mirrored = power_theta[mirror_length - 1:len(data) + mirror_length - 1]
    power_alpha_mirrored = power_alpha[mirror_length - 1:len(data) + mirror_length - 1]
    power_beta_mirrored = power_beta[mirror_length - 1:len(data) + mirror_length - 1]
    power_gamma_mirrored = power_gamma[mirror_length - 1:len(data) + mirror_length - 1]

    # Get the times of the power
    times_mirrored = np.arange(0, power_delta_mirrored.shape[0])

    # Combine all power arrays into one dictionary
    power = {"times": times_mirrored, "channels": eeg.ch_names, "delta": power_delta_mirrored, "theta": power_theta_mirrored, "alpha": power_alpha_mirrored, "beta": power_beta_mirrored, "gamma": power_gamma_mirrored}

    return power


def calculate_power_roi(power, roi):
    """
    Calculate the mean power of the EEG frequency bands for a region of interest (ROI).

    Parameters
    ----------
    power : dict
        Power of all EEG frequency bands.
    roi : list[str]
        Channels of the region of interest (ROI).

    Returns
    -------
    power_roi : dict
        Mean power of the EEG frequency bands for the region of interest.
    """
    # Get channel indices of the ROI
    roi_indices = np.isin(power["channels"], roi)
    # Get the power of the ROI
    power_roi = {"times": power["times"], "channels": roi}
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        power_roi[band] = np.mean(power[band][:, roi_indices])

    return power_roi


def integrate_power(frequencies, PSD, band):
    """
    Integrate power over a specific frequency band.
    Using the trapezoid rule to approximate the integral of the power spectral density (PSD) over the frequency band.

    Parameters
    ----------
    frequencies : array
        Frequencies of the PSD.
    PSD : array[float]
        Power spectral density.
    band : list[float]
        Frequency band of interest.


    Returns
    -------
    power : float
        Integrated power over the frequency band.
    """
    # Initialize power array
    power = np.zeros(PSD.shape[0])
    # Get the frequencies within the band
    frequencies_band = frequencies[np.logical_and(frequencies >= band[0], frequencies <= band[1])]
    # Integrate the power over the frequency band
    for i in range(PSD.shape[0]):
        psd = PSD[i]
        psd_band = psd[np.logical_and(frequencies >= band[0], frequencies <= band[1])]
        power[i] = scipy.integrate.trapezoid(psd_band, frequencies_band)

    return power

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. LOAD DATA
if __name__ == "__main__":
    # Loop over subjects
    for subject_index, subject in enumerate(subjects):
        print("--------------------------------------------------------------------------------")
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
        print("--------------------------------------------------------------------------------")

        # Read in file with excluded participants
        with (data_dir / exp_name / derivative_name / preprocessed_name / "excluded_participants.tsv").open("r") as f:
            excluded_participants = f.read().splitlines()

        # Skip excluded participants
        if subject in excluded_participants:
            print(f"Subject {subject} was excluded from the analysis. Skipping subject...")
            continue

        # Define the path to the data
        subject_data_path = (
            data_dir / exp_name / derivative_name / preprocessed_name / f"sub-{subject}" / datatype_name
        )

        # Define the path to the extracted features
        subject_features_folder = (
            data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name
        )

        # Define the path to the results
        subject_results_folder = results_dir / exp_name / f"sub-{subject}" / datatype_name

        if "Load Data" in steps:
            print("********** Loading data **********\n")

            print("Loading participant metadata...")
            # Get the metadata json file
            metadata_file = subject_data_path / f"sub-{subject}_task-{task}_physio_preprocessing_metadata.json"
            with metadata_file.open("r") as file:
                subject_metadata = json.load(file)

            print("Loading ECG data...")
            # Read the preprocessed ECG data
            ecg_file = subject_data_path / f"sub-{subject}_task-{task}_physio_ecg_preprocessed_scaled.tsv"
            ecg_data = pd.read_csv(ecg_file, sep="\t")

            print("Loading EEG data...\n")
            # Read the preprocessed EEG data
            eeg_file = subject_data_path / f"sub-{subject}_task-{task}_eeg_preprocessed_filtered_0.1-45_after_ica.fif"
            eeg_data = mne.io.read_raw_fif(eeg_file, preload=True)

        # %% STEP 2. FEATURE EXTRACTION ECG
        if "Feature Extraction ECG" in steps:
            print("********** Extracting features from ECG data **********\n")

            # Get IBI and heart rate and drop NaN values
            # IBI and HR are already sampled at 1 Hz from the preprocessing
            ibi = ecg_data["IBI"].dropna()
            heart_rate = ecg_data["HR"].dropna()

            print("Extracting IBI, HRV, HF-HRV and LF-HRV...")
            # Compute HRV features using calculate_hrv function
            lf_power, hf_power, hrv_power, times, ibi_mirrored, fig = calculate_hrv(
                ibi, sampling_frequency_ibi, lf_band, hf_band, mirror_length, window_length, overlap, show_plots
            )

            print("Saving ECG features...")
            # Create subject column that has the same length as the features
            subject_column = [subject] * len(times)
            # Save the HRV features in a tsv file
            hrv_features = pd.DataFrame(
                {"timestamp": times, "ibi": ibi_mirrored, "hrv": hrv_power, "lf-hrv": lf_power, "hf-hrv": hf_power}
            )
            hrv_features.to_csv(
                subject_features_folder / f"sub-{subject}_task-{task}_ecg_features.tsv", sep="\t", index=False
            )

            # Save Time-Frequency representation figure
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ecg_tfr_cwt.png")

            # Plot the HRV features
            # Plot Clean IBI
            fig, ax = plt.subplots()
            ax.plot(times, ibi_mirrored, c=colors["ECG"]["IBI"], linewidth=1)
            ax.set_ylabel("IBI (s)")
            # Transform x-axis labels to minutes
            x_ticks = ax.get_xticks()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
            ax.set_xlabel("Time (min)")
            ax.set_title(f"Clean IBI for subject {subject}")
            ax.set_xlim([times[0], times[-1]])

            # Save the IBI plot
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ecg_ibi.png")

            if show_plots:
                plt.show()

            plt.close()

            # Plot HRV features (all in one plot)
            fig, ax = plt.subplots()
            for hrv in ["hrv_power", "hf_power", "lf_power"]:
                if hrv == "lf_power":
                    ax.plot(times, lf_power, c=colors["ECG"]["LF-HRV"], linewidth=1)
                elif hrv == "hf_power":
                    ax.plot(times, hf_power, c=colors["ECG"]["HF-HRV"], linewidth=1)
                elif hrv == "hrv_power":
                    ax.plot(times, hrv_power, c=colors["ECG"]["HRV"], linewidth=1)
            ax.set_ylabel("Power")
            # Transform x-axis labels to minutes
            x_ticks = ax.get_xticks()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
            ax.set_xlabel("Time (min)")
            ax.set_title(f"Clean HRV for subject {subject}")
            ax.set_xlim([times[0], times[-1]])
            ax.legend(["HRV", "HF-HRV", "LF-HRV"])

            print("Saving plots...")
            # Save the HRV plot
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ecg_hrv.png")

            if show_plots:
                plt.show()

            plt.close()

            print("Done with ECG feature extraction!\n")

        # %% STEP 3. FEATURE EXTRACTION EEG
        if "Feature Extraction EEG" in steps:
            print("********** Extracting features from EEG data **********\n")

            # Get only EEG channels
            eeg = eeg_data.pick_types(eeg=True)

            print("Extracting power of EEG frequency bands...")
            # Compute power of all EEG frequency bands using calculate_power() function
            power = calculate_power(eeg, sampling_frequency_eeg, bands, frequencies, mirror_length, window_length, overlap)

            # Power timeseries are now also sampled at 1 Hz

            # Initiate empty dictionary for the power of the ROIs
            power_rois = {}
            # Loop over ROIs and compute power for each ROI using calculate_power_roi() function
            for roi in rois:
                power_roi = calculate_power_roi(power, rois[roi])
                # Add the power of the ROI to the dictionary
                power_rois[roi] = power_roi

            print("Saving EEG features...")

            # Save the power of all EEG frequency bands in a tsv file
            power_df = pd.DataFrame(power)
            power_df.to_csv(
                subject_features_folder / f"sub-{subject}_task-{task}_eeg_features.tsv", sep="\t", index=False
            )

            # Save the power of the ROIs in a tsv file
            power_rois_df = pd.DataFrame(power_rois)
            power_rois_df.to_csv(
                subject_features_folder / f"sub-{subject}_task-{task}_eeg_features_rois.tsv", sep="\t", index=False
            )



# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
