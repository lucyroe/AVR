"""
Script to extract features from physiological data (ECG, EEG) for AVR phase 3 to be later used in statistical analysis.

Required packages:  mne, neurokit2, fcwt, scipy, fooof

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 23 July 2024
Last update: 13 August 2024
"""
# %%
def extract_features(subjects = ["001", "002", "003"],  # noqa: C901, PLR0912, PLR0915, B006
            data_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
            results_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
            show_plots=False,
            debug=False):
    """
    Extract features from EEG and ECG data for AVR phase 3.

    Inputs:     Preprocessed data (EEG, ECG) in tsv files (ECG) and fif files (EEG, after ICA)

    Outputs:    Features extracted from the data in tsv files
                - ECG: IBI, HRV, HF-HRV, LF-HRV
                - EEG: Power of delta, theta, alpha, beta, gamma bands for all electrodes and for each of the ROIs

    Functions:
                calculate_hrv(): Compute heart rate variability (HRV) features from inter-beat intervals (IBI) data.
                calculate_power(): Compute the power of all EEG frequency bands.
                calculate_power_roi(): Compute the mean power of the EEG frequency bands for a region of interest (ROI)
                integrate_power(): Integrate power over a specific frequency band.

    Steps:
    1. LOAD DATA
    2. FEATURE EXTRACTION ECG
        2a. Calculate HRV features from IBI data
        2b. Save HRV features in a tsv file
        2c. Plot HRV features
    3. FEATURE EXTRACTION EEG
        3a. Calculate the power of all EEG frequency bands
        3b. Calculate the mean power of the EEG frequency bands for a region of interest (ROI)
        3c. Save EEG power features in a tsv file
        3d. Plot EEG power features
    4. AVERAGE SELECTED FEATURES ACROSS PARTICIPANTS
    """
    # %% Import
    import json
    import time
    from pathlib import Path

    import fcwt
    import fooof
    import matplotlib.pyplot as plt
    import mne
    import neurokit2 as nk
    import numpy as np
    import pandas as pd
    import scipy

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    task = "AVR"

    # Only analyze one subject when debug mode is on
    if debug:
        subjects = [subjects[0]]

    # Define which steps to run
    steps = []
    # "Load Data", "Feature Extraction ECG", "Feature Extraction EEG", "Average Across Participants"

    data_dir = Path(data_dir) / "phase3"
    exp_name = "AVR"
    derivative_name = "derivatives"  # derivates folder
    preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
    averaged_name = "avg"  # averaged data folder (inside preprocessed)
    feature_name = "features"  # feature extraction folder (inside derivatives)
    datatype_name = "eeg"  # data type specification
    results_dir = Path(results_dir) / "phase3"

    # Create the features data folder if it does not exist
    for subject in subjects:
        subject_features_folder = (data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" /
        datatype_name)
        subject_features_folder.mkdir(parents=True, exist_ok=True)
    avg_features_folder = data_dir / exp_name / derivative_name / feature_name / averaged_name / datatype_name
    avg_features_folder.mkdir(parents=True, exist_ok=True)

    # Define parameters for time-frequency analysis of both ECG and EEG data
    window_length = 2  # 2s window  # Length of the window for smoothing
    overlap = 0.5  # 50% overlap    # Overlap of the windows for smoothing
    mirror_length = 180  # Length of the mirror extension for symmetric padding

    # Define parameters for time-frequency analysis of ECG data
    sampling_frequency_ibi = 1  # Hz    # IBI and HR data are already sampled at 1 Hz from the preprocessing

    # Define low and high frequency bands for HRV analysis
    lf_band = [0.04, 0.15]
    hf_band = [0.15, 0.4]

    # Define parameters for time-frequency analysis of EEG data
    sampling_frequency_eeg = 100  # Hz    # EEG data will be downsampled to 100 Hz
    frequencies = np.arange(0.5, 50.5, 0.5)  # Resolution 0.5 Hz

    # Define frequency bands of interest
    # Berger, 1929: delta (δ; 0.3-4 Hz), theta (θ; 4-8 Hz),
    # alpha (α; 8-12 Hz), beta (β; 12-30 Hz) and gamma (γ; 30-45 Hz)  # noqa: RUF003
    bands = fooof.bands.Bands({"delta": [0.3, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30],
    "gamma": [30, 45]})

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
        "fronto-central": ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6"],
        "central": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
        "centro-parietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
        "parietal": ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
        "parieto-occipital": ["PO7", "PO3", "POz", "PO4", "PO8"],
        "occipital": ["O1", "Oz", "O2"],
        "posterior": [
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
        ],
    }

    # Create color palette for plots
    colors = {
        "ECG": {"IBI": "#F0E442", "HRV": "#CC79A7", "LF-HRV": "#E69F00", "HF-HRV": "#D55E00"},
        # yellow, pink, light orange, dark orange
        "EEG": {"delta": "#F0E442", "theta": "#D55E00", "alpha": "#CC79A7", "beta": "#56B4E9", "gamma": "#009E73"},
        # yellow, dark orange, pink, light blue, green
    }

    # Features for averaging across participants
    features_averaging = {"ecg": ["ibi", "hrv", "lf-hrv", "hf-hrv"],
                "eeg": ["posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"]}

    # Get rid of the sometimes excessive logging of MNE
    mne.set_log_level("error")

    # Enable interactive plots (only works when running in interactive mode)
    # %matplotlib qt


    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
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
        lf_psds = tfr_smooth[lf_index, :]
        hf_psds = tfr_smooth[hf_index, :]
        hrv_psds = tfr_smooth[hrv_index, :]
        # Integrate over frequency bands to get the power
        lf_power = scipy.integrate.trapezoid(lf_psds.transpose(), lf_frequencies)
        hf_power = scipy.integrate.trapezoid(hf_psds.transpose(), hf_frequencies)
        hrv_power = scipy.integrate.trapezoid(hrv_psds.transpose(), hrv_frequencies)

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


    def calculate_power(eeg, sampling_frequency, bands, frequencies, mirror_length, window_length, overlap):  # noqa: PLR0913, PLR0915
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
        print(f"Downsampling EEG data to {sampling_frequency} Hz...")
        # Downsampling the EEG to 100 Hz
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
                (
                    np.flip(data[e])[-mirror_length * sampling_frequency :],
                    data[e],
                    np.flip(data[e][-mirror_length * sampling_frequency :]),
                )
            )

        print("Performing Continuous Wavelet Transform (CWT) for TFR on EEG data...")
        # Get the current time
        start_time = time.ctime()
        print(f"Start time: {start_time}")
        # Perform Time-Frequency analysis
        # Continuous Wavelet Transform (CWT) on mirrored EEG data

        # Loop over all channels and compute the CWT for each channel
        # Initialize array for the CWT
        tfr = np.empty((eeg_data_mirrored.shape[0], len(frequencies), eeg_data_mirrored.shape[1]))
        for channel in range(0, eeg_data_mirrored.shape[0]):
            # Compute the TFR for each channel
            _, tfr_channel = fcwt.cwt(
                eeg_data_mirrored[channel], sampling_frequency, min(frequencies), max(frequencies), len(frequencies)
            )
            # Convert complex TFR values to real values
            tfr_channel = np.abs(tfr_channel)
            # Flip the TFR matrix to match the frequencies
            tfr_channel = np.flip(tfr_channel, axis=0)
            tfr[channel] = tfr_channel

        # Compute the power of the TFR
        tfr_power = tfr**2

        # Get the current time
        end_time = time.ctime()
        print(f"End time: {end_time}")
        print("CWT for TFR on EEG data completed.")

        print("Smoothing the Time-Frequency representation by averaging over 2s windows, "
        "thereby downsampling to 1 Hz...")
        # Smooth the Time-Frequency representation
        # Average over 2s windows with 50% overlap
        # This down-samples the power timeseries to 1 Hz
        psds = np.empty((tfr_power.shape[0], tfr_power.shape[1], int(tfr_power.shape[2] / sampling_frequency) - 1))
        for e in range(0, tfr_power.shape[0]):
            psds_channel = np.empty((tfr_power.shape[1], int(tfr_power.shape[2] / sampling_frequency) - 1))
            for f in range(0, tfr_power.shape[1]):
                window_avg = [
                    np.mean(tfr_power[e, f, i : i + window])
                    for i in range(0, len(tfr_power[e, f]), noverlap)
                    if i + window <= len(tfr_power[e, f])
                ]
                times = np.arange(0, len(window_avg))
                psds_channel[f] = np.asarray(window_avg)

            # Save the Time-Frequency representation
            psds[e] = psds_channel

        # Create a plot with all TFRs for all channels
        figure, axs = plt.subplots(12, 5, figsize=(20, 20))
        for e in range(0, psds.shape[0]):
            row = e // 5
            col = e % 5
            subfigure = axs[row, col].pcolormesh(times, frequencies, psds[e], shading="gouraud")
            axs[row, col].set_title(f"Channel {eeg.ch_names[e]}")
            axs[row, col].set_ylabel("Frequency (Hz)")
            figure.colorbar(subfigure, ax=axs[row, col])
            axs[row, col].set_xlim([times[0], times[-1]])
            # Convert x-axis labels to minutes
            x_ticks = axs[row, col].get_xticks()[:-1]
            axs[row, col].set_xticks(x_ticks)
            axs[row, col].set_xticklabels([f"{round(x/60)}" for x in x_ticks])
            axs[row, col].set_xlabel("Time (min)")
        figure.suptitle(f"Smoothed Time-Frequency Representation (fCWT) for EEG data for subject {subject}")
        figure.tight_layout()

        # Save the Time-Frequency representation plot
        figure.savefig(subject_results_folder / f"sub-{subject}_task-{task}_eeg_tfr_cwt.png")

        if show_plots:
            plt.show()

        plt.close()

        print("Calculating power of each EEG frequency band...")
        # Separate the power into frequency bands
        # Initialize arrays for the power of each frequency band
        power_delta = np.empty(
            (
                psds.shape[2],
                psds.shape[0],
            )
        )
        power_theta = np.empty(
            (
                psds.shape[2],
                psds.shape[0],
            )
        )
        power_alpha = np.empty(
            (
                psds.shape[2],
                psds.shape[0],
            )
        )
        power_beta = np.empty(
            (
                psds.shape[2],
                psds.shape[0],
            )
        )
        power_gamma = np.empty(
            (
                psds.shape[2],
                psds.shape[0],
            )
        )
        # Integrate the power over the frequency bands
        for t in range(0, psds.shape[2]):
            psd = psds[:, :, t]
            # Integrate full PSDs over the different frequency bands using the integrate_power() function
            power_delta[t] = integrate_power(frequencies, psd, bands.delta)
            power_theta[t] = integrate_power(frequencies, psd, bands.theta)
            power_alpha[t] = integrate_power(frequencies, psd, bands.alpha)
            power_beta[t] = integrate_power(frequencies, psd, bands.beta)
            power_gamma[t] = integrate_power(frequencies, psd, bands.gamma)

        print("Cutting the mirrored part of the power timeseries from the data...")
        # Cut the mirrored part of the power timeseries from the data (data w/o artifacts)
        power_delta_mirrored = power_delta[mirror_length - 1 : power_delta.shape[0] - mirror_length - 1]
        power_theta_mirrored = power_theta[mirror_length - 1 : power_theta.shape[0] - mirror_length - 1]
        power_alpha_mirrored = power_alpha[mirror_length - 1 : power_alpha.shape[0] - mirror_length - 1]
        power_beta_mirrored = power_beta[mirror_length - 1 : power_beta.shape[0] - mirror_length - 1]
        power_gamma_mirrored = power_gamma[mirror_length - 1 : power_gamma.shape[0] - mirror_length - 1]

        # Get the times of the power
        times_mirrored = np.arange(0, power_delta_mirrored.shape[0])

        # Combine all power arrays into one dictionary
        power = {
            "times": times_mirrored,
            "channels": eeg.ch_names,
            "delta": power_delta_mirrored,
            "theta": power_theta_mirrored,
            "alpha": power_alpha_mirrored,
            "beta": power_beta_mirrored,
            "gamma": power_gamma_mirrored,
        }

        print("Done with EEG power calculation!")

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
            power_roi_band = power[band][:, roi_indices]
            # Calculate the mean power of the ROI for each time point
            power_roi[band] = np.mean(power_roi_band, axis=1)

        return power_roi


    def integrate_power(frequencies, psd, band):
        """
        Integrate power over a specific frequency band.

        Using the trapezoid rule to approximate the integral of the power spectral density (PSD) over the band

        Parameters
        ----------
        frequencies : array
            Frequencies of the psd.
        psd : array[float]
            Power spectral density.
        band : list[float]
            Frequency band of interest.


        Returns
        -------
        power : float
            Integrated power over the frequency band.
        """
        # Initialize power array
        power = np.zeros(psd.shape[0])
        # Get the frequencies within the band
        frequencies_band = frequencies[np.logical_and(frequencies >= band[0], frequencies <= band[1])]
        # Integrate the power over the frequency band
        for i in range(psd.shape[0]):
            psd_channel = psd[i]
            psd_band = psd_channel[np.logical_and(frequencies >= band[0], frequencies <= band[1])]
            power[i] = scipy.integrate.trapezoid(psd_band, frequencies_band)

        return power


    # %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. LOAD DATA
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

        # Read in events file
        events_file = data_dir / exp_name / derivative_name / preprocessed_name / "events_experiment.tsv"
        events_experiment = pd.read_csv(events_file, sep="\t")

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

            print("Loading ECG data...")
            # Read the preprocessed ECG data
            ecg_file = (subject_data_path /
                f"sub-{subject}_task-{task}_physio_ecg_preprocessed_scaled_manually-cleaned.tsv")
            ecg_data = pd.read_csv(ecg_file, sep="\t")

            print("Loading EEG data...\n")
            # Read the preprocessed EEG data
            eeg_file = subject_data_path / f"sub-{subject}_task-{task}_eeg_preprocessed_filtered_0.1-45_after_ica.fif"
            eeg_data = mne.io.read_raw_fif(eeg_file, preload=True)

        # %% STEP 2. FEATURE EXTRACTION ECG
        if "Feature Extraction ECG" in steps:
            print("********** Extracting features from ECG data **********\n")

            # Get IBI and drop NaN values
            # IBI is already sampled at 1 Hz from the preprocessing
            ibi = ecg_data["IBI"].dropna()

            print("Extracting IBI, HRV, HF-HRV and LF-HRV...")
            # Compute HRV features using calculate_hrv function
            lf_power, hf_power, hrv_power, times, ibi_mirrored, fig = calculate_hrv(
                ibi, sampling_frequency_ibi, lf_band, hf_band, mirror_length, window_length, overlap, show_plots
            )

            print("Saving ECG features...")
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
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(times, ibi_mirrored, c=colors["ECG"]["IBI"], linewidth=1)
            ax.set_ylabel("IBI (s)")
            # Add vertical lines for event markers
            # Exclude first and last event markers
            # And only use every second event marker to avoid overlap
            for _, row in events_experiment.iloc[0:-1:2].iterrows():
                plt.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)
                plt.text(row["onset"] - 100, min(ibi_mirrored)-0.1*min(ibi_mirrored), row["event_name"], rotation=30,
                fontsize=8, color="gray")

            # Transform x-axis labels to minutes
            x_ticks = ax.get_xticks()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
            ax.set_xlabel("Time (min)")
            ax.set_title(f"Clean IBI for subject {subject}")

            # Save the IBI plot
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ecg_ibi.png")

            if show_plots:
                plt.show()

            plt.close()

            # Plot HRV features (all in one plot)
            fig, ax = plt.subplots(figsize=(15, 5))
            for hrv in ["hrv_power", "hf_power", "lf_power"]:
                if hrv == "lf_power":
                    ax.plot(times, lf_power, c=colors["ECG"]["LF-HRV"], linewidth=1)
                elif hrv == "hf_power":
                    ax.plot(times, hf_power, c=colors["ECG"]["HF-HRV"], linewidth=1)
                elif hrv == "hrv_power":
                    ax.plot(times, hrv_power, c=colors["ECG"]["HRV"], linewidth=1)
            ax.set_ylabel("Power (a.u.)")
            # Add vertical lines for event markers
            # Exclude first and last event markers
            # And only use every second event marker to avoid overlap
            for _, row in events_experiment.iloc[0:-1:2].iterrows():
                plt.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)
                plt.text(row["onset"] - 100, min(lf_power)-1.5*min(lf_power), row["event_name"], rotation=30,
                fontsize=8, color="gray")

            # Transform x-axis labels to minutes
            x_ticks = ax.get_xticks()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
            ax.set_xlabel("Time (min)")
            ax.set_title(f"Clean HRV for subject {subject}")
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
            power = calculate_power(
                eeg, sampling_frequency_eeg, bands, frequencies, mirror_length, window_length, overlap
            )

            # Power timeseries are now also sampled at 1 Hz

            print("Calculating power of EEG frequency bands for all ROIs...")
            # Initiate empty dictionary for the power of the ROIs
            power_rois = {}
            # Loop over ROIs and compute power for each ROI using calculate_power_roi() function
            for roi in rois:
                print(f"Calculating power of EEG frequency bands for {roi} electrodes...")
                power_roi = calculate_power_roi(power, rois[roi])

                # Add the power of the ROI to the dictionary
                power_rois[roi] = power_roi

                # Create a plot of the power values of all frequencies for each ROI
                figure, ax = plt.subplots(figsize=(15, 5))
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    ax.plot(power_roi["times"], power_roi[band], c=colors["EEG"][band], linewidth=1)
                ax.set_ylabel("Power (a.u.)")

                # Add vertical lines for event markers
                # Exclude first and last event markers
                # And only use every second event marker to avoid overlap
                for _, row in events_experiment.iloc[0:-1:2].iterrows():
                    plt.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)
                    plt.text(row["onset"] - 100, max(power_roi["alpha"]) - 0.1 * max(power_roi["alpha"]),
                    row["event_name"], rotation=30, fontsize=8, color="gray")
                # Transform x-axis labels to minutes
                x_ticks = ax.get_xticks()
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
                ax.set_xlabel("Time (min)")
                ax.set_title(f"Power of EEG frequency bands for {roi} electrodes for subject {subject}")
                ax.legend(["Delta", "Theta", "Alpha", "Beta", "Gamma"])

                # Save the plot
                figure.savefig(subject_results_folder / f"sub-{subject}_task-{task}_eeg_{roi}_power.png")

                if show_plots:
                    plt.show()

            print("Saving plots...")

            print("Saving EEG features...")

            # Save the power of each frequency band in a tsv file
            for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                power_band_df = pd.DataFrame()
                # First column is the time
                power_band_df["timestamp"] = power["times"]
                # Add the power of each channel
                for channel in power["channels"]:
                    power_band_df[channel] = power[band][:, power["channels"].index(channel)]
                # Save the power of the frequency band in a tsv file
                power_band_df.to_csv(
                    subject_features_folder / f"sub-{subject}_task-{task}_eeg_features_all_channels_{band}_power.tsv",
                    sep="\t",
                    index=False,
                )

            # Save the power of each ROI in a tsv file
            for roi in power_rois:
                power_roi_df = pd.DataFrame()
                # First column is the time
                power_roi_df["timestamp"] = power_rois[roi]["times"]
                # Add the power of each frequency band
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    power_roi_df[band] = power_rois[roi][band]
                # Save the power of the ROI in a tsv file
                power_roi_df.to_csv(
                    subject_features_folder / f"sub-{subject}_task-{task}_eeg_features_{roi}_power.tsv",
                    sep="\t",
                    index=False,
                )

            # Create a metadata json file with the channels for each ROI
            metadata = {"ROIs": {roi: rois[roi] for roi in rois}}

            # Save the metadata json file
            with (subject_features_folder / f"sub-{subject}_task-{task}_eeg_features_metadata.json").open("w") as file:
                json.dump(metadata, file)

            print("Done with EEG feature extraction!\n")

    # %% STEP 4. AVERAGE SELECTED FEATURES ACROSS PARTICIPANTS
    if "Average Across Participants" in steps:
        print("********** Averaging features across participants **********\n")

        # Initiate empty dataframes for the features
        ecg_features_all = pd.DataFrame()
        eeg_features_all = pd.DataFrame()

        # Load the features of all participants
        for subject in subjects:
            ecg_features = pd.read_csv(
                data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name /
                f"sub-{subject}_task-{task}_ecg_features.tsv",
                sep="\t",
            )
            ecg_features_selected = pd.DataFrame()
            for feature in features_averaging["ecg"]:
                ecg_features_selected[feature] = ecg_features[feature]

            # Add timestamp column as first column
            ecg_features_selected.insert(0, "timestamp", ecg_features["timestamp"])

            # Add column with subject ID as second column
            ecg_features_selected.insert(1, "subject", subject)

            # Add the features of the participant to the dataframe
            ecg_features_all = pd.concat([ecg_features_all, ecg_features_selected], axis=0)

            eeg_features_selected = pd.DataFrame()
            for feature in features_averaging["eeg"]:
                if len(feature.split("_")) == 2:
                    feature_roi, feature_band = feature.split("_")
                    eeg_features = pd.read_csv(
                        data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name /
                        f"sub-{subject}_task-{task}_eeg_features_{feature_roi}_power.tsv",
                        sep="\t",
                    )
                    eeg_features_selected[feature] = eeg_features[feature_band]
                else:
                    eeg_features = pd.read_csv(
                        data_dir / exp_name / derivative_name / feature_name / f"sub-{subject}" / datatype_name /
                        f"sub-{subject}_task-{task}_eeg_features_whole-brain_power.tsv",
                        sep="\t",
                    )
                    eeg_features_selected[feature] = eeg_features[feature]

            # Add timestamp column as first column
            eeg_features_selected.insert(0, "timestamp", eeg_features["timestamp"])

            # Add column with subject ID as second column
            eeg_features_selected.insert(1, "subject", subject)

            # Add the features of the participant to the dataframe
            eeg_features_all = pd.concat([eeg_features_all, eeg_features_selected], axis=0)

        # Select only numeric columns
        numeric_columns_ecg = ecg_features_all.select_dtypes(include=[np.number]).columns
        numeric_columns_eeg = eeg_features_all.select_dtypes(include=[np.number]).columns

        # Average the features across participants
        ecg_features_mean = ecg_features_all[numeric_columns_ecg].groupby("timestamp").mean().reset_index()
        eeg_features_mean = eeg_features_all[numeric_columns_eeg].groupby("timestamp").mean().reset_index()

        # Make sure both dataframes have the same length
        min_length = min(len(ecg_features_mean), len(eeg_features_mean))
        ecg_features_mean = ecg_features_mean.iloc[:min_length]
        eeg_features_mean = eeg_features_mean.iloc[:min_length]

        # Add the features to one dataframe
        features_all = pd.concat([ecg_features_mean, eeg_features_mean], axis=1)

        # Delete the timestamp column of the second dataframe
        features_all = features_all.loc[:, ~features_all.columns.duplicated()]

        # Save the averaged features in a tsv file
        features_all.to_csv(
            data_dir / exp_name / derivative_name / feature_name / averaged_name / datatype_name /
            f"avg_task-{task}_physio_features.tsv", sep="\t", index=False
        )

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    extract_features()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
