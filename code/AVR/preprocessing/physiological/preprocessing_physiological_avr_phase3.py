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
Last update: 16 July 2024
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
from autoreject import AutoReject
import time

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

# Define whether scaling of the ECG and PPG data should be done
scaling = True
scale_factor = 0.01

# Define cutoff frequencies for bandfiltering EEG data
low_frequency = 0.1
high_frequency = 30

# Define whether to resample the data (from the original 500 Hz)
resample = True
if resample:
    resampling_rate = 250   # in Hz
else:
    resampling_rate = 500   # in Hz

# Define whether autoreject method should be used to detect bad channels and epochs in EEG data
autoreject = True

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

# Enable interactive plots (only works when running in interactive mode)
#%matplotlib qt

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

def preprocess_eeg(raw_data,
    low_frequency: float,
    high_frequency: int,
    resample_rate: float,
    autoreject: bool):
    """
    Preprocess EEG data using the MNE toolbox.

    Filter the data with a bandpass filter, resample it to a specified sampling rate,
    segment it into epochs of 10s, and use Autoreject to detect bad channels and epochs.

    Arguments:
    ----
    low_frequency: float
        Low cut-off frequency in Hz.
    high_frequency: int
        High cut-off frequency in Hz.
    resample_rate: float
        New sampling frequency in Hz.
    autoreject: bool
        If True, autoreject is used to detect bad channels and epochs.

    Returns:
    -------
    resampled_data: mne.io.Raw
    epochs: mne.epochs.Epochs
    reject_log: autoreject.autoreject.RejectLog

    """

    # Filtering
    filtered_data = raw_data.copy().filter(l_freq=low_frequency, h_freq=high_frequency)

    # Resampling
    resampled_data = filtered_data.resample(resample_rate)

    # Segment data into epochs of 10s
    # Even though data is continuous, it is good practice to break it into epochs
    # before detecting bad channels and running ICA
    tstep = 10  # in seconds
    events = mne.make_fixed_length_events(resampled_data, duration=tstep)
    epochs = mne.Epochs(resampled_data, events, tmin=0, tmax=tstep, baseline=None, preload=True)

    # Pick only EEG channels for Autoreject bad channel detection
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

    # Use Autoreject to detect bad channels
    # Autoreject interpolates bad channels, takes quite long and identifies a lot of epochs
    # Here we do not remove any data, we only identify bad channels and epochs and store them in a log
    # Define random state to ensure reproducibility
    if autoreject:
        # Print the running time that it takes to perform Autoreject
        start_time = time.ctime()
        print("Running Autoreject to detect bad channels and epochs...")
        print("This may take a while...")
        print("Start time: ", start_time)
        ar = AutoReject(random_state=42, picks=picks, n_jobs=3, verbose="progressbar")
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)
    
    end_time = time.ctime()
    print("Done with preprocessing and creating clean epochs at time: ", end_time)
    print("Total duration of preprocessing: ", end_time - start_time)
    return resampled_data, epochs, reject_log


def run_ica(epochs: mne.epochs.Epochs, rejected_epochs: np.array):
    """
    Run Independent Component Analysis (ICA) on the preprocessed EEG data (in epochs).

    Arguments:
    ----
    epochs: TODO: specify
    rejected_epochs: TODO: specify

    Returns:
    -------
    ica: TODO: specify

    """
    # Set ICA parameters
    random_state = 42   # ensures ICA is reproducible each time it's run
    ica_n_components = .99  # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
    ica.fit(epochs[~rejected_epochs], decim=3)  # decim reduces the number of time points to speed up computation

    print("Done with ICA.")

    return ica


def ica_correlation(ica: mne.preprocessing.ICA, epochs: mne.epochs.Epochs):
    """
    Select ICA components semi-automatically using a correlation approach with eye movements and cardiac data.

    Arguments:
    ----
    ica: TODO: specify
    epochs: TODO: specify

    Returns:
    ica: TODO: specify
    eog_indices: TODO: specify
    eog_scores: TODO: specify
    ecg_indices: TODO: specify
    ecg_scores: TODO: specify
    -------
    """
    # Create list of components to exclude
    ica.exclude = []

    # Find the right threshold for identifying bad EOG components
    # The default value is 3.0 but this is not the optimal threshold for all datasets
    # We use a while loop to iterate over a range of thresholds until at least 2 ICs are identified
    # We would expect at least 2 ICs from both blinks and horizontal eye movements
    number_ics_eog = 0
    max_ics_eog = 2
    z_threshold = 3.5
    z_step = 0.1

    while number_ics_eog < max_ics_eog:
        # Correlate with EOG channels
        eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold=z_threshold)
        number_ics_eog = len(eog_indices)
        z_threshold -= z_step   # won't impact things if number_ics_eog is already >= max_ics_eog

    print('Final threshold for EOG components: ' + str(z_threshold))
    print('Number of EOG components identified: ' + str(len(eog_indices)))

    # For ECG components, we use the default threshold of 3.0
    # Correlate with ECG channels
    ecg_indices, ecg_scores = ica.find_bads_ecg(epochs, threshold='auto', method='correlation')
    print('Number of ECG components identified: ' + str(len(ecg_indices)))

    # Assign the bad EOG components to the ICA.exclude attribute so they can be removed later
    ica.exclude = eog_indices + ecg_indices

    return ica, eog_indices, eog_scores, ecg_indices, ecg_scores


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
            #%matplotlib qt

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

        # TODO: calculate LF-HRV and HF-HRV from IBI data

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

        # Attributes
        if scaling and manual_cleaning:
            attributes_cardiac = "_scaled_manually-cleaned"
        elif scaling and not manual_cleaning:
            attributes_cardiac = "_scaled"
        elif manual_cleaning and not scaling:
            attributes_cardiac = "_manually-cleaned"
        else:
            attributes_cardiac = ""

        # Save ECG data to tsv file
        ecg_data_df.to_csv(
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_ecg_preprocessed{attributes_cardiac}.tsv", sep="\t", index=False
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
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_ppg_preprocessed{attributes_cardiac}.tsv", sep="\t", index=False
        )

        # ---------------------- 2d. Preprocess EEG data ----------------------
        # Set Montage
        # Set EEG channel layout for topo plots
        montage_filename = data_dir / exp_name / rawdata_name / "CACS-64_REF.bvef"
        montage = mne.channels.read_custom_montage(montage_filename)
        cropped_eeg_data.set_montage(montage)

        # Interpolate the ECG data to match the EEG data
        if len(cleaned_ecg) < len(cropped_eeg_data.times):
            cleaned_ecg = np.interp(cropped_eeg_data.times, np.linspace(0, len(cleaned_ecg), len(cleaned_ecg)), cleaned_ecg)
        # Or crop the ECG data to match the EEG data
        elif len(cleaned_ecg) > len(cropped_eeg_data.times):
            cleaned_ecg = cleaned_ecg[:len(cropped_eeg_data.times)]
        # Or leave it as it is
        else:
            pass
        
        # Add ECG data as a new channel to the EEG data
        ecg_data_channel = mne.io.RawArray([cleaned_ecg], mne.create_info(["ECG"], sampling_rates["ecg"], ["ecg"]))
        cropped_eeg_data.add_channels([ecg_data_channel])

        # Preprocessing EEG data using preprocessing_eeg function
        resampled_data, epochs, reject_log = preprocess_eeg(
            cropped_eeg_data, low_frequency, high_frequency, resampling_rate, autoreject=autoreject)

        # Plot reject_log
        if show_plots:
            fig, ax = plt.subplots(figsize=[15, 5])
            reject_log.plot('horizontal', ax=ax, aspect='auto')
            plt.show()

        # Artifact rejection with ICA using run_ica function
        ica = run_ica(epochs, reject_log.bad_epochs)

        # Plot results of ICA
        if show_plots:
            ica.plot_components(inst=epochs)
            ica.plot_sources(resampled_data)
            ica.plot_overlay(resampled_data, exclude=[0], picks='eeg')

        # Semi-automatic selection of ICA components using ica_correlation function
        ica, eog_indices, eog_scores, ecg_indices, ecg_scores = ica_correlation(ica, epochs)

        # Number of components removed
        print(f"Number of components removed: {len(ica.exclude)}")

        # Plot components and correlation scores
        if show_plots:
            ica.plot_components(inst=epochs, picks=ica.exclude, title="EOG and ECG components to be excluded")
            ica.plot_scores(eog_scores, exclude=ica.exclude, title="EOG scores")

        # Get the explained variance of the ICA components
        explained_variance_ratio = ica.get_explained_variance_ratio(epochs)['eeg']
        print(f"Explained variance ratio of ICA components: {explained_variance_ratio}")

        # Reject components in the resampled data that are not brain related
        eeg_clean = ica.apply(resampled_data.copy())

        if resample and autoreject:
            attributes_eeg = f"resampled_{resampling_rate}_autoreject_filtered_{low_frequency}-{high_frequency}"
        elif resample and not autoreject:
            attributes_eeg = f"resampled_{resampling_rate}_filtered_{low_frequency}-{high_frequency}"
        elif not resample and autoreject:
            attributes_eeg = f"autoreject_filtered_{low_frequency}-{high_frequency}"
        else:
            attributes_eeg = f"filtered_{low_frequency}-{high_frequency}"
        
        # Save the raw data before ICA
        resampled_data.save(subject_preprocessed_folder /
            f"sub-{subject}_task-{task}_eeg_preprocessed_{attributes_eeg}_before_ica.fif", overwrite=True)

        # Save the clean data after ICA
        eeg_clean.save(subject_preprocessed_folder /
            f"sub-{subject}_task-{task}_eeg_preprocessed_{attributes_eeg}_after_ica.fif", overwrite=True)

        # Save the ICA object with the bad components
        ica.save(subject_preprocessed_folder / f"sub-{subject}_task-{task}_eeg_{attributes_eeg}_ica.fif", overwrite=True)



    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS

    # TODO

# %%
