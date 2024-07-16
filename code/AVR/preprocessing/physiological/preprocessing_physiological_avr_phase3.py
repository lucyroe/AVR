"""
Script to preprocess physiological data (EEG, ECG, PPG) for AVR phase 3.

Inputs: Raw EEG data in .edf files, ECG and PPG data in tsv.gz files

Outputs: Preprocessed data (EEG, ECG, PPG) in tsv files

Functions:
    plot_peaks(): Plot ECG or PPG signal with peaks.
    preprocess_eeg(): Preprocess EEG data using the MNE toolbox.
    run_ica(): Run Independent Component Analysis (ICA) on the preprocessed EEG data (in epochs).
    ica_correlation(): Select ICA components semi-automatically using a correlation approach with eye movements
                        and cardiac data.

Steps:
1. LOAD DATA
    1a. Load EEG data
    1b. Load ECG and PPG data
    1c. Load event markers
    1d. Load event mapping
2. PREPROCESS DATA
    2a. Cutting data
    2b. Format data
    2c. Preprocess ECG and PPG data & save to tsv files
    2d. Preprocess EEG data & save to fif files
3. AVERAGE OVER ALL PARTICIPANTS
    3a. TODO

Required packages: mne, neurokit, systole, seaborn, autoreject

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 6 July 2024
Last update: 16 July 2024
"""
# %% Import
import gzip
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
from autoreject import AutoReject
from IPython.display import display
from systole.interact import Editor

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = []  # Adjust as needed
# "001", "002", "003"   # already done
task = "AVR"

# Only analyze one subject when debug mode is on
debug = False
if debug:
    subjects = [subjects[0]]

# Define if plots for preprocessing steps should be shown
show_plots = False

# Define whether manual cleaning of R-peaks should be done
manual_cleaning = False

# Define whether scaling of the ECG and PPG data should be done
scaling = True
if scaling:
    scale_factor = 0.01

# Define cutoff frequencies for bandfiltering EEG data
low_frequency = 0.1 # in Hz
high_frequency = 30 # in Hz

# Define whether to resample the data (from the original 500 Hz)
resample = True
resampling_rate = 250 if resample else 500  # in Hz

# Define whether autoreject method should be used to detect bad channels and epochs in EEG data
autoreject = True

# Define if the first and last 2.5 seconds of the data should be cut off
# To avoid any potential artifacts at the beginning and end of the experiment
cut_off_seconds = 2.5

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

# Create color palette for plots
colors = {"ECG": ["#F0E442", "#D55E00"],    # yellow and dark orange
            "PPG": ["#E69F00", "#CC79A7"],   # light orange and pink
            "EEG": ["#56B4E9", "#0072B2", "#009E73"],  # light blue, dark blue, and green
            "others": ["#FFFFFF", "#6C6C6C", "#000000"] # white, gray, and black
            }

# Get rid of the sometimes excessive logging of MNE
mne.set_log_level("error")

# Enable interactive plots (only works when running in interactive mode)
# %matplotlib qt


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def plot_peaks(
    cleaned_signal: dict | np.ndarray,
    peaks: np.ndarray,
    time_range: tuple[float, float],
    plot_title: str,
    sampling_rate: int,
):
    """
    Plot ECG or PPG signal with peaks.

    Arguments:
    ---------
    cleaned_signal : dict or np.ndarray
        The signal data to be plotted. Can be a dictionary or a NumPy ndarray.
    peaks : np.ndarry
        Indices of the peaks within the signal.
    time_range : tuple
        A tuple containing the starting and ending times (in seconds) of the interval to be plotted.
    plot_title : str
        The title of the plot. This argument is optional.
    sampling_rate : int
        The sampling rate of the signal, in Hz.

    """
    # Transform min_time and max_time to samples
    min_sample = int(time_range[0] * sampling_rate)
    max_sample = int(time_range[1] * sampling_rate)
    # Create a time vector (samples)
    time = np.arange(min_sample, max_sample)
    fig, axs = plt.subplots(figsize=(15, 5))

    # Select signal and peaks interval to plot
    selected_signal = cleaned_signal[min_sample:max_sample]
    selected_peaks = peaks[(peaks < max_sample) & (peaks >= min_sample)]

    # Transform data from mV to V
    selected_signal = selected_signal / 1000

    # Choose color palette based on the plot title
    if "ECG" in plot_title:
        linecolor = colors["ECG"][1]
        circlecolor = colors["ECG"][0]
    elif "PPG" in plot_title:
        linecolor = colors["PPG"][1]
        circlecolor = colors["PPG"][0]

    axs.plot(time, selected_signal, linewidth=1, label="Signal", color=linecolor)
    axs.scatter(selected_peaks, selected_signal[selected_peaks - min_sample], color=circlecolor,
        edgecolor=circlecolor, linewidth=1, alpha=0.6)
    axs.set_ylabel("ECG" if "ECG" in plot_title else "PPG")
    axs.set_xlabel("Time (s)")
    x_ticks = axs.get_xticks()
    axs.set_xticks(x_ticks)
    # Transform x-axis to seconds
    axs.set_xticklabels([f"{x/sampling_rate}" for x in x_ticks])
    if plot_title:
        axs.set_title(plot_title)

    sns.despine()
    plt.show()
    plt.close()


def preprocess_eeg(
    raw_data: mne.io.Raw, low_frequency: float, high_frequency: int, resample_rate: float, autoreject: bool
):
    """
    Preprocess EEG data using the MNE toolbox.

    Filter the data with a bandpass filter, resample it to a specified sampling rate,
    segment it into epochs of 10s, and use Autoreject to detect bad channels and epochs.

    Arguments:
    ---------
    raw_data: mne.io.Raw
        The raw EEG data to be preprocessed. This should be an instance of mne.io.Raw, which contains the EEG signal
        data along with additional information about the recording.
    low_frequency: float
        Low cut-off frequency in Hz for the bandpass filter.
    high_frequency: int
        High cut-off frequency in Hz for the bandpass filter.
    resample_rate: float
        New sampling frequency in Hz. The data will be resampled to this frequency.
    autoreject: bool
        If True, autoreject is used to detect and interpolate bad channels and epochs automatically.

    Returns:
    -------
    resampled_data: mne.io.Raw
        The resampled raw data after preprocessing.
    epochs: mne.epochs.Epochs
        The segmented epochs extracted from the resampled data.
    reject_log: autoreject.autoreject.RejectLog
        A log of the rejected epochs and channels.
    """
    # Filtering
    print("Filtering data...")
    filtered_data = raw_data.copy().filter(l_freq=low_frequency, h_freq=high_frequency)

    # Resampling
    print("Resampling data...")
    resampled_data = filtered_data.resample(resample_rate)

    # Segment data into epochs of 10s
    # Even though data is continuous, it is good practice to break it into epochs
    # before detecting bad channels and running ICA
    print("Segmenting data into epochs...")
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
        print("Detecting bad channels and epochs...")
        print("This may take a while...")
        print("Start time: ", start_time)
        ar = AutoReject(random_state=42, picks=picks, n_jobs=3, verbose="progressbar")
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)

    end_time = time.ctime()
    print("Done with preprocessing and creating clean epochs at time: ", end_time)

    # Convert time strings to struct_time
    start_time_struct = time.strptime(start_time, "%a %b %d %H:%M:%S %Y")
    end_time_struct = time.strptime(end_time, "%a %b %d %H:%M:%S %Y")
    # Convert struct_time to epoch timestamp
    start_timestamp = time.mktime(start_time_struct)
    end_timestamp = time.mktime(end_time_struct)
    # Calculate the total duration of the preprocessing
    duration_seconds = end_timestamp - start_timestamp
    # Convert seconds to more readable format
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total duration of preprocessing: {int(minutes)} minutes, {int(seconds)} seconds")

    return resampled_data, epochs, reject_log


def run_ica(epochs: mne.epochs.Epochs, rejected_epochs: np.array):
    """
    Run Independent Component Analysis (ICA) on the preprocessed EEG data (in epochs).

    Arguments:
    ---------
    epochs: mne.epochs.Epochs
        The epochs on which ICA will be run. This should be the output from the preprocess_eeg().
    rejected_epochs: np.array
        An array of indices for epochs that have been marked as bad and should be excluded from the ICA.

    Returns:
    -------
    ica: mne.preprocessing.ICA
        The ICA object after fitting it to the epochs data, excluding the rejected epochs.
    """
    # Set ICA parameters
    random_state = 42  # ensures ICA is reproducible each time it's run
    ica_n_components = 0.99  # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    print("Fitting ICA...")
    ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
    ica.fit(epochs[~rejected_epochs], decim=3)  # decim reduces the number of time points to speed up computation
    print("Done with ICA.")

    return ica


def ica_correlation(ica: mne.preprocessing.ICA, epochs: mne.epochs.Epochs):
    """
    Select ICA components semi-automatically using a correlation approach with eye movements and cardiac data.

    Arguments:
    ---------
    ica: mne.preprocessing.ICA
        The ICA object containing the components to be examined. This should be the output from run_ica().
    epochs: mne.epochs.Epochs
        The epochs data used for correlating with the ICA components. This should be the output from preprocess_eeg().

    Returns:
    -------
    ica: mne.preprocessing.ICA
        The ICA object with the bad components marked for exclusion.
    eog_indices: list
        Indices of the ICA components identified as related to eye movements.
    eog_scores: list
        Correlation scores for the eye movement components.
    ecg_indices: list
        Indices of the ICA components identified as related to cardiac activity.
    ecg_scores: list
        Correlation scores for the cardiac components.
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

    print("Finding threshold for EOG components...")
    print("Correlating ICs with EOG channels...")
    while number_ics_eog < max_ics_eog:
        # Correlate with EOG channels
        eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold=z_threshold)
        number_ics_eog = len(eog_indices)
        z_threshold -= z_step  # won't impact things if number_ics_eog is already >= max_ics_eog

    print("Final threshold for EOG components: " + str(z_threshold))
    print("Number of EOG components identified: " + str(len(eog_indices)))

    # For ECG components, we use the default threshold of 3.0
    # Correlate with ECG channels
    print("Using the default threshold of 3.0 for ECG components...")
    print("Correlating ICs with ECG channels...")
    ecg_indices, ecg_scores = ica.find_bads_ecg(epochs, threshold="auto", method="correlation")
    print("Number of ECG components identified: " + str(len(ecg_indices)))

    # Assign the bad EOG components to the ICA.exclude attribute so they can be removed later
    ica.exclude = eog_indices + ecg_indices
    print("Correlation done.")

    return ica, eog_indices, eog_scores, ecg_indices, ecg_scores


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. LOAD DATA
if __name__ == "__main__":
    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print("--------------------------------------------------------------------------------")
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
        print("--------------------------------------------------------------------------------")

        # Define the path to the data
        subject_data_path = data_dir / exp_name / rawdata_name / f"sub-{subject}" / datatype_name

        # Define the path to the preprocessed data
        subject_preprocessed_folder = (data_dir / exp_name / derivative_name / preprocessed_name
                                    / f"sub-{subject}" / datatype_name)

        # Define the path to the results
        subject_results_folder = results_dir / exp_name / f"sub-{subject}" / datatype_name

        print("********** Loading data **********\n")

        # Get the info json files
        info_eeg_path = subject_data_path / f"sub-{subject}_task-{task}_eeg.json"
        with info_eeg_path.open() as info_eeg_file:
            info_eeg = json.load(info_eeg_file)

        info_channels_path = subject_data_path / f"sub-{subject}_task-{task}_channels.tsv"
        info_channels = pd.read_csv(info_channels_path, sep="\t")

        info_physio_path = subject_data_path / f"sub-{subject}_task-{task}_physio.json"
        with info_physio_path.open() as info_physio_file:
            info_physio = json.load(info_physio_file)

        # Get the EOG channels
        eog_channels = []
        for channel in info_channels.iterrows():
            if "EOG" in channel[1]["type"]:
                eog_channels.append(channel[1]["name"])

        # Read in EEG data
        raw_eeg_data = mne.io.read_raw_edf(
            subject_data_path / f"sub-{subject}_task-{task}_eeg.edf", eog=eog_channels, preload=True
        )

        # Unzip and read in other physiological data (ECG, PPG)
        file = subject_data_path / f"sub-{subject}_task-{task}_physio.tsv.gz"
        with gzip.open(file, "rt") as f:
            raw_physio_data = pd.read_csv(f, sep="\t")
        # Separate ECG and PPG data
        raw_ecg_data = pd.DataFrame(data=raw_physio_data, columns=["timestamp", "cardiac"])
        raw_ppg_data = pd.DataFrame(data=raw_physio_data, columns=["timestamp", "ppg"])

        # Get the sampling rates of the data from info json files
        sampling_rates = {
            "eeg": info_eeg["SamplingFrequency"],
            "ecg": info_physio["SamplingFrequency"],
            "ppg": info_physio["SamplingFrequency"],
        }

        # Load event markers for subject
        event_markers = pd.read_csv(
            data_dir / exp_name / rawdata_name / f"sub-{subject}" / "beh" / f"sub-{subject}_task-{task}_events.tsv",
            sep="\t",
        )

        # Load mapping for event markers to real events
        mapping_filename = data_dir / exp_name / rawdata_name / "events_mapping.tsv"
        if mapping_filename.exists():
            event_mapping = pd.read_csv(mapping_filename, sep="\t")
        else:
            print(
            "ERROR! No event mapping file found. Using default event markers, which means that the different videos "
            "cannot be separated by their names."
            )
            event_mapping = event_markers.copy()

        # Drop column with trial type
        event_mapping = event_mapping.drop(columns=["trial_type"])

        # Add column with event names to event markers
        events = pd.concat([event_markers, event_mapping], axis=1)

        # Drop unnecessary columns
        events = events.drop(columns=["duration"])

        # Set event time to start at delay of first event after beginning of recording
        events["onset"] = events["onset"] - raw_ecg_data["timestamp"].loc[0]

        # Set time to start at 0
        # EEG data already starts at 0
        raw_ecg_data["timestamp"] = raw_ecg_data["timestamp"] - raw_ecg_data["timestamp"].loc[0]
        raw_ppg_data["timestamp"] = raw_ppg_data["timestamp"] - raw_ppg_data["timestamp"].loc[0]

        # %% STEP 2. PREPROCESS DATA
        # ---------------------- 2a. Cutting data ----------------------
        print("********** Cutting data **********\n")
        # Get start and end time of the experiment
        start_time = events[events["event_name"] == "start_spaceship"].reset_index()["onset"].tolist()[0]
        end_time = events[events["event_name"] == "end_spaceship"].reset_index()["onset"].tolist()[-1]

        # Get events for experiment (from start to end of experiment)
        events_experiment = events[(events["onset"] >= start_time) & (events["onset"] <= end_time)]
        # Delete unnecessary column trial_type
        events_experiment = events_experiment.drop(columns=["trial_type"])

        print("Cutting resting state, training phase and instructions before the experiment...")
        print("Data that is left is only of the experiment itself.")

        # Cut data to start and end time
        # And remove first and last 2.5 seconds of data (if specified above)
        if cut_off_seconds > 0:
            print(f"Removing first and last {cut_off_seconds} seconds of data...\n")
            cropped_eeg_data = raw_eeg_data.copy().crop(
                tmin=(start_time + cut_off_seconds), tmax=(end_time - cut_off_seconds)
            )
            cropped_ecg_data = raw_ecg_data[
                (raw_ecg_data["timestamp"] >= (start_time + cut_off_seconds))
                & (raw_ecg_data["timestamp"] <= (end_time - cut_off_seconds))
            ]
            cropped_ppg_data = raw_ppg_data[
                (raw_ppg_data["timestamp"] >= (start_time + cut_off_seconds))
                & (raw_ppg_data["timestamp"] <= (end_time - cut_off_seconds))
            ]
        else:
            cropped_eeg_data = raw_eeg_data.copy().crop(tmin=(start_time), tmax=(end_time))
            cropped_ecg_data = raw_ecg_data[
                (raw_ecg_data["timestamp"] >= (start_time)) & (raw_ecg_data["timestamp"] <= (end_time))
            ]
            cropped_ppg_data = raw_ppg_data[
                (raw_ppg_data["timestamp"] >= (start_time)) & (raw_ppg_data["timestamp"] <= (end_time))
            ]

        # ---------------------- 2b. Format data ----------------------
        print("********** Formatting data **********\n")
        # Set time to start at 0
        # EEG data already starts at 0
        print("Set time to start at 0...")
        cropped_ecg_data.loc[:, "timestamp"] = (
            cropped_ecg_data["timestamp"] - cropped_ecg_data["timestamp"].tolist()[0]
        )
        cropped_ppg_data.loc[:, "timestamp"] = (
            cropped_ppg_data["timestamp"] - cropped_ppg_data["timestamp"].tolist()[0]
        )

        # Adjust event time so first marker starts not at 0 but at - cut_off_seconds
        events_experiment["onset"] = events_experiment["onset"] - start_time - cut_off_seconds

        print("Round onset column to 3 decimal places (1 ms accuracy) for ECG and PPG data...")
        # Round onset column to 3 decimal places (1 ms accuracy)
        # To account for small differences in onset times between participants
        cropped_ecg_data.loc[:, "timestamp"] = cropped_ecg_data.loc[:, "timestamp"].round(3)
        cropped_ppg_data.loc[:, "timestamp"] = cropped_ppg_data.loc[:, "timestamp"].round(3)
        events_experiment["onset"] = events_experiment["onset"].round(3)

        # Reset index
        events_experiment = events_experiment.reset_index(drop=True)
        cropped_ecg_data = cropped_ecg_data.reset_index(drop=True)
        cropped_ppg_data = cropped_ppg_data.reset_index(drop=True)

        # Scale ECG and PPG data
        if scaling:
            print(f"Scaling ECG and PPG data by {scale_factor}...\n")
            cropped_ecg_data["cardiac"] = cropped_ecg_data["cardiac"] * scale_factor
            cropped_ppg_data["ppg"] = cropped_ppg_data["ppg"] * scale_factor

        # ---------------------- 2c. Preprocess ECG and PPG data ----------------------
        print("********** Preprocessing ECG and PPG **********\n")
        # Flip ECG signal (as it is inverted)
        print("Flipping ECG signal...")
        ecg_data_flipped = nk.ecg_invert(cropped_ecg_data["cardiac"], sampling_rate=sampling_rates["ecg"], force=True)[
            0
        ]

        # Data Cleaning using NeuroKit for ECG data
        # A 50 Hz powerline filter and
        # 4th-order Butterworth filters (0.5 Hz high-pass, 30 Hz low-pass)
        # are applied to the ECG signal.
        print("Cleaning ECG data...")
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
        print("Detecting R-peaks in ECG data...")
        # R-peaks detection using NeuroKit for ECG data
        r_peaks_ecg, info_ecg = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_rates["ecg"])

        # Data Cleaning using NeuroKit for PPG data
        # Uses the preprocessing pipeline "elgendi" and "templatematch" to asses quality of method
        # R-peaks detection using NeuroKit for PPG data
        print("Cleaning PPG data...")
        print("Detecting PPG-peaks in PPG data...\n")
        signals_ppg, info_ppg = nk.ppg_process(
            cropped_ppg_data["ppg"],
            sampling_rate=sampling_rates["ppg"],
            method="elgendi",
            method_quality="templatematch",
        )

        # Plot cleaned ECG data and R-peaks for the first 10s
        if show_plots:
            plot_peaks(
                cleaned_signal=cleaned_ecg,
                peaks=info_ecg["ECG_R_Peaks"],
                time_range=(0, 10),
                plot_title=f"Cleaned ECG signal with R-peaks for subject {subject} for the first 10 seconds",
                sampling_rate=sampling_rates["ecg"],
            )

        # Plot PPG data and PPG-peaks for the first 10s
        if show_plots:
            plot_peaks(
                cleaned_signal=signals_ppg["PPG_Clean"],
                peaks=info_ppg["PPG_Peaks"],
                time_range=(0, 10),
                plot_title=f"Cleaned PPG signal with PPG-peaks for subject {subject} for the first 10 seconds",
                sampling_rate=sampling_rates["ppg"],
            )

        # Perform manual cleaning of peaks if specified
        if manual_cleaning:
            print("* * * * * * Manual correction of peaks * * * * * *\n")
            # Manual correction of R-peaks
            # Save JSON file with corrected R-peaks and bad segments indices
            ecg_corr_fname = f"sub-{subject}_task-{exp_name}_rpeaks-corrected.json"
            ecg_corr_fpath = Path(subject_preprocessed_folder) / ecg_corr_fname

            # Transform array of R-peaks marked as 1s in a list of 0s to a boolean array
            r_peaks_ecg_boolean = r_peaks_ecg["ECG_R_Peaks"].astype(bool)

            # Display interactive plot
            # TODO: make this better by scaling it to 10 seconds for each window # noqa: FIX002
            # and then clicking through them
            # Also, how do I actually correct anything?!
            editor_ecg = Editor(
                signal=cleaned_ecg,
                corrected_json=ecg_corr_fpath,
                sfreq=sampling_rates["ecg"],
                corrected_peaks=r_peaks_ecg_boolean,
                signal_type="ECG",
                figsize=(15, 5),
            )

            display(editor_ecg.commands_box)

            # Manual correction of PPG-peaks
            # Save JSON file with corrected PPG-peaks and bad segments indices
            ppg_corr_fname = f"sub-{subject}_task-{exp_name}_ppg-peaks-corrected.json"
            ppg_corr_fpath = Path(subject_preprocessed_folder) / ppg_corr_fname

            # Transform array of PPG-peaks marked as 1s in a list of 0s to a boolean array
            ppg_peaks_boolean = signals_ppg["PPG_Peaks"].astype(bool)

            editor_ppg = Editor(
                signal=signals_ppg["PPG_Clean"],
                corrected_json=ppg_corr_fpath,
                sfreq=sampling_rates["ppg"],
                corrected_peaks=ppg_peaks_boolean,
                signal_type="PPG",
                figsize=(15, 5),
            )

            display(editor_ppg.commands_box)

        # Execute only when manual peak correction is done
        if manual_cleaning:
            print("Saving corrected R-peaks and PPG-peaks...")
            editor_ecg.save()
            editor_ppg.save()

        # Load corrected R-peaks and PPG-peaks if manual cleaning was done
        if manual_cleaning:
            print("Loading corrected R-peaks and PPG-peaks...")
            # Load corrected R-peaks
            with ecg_corr_fpath.open("r") as f:
                corrected_rpeaks = f.read()
            # Load corrected PPG-peaks
            with ppg_corr_fpath.open("r") as f:
                corrected_ppg_peaks = json.load(f)

        print("Calculating IBI and HR from ECG and PPG data...")
        # Calculate inter-beat-intervals (IBI) from peaks
        r_peaks_indices = corrected_rpeaks["ecg"]["corrected_peaks"] if manual_cleaning else info_ecg["ECG_R_Peaks"]
        ibi_ecg = nk.signal_period(peaks=r_peaks_indices, sampling_rate=sampling_rates["ecg"])

        ppg_peaks_indices = corrected_ppg_peaks["ppg"]["corrected_peaks"] if manual_cleaning else info_ppg["PPG_Peaks"]
        ibi_ppg = nk.signal_period(peaks=ppg_peaks_indices, sampling_rate=sampling_rates["ppg"])

        # Calculate heart rate (HR) from peaks
        heart_rate_ecg = nk.ecg_rate(peaks=r_peaks_indices, sampling_rate=sampling_rates["ecg"])
        heart_rate_ppg = nk.ppg_rate(peaks=ppg_peaks_indices, sampling_rate=sampling_rates["ppg"])

        # Plot IBI and HR for ECG and PPG data
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        axs[0, 0].plot(ibi_ecg, color=colors["ECG"][0])
        axs[0, 0].set_ylabel("IBI from ECG")
        axs[0, 1].plot(heart_rate_ecg, color=colors["ECG"][1])
        axs[0, 1].set_ylabel("HR from ECG")
        axs[1, 0].plot(ibi_ppg, color=colors["PPG"][0])
        axs[1, 0].set_ylabel("IBI from PPG")
        axs[1, 1].plot(heart_rate_ppg, color=colors["PPG"][1])
        axs[1, 1].set_ylabel("HR from PPG")
        fig.suptitle(f"IBI and HR from ECG and PPG data for subject {subject} "
        "(no manual cleaning)" if not manual_cleaning else "(after manual cleaning)", fontsize=16)
        # Set x-axis labels to minutes instead of seconds for all axes
        for ax in axs.flat:
            ax.set_xlabel("Time (s)")
            x_ticks = ax.get_xticks()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{x/sampling_rates['ecg']}" for x in x_ticks])

        # Save plot to results directory
        plt.savefig(subject_results_folder / f"sub-{subject}_task-{task}_IBI-HR.png")

        if show_plots:
            plt.show()

        plt.close()

        print("Saving preprocessed ECG and PPG data to tsv files...")

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

        # Attributes for file naming
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
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_ecg_preprocessed{attributes_cardiac}.tsv",
            sep="\t",
            index=False,
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
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_ppg_preprocessed{attributes_cardiac}.tsv",
            sep="\t",
            index=False,
        )

        print("Preprocessed ECG and PPG data saved to tsv files.\n")

        # ---------------------- 2d. Preprocess EEG data ----------------------
        print("********** Preprocessing EEG **********\n")
        # Set Montage
        print("Set Montage for EEG data...")
        # Set EEG channel layout for topo plots
        montage_filename = data_dir / exp_name / rawdata_name / "CACS-64_REF.bvef"
        if montage_filename.exists():
            montage = mne.channels.read_custom_montage(montage_filename)
            cropped_eeg_data.set_montage(montage)
        else:
            print("ERROR! No montage file found. Make sure to download the CACS-64_REF.bvef file from Brainvision "
            "(https://www.brainproducts.com/downloads/cap-montages/) and place it in the rawdata folder.")
            # Exit the program if no montage file is found
            sys.exit()

        # Interpolate the ECG data to match the EEG data
        if len(cleaned_ecg) < len(cropped_eeg_data.times):
            cleaned_ecg = np.interp(
                cropped_eeg_data.times, np.linspace(0, len(cleaned_ecg), len(cleaned_ecg)), cleaned_ecg
            )
        # Or crop the ECG data to match the EEG data
        elif len(cleaned_ecg) > len(cropped_eeg_data.times):
            cleaned_ecg = cleaned_ecg[: len(cropped_eeg_data.times)]
        # Or leave it as it is
        else:
            pass

        print("Add ECG data as a new channel to the EEG data...")
        # Add ECG data as a new channel to the EEG data
        ecg_data_channel = mne.io.RawArray([cleaned_ecg], mne.create_info(["ECG"], sampling_rates["ecg"], ["ecg"]))
        cropped_eeg_data.add_channels([ecg_data_channel])

        # Preprocessing EEG data using preprocessing_eeg function
        print("Preprocessing EEG data...")
        resampled_data, epochs, reject_log = preprocess_eeg(
            cropped_eeg_data, low_frequency, high_frequency, resampling_rate, autoreject=autoreject
        )

        # Plot reject_log
        fig, ax = plt.subplots(figsize=(15, 10))
        reject_log.plot(orientation="horizontal", show_names=1, aspect="auto", ax=ax, show=False)
        ax.set_title(f"Autoreject: Rejected epochs and channels for subject {subject}", fontsize=16)

        # Save plot to results directory
        fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_autoreject.png")

        if show_plots:
            plt.show()

        plt.close()

        # Artifact rejection with ICA using run_ica function
        print("Running ICA for artifact rejection...")
        ica = run_ica(epochs, reject_log.bad_epochs)

        # Plot results of ICA for the first 5s
        fig = ica.plot_overlay(resampled_data,
            picks="eeg", start=0, stop=5*resampling_rate,
            title=f"ICA overlay for subject {subject}",
            show=False)

        # Save plot to results directory
        fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ica_overlay.png")

        if show_plots:
            plt.show()

        plt.close()

        # Semi-automatic selection of ICA components using ica_correlation function
        print("Selecting ICA components semi-automatically...")
        ica, eog_indices, eog_scores, ecg_indices, ecg_scores = ica_correlation(ica, epochs)

        # Number of components removed
        print(f"Number of components removed: {len(ica.exclude)}")

        # Plot components
        fig, axs = plt.subplots(1, len(ica.exclude), figsize=[15, 5])
        for index, component in enumerate(ica.exclude):
            ica.plot_components(inst=epochs, picks=component,
                axes=axs[index], show_names=True, colorbar=True, show=False)
        fig.suptitle(f"EOG and ECG components to be excluded for subject {subject}", fontsize=16)

        # Save plot to results directory
        fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_rejected_components.png")

        if show_plots:
            plt.show()

        plt.close()

        # Plot correlation scores
        fig = ica.plot_scores(eog_scores, exclude=eog_indices,
            title=f"Correlation scores for EOG components for subject {subject}", show=False)
        # Save plot to results directory
        fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_eog_correlation_scores.png")

        if show_plots:
            plt.show()

        plt.close()

        fig = ica.plot_scores(ecg_scores, exclude=ecg_indices,
            title=f"Correlation scores for ECG components for subject {subject}", show=False)
        # Save plot to results directory
        fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ecg_correlation_scores.png")

        if show_plots:
            plt.show()

        plt.close()

        # Get the explained variance of the ICA components
        explained_variance_ratio = ica.get_explained_variance_ratio(epochs)["eeg"]
        print(f"Explained variance ratio of ICA components: {explained_variance_ratio}")

        # Reject components in the resampled data that are not brain related
        print("Rejecting components in the resampled data that are not brain related...")
        eeg_clean = ica.apply(resampled_data.copy())

        if resample and autoreject:
            attributes_eeg = f"resampled_{resampling_rate}_autoreject_filtered_{low_frequency}-{high_frequency}"
        elif resample and not autoreject:
            attributes_eeg = f"resampled_{resampling_rate}_filtered_{low_frequency}-{high_frequency}"
        elif not resample and autoreject:
            attributes_eeg = f"autoreject_filtered_{low_frequency}-{high_frequency}"
        else:
            attributes_eeg = f"filtered_{low_frequency}-{high_frequency}"

        print("Saving preprocessed EEG data to fif files...")

        # Save the raw data before ICA
        resampled_data.save(
            subject_preprocessed_folder
            / f"sub-{subject}_task-{task}_eeg_preprocessed_{attributes_eeg}_before_ica.fif",
            overwrite=True,
        )

        # Save the clean data after ICA
        eeg_clean.save(
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_eeg_preprocessed_{attributes_eeg}_after_ica.fif",
            overwrite=True,
        )

        # Save the ICA object with the bad components
        ica.save(
            subject_preprocessed_folder / f"sub-{subject}_task-{task}_eeg_{attributes_eeg}_ica.fif", overwrite=True
        )

        print("Preprocessed EEG data saved to fif files.\n")

    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS

    # TODO: Implement averaging over all participants  # noqa: FIX002

# %%
