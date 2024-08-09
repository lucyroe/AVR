"""
Script to preprocess physiological data (EEG, ECG, PPG) for AVR phase 3.

Required packages: mne, neurokit, systole, seaborn, autoreject

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 6 July 2024
Last update: 8 August 2024
"""
#%%
def preprocess_physiological(subjects=["001"],  # noqa: PLR0915, B006, C901, PLR0912, PLR0913
            data_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
            results_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
            show_plots=True,
            debug=False,
            manual_cleaning=True):
    """
    Preprocess physiological data for AVR phase 3.

    Inputs:     Raw EEG data in .edf files
                ECG and PPG data in tsv.gz files
                Event markers in tsv files
                Event mapping in tsv files
                BrainVision actiCap montage file ("CACS-64_REF.bvef")

    Outputs:    Preprocessed data (EEG, ECG, PPG) in tsv files (ECG & PPG) and fif files (EEG, before and after ICA)
                Participant metadata in json files
                Excluded participants (with too many bad channels) in a list
                Plots of preprocessing steps
                Data of all participants concatenated in tsv files (ECG & PPG) and fif files (EEG, after ICA)
                Averaged data over all participants in tsv files (ECG & PPG)

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
        3a. Concatenate all ECG & PPG data
        3b. Average over all participants (ECG & PPG) & save to tsv files

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

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    task = "AVR"

    # Only analyze one subject when debug mode is on
    if debug:
        subjects = [subjects[0]]

    # Defne preprocessing steps to perform
    steps = ["Cutting", "Formatting", "Preprocessing ECG + PPG", "Preprocessing EEG", "Averaging"]  # Adjust as needed

    # Define whether scaling of the ECG and PPG data should be done
    scaling = True
    if scaling:
        scale_factor = 0.01

    # Define power line frequency
    powerline = 50  # in Hz

    # Define cutoff frequencies for bandfiltering EEG data
    low_frequency = 0.1  # in Hz
    high_frequency = 45  # in Hz

    low_frequency_ica = 1  # in Hz

    # Define the percentage of bad epochs that should be tolerated
    bad_epochs_threshold = 30  # in percent

    # Define the artifact threshold for epochs
    artifact_threshold = 100  # in microvolts

    # Define if the first and last 2.5 seconds of the data should be cut off
    # To avoid any potential artifacts at the beginning and end of the experiment
    cut_off_seconds = 2.5

    # Specify the data path info (in BIDS format)
    # Change with the directory of data storage
    data_dir = Path(data_dir) / "phase3"
    exp_name = "AVR"
    rawdata_name = "rawdata"  # rawdata folder
    derivative_name = "derivatives"  # derivates folder
    preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
    averaged_name = "avg"  # averaged data folder (inside preprocessed)
    datatype_name = "eeg"  # data type specification
    results_dir = Path(results_dir) / "phase3"

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
    colors = {
        "ECG": ["#F0E442", "#D55E00"],  # yellow and dark orange
        "PPG": ["#E69F00", "#CC79A7"],  # light orange and pink
        "EEG": ["#56B4E9", "#0072B2", "#009E73"],  # light blue, dark blue, and green
        "others": ["#FFFFFF", "#6C6C6C", "#000000"],  # white, gray, and black
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
        axs.scatter(
            selected_peaks,
            selected_signal[selected_peaks - min_sample],
            color=circlecolor,
            edgecolor=circlecolor,
            linewidth=1,
            alpha=0.6,
        )
        axs.set_ylabel("ECG" if "ECG" in plot_title else "PPG")
        axs.set_xlabel("Time (s)")
        x_ticks = axs.get_xticks()
        axs.set_xticks(x_ticks)
        # Transform x-axis to seconds
        axs.set_xticklabels([f"{x/sampling_rate}" for x in x_ticks])
        if plot_title:
            axs.set_title(plot_title)

        sns.despine()
        if show_plots:
            plt.show()
        
        plt.close()


    def preprocess_eeg(  # noqa: PLR0915
        raw_data: mne.io.Raw, low_frequency: float, high_frequency: int, autoreject: bool
    ):
        """
        Preprocess EEG data using the MNE toolbox.

        Remove Line-noise of power line, rereference to average, filter the data with a bandpass filter,
        segment it into epochs of 10s, and use Autoreject to detect bad channels and epochs.

        Arguments:
        ---------
        raw_data: mne.io.Raw
            The raw EEG data to be preprocessed. This should be an instance of mne.io.Raw, which contains the EEG
            signal data along with additional information about the recording.
        low_frequency: float
            Low cut-off frequency in Hz for the bandpass filter.
        high_frequency: int
            High cut-off frequency in Hz for the bandpass filter.
        autoreject: bool
            If True, autoreject is used to detect and interpolate bad channels and epochs automatically.

        Returns:
        -------
        filtered_data: mne.io.Raw
            The filtered raw data after preprocessing.
        epochs: mne.epochs.Epochs
            The segmented epochs extracted from the filtered data.
        reject_log: autoreject.autoreject.RejectLog (if autoreject=True)
            A log of the rejected epochs and channels.
        """
        # Line-noise removal
        print("Removing line noise...")
        raw_data_50 = raw_data.copy().notch_filter(freqs=powerline, method="spectrum_fit")

        # Rereference to average
        print("Rereferencing to average...")
        rereferenced_data = raw_data_50.copy().set_eeg_reference("average", projection=True)

        # Filtering
        print("Filtering data...")
        filtered_data = rereferenced_data.copy().filter(l_freq=low_frequency, h_freq=high_frequency)

        # Segment data into epochs of 10s
        # Even though data is continuous, it is good practice to break it into epochs
        # before detecting bad channels and running ICA
        print("Segmenting data into epochs...")
        tstep = 10  # in seconds
        events = mne.make_fixed_length_events(filtered_data, duration=tstep)
        epochs = mne.Epochs(filtered_data, events, tmin=0, tmax=tstep, baseline=None, preload=True)

        if autoreject:
            # Pick only EEG channels for Autoreject bad channel detection
            picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

            # Use Autoreject to detect bad channels
            # Autoreject interpolates bad channels, takes quite long and identifies a lot of epochs
            # Here we do not remove any data, we only identify bad channels and epochs and store them in a log
            # Define random state to ensure reproducibility
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

            # Get the number of bad epochs and channels
            n_bad_epochs = np.sum(reject_log.bad_epochs)
            n_bad_channels = np.sum(reject_log.labels == 1)

            # Get the number of interpolated channels
            n_interpolated_channels = np.sum(reject_log.labels == 2)  # noqa: PLR2004

            # Get the number of all epochs and channels
            n_epochs = len(epochs)
            n_channels = len(reject_log.ch_names) * n_epochs

            # Get the percentage of bad epochs and channels
            perc_bad_epochs = (n_bad_epochs / n_epochs) * 100
            perc_bad_channels = (n_bad_channels / n_channels) * 100

            # Get the percentage of interpolated channels * epochs
            perc_interpolated_channels = (n_interpolated_channels / n_channels) * 100

            print(f"Number of bad epochs: {n_bad_epochs} of {n_epochs} ({perc_bad_epochs:.2f}%).")
            print(
                f"Number of bad channels across all epochs: {n_bad_channels} of {n_channels}"
                f"({perc_bad_channels:.2f}%)."
            )
            print(
                "Number of interpolated channels across all epochs: "
                f"{n_interpolated_channels} of {n_channels} ({perc_interpolated_channels:.2f}%)."
            )
            if perc_bad_epochs > bad_epochs_threshold or perc_bad_channels > bad_epochs_threshold:
                print(
                    f"WARNING! More than {bad_epochs_threshold}% of epochs or channels are bad."
                    "Excluding the participant is recommended."
                )

            answer = input("Do you want to exclude this participant? (Y/n): ")
            if answer == "Y":
                exclude = True
                print("Participant will be added to the list of excluded participants.")
            else:
                exclude = False
                print("Participant will not be excluded.")

            print("Continuing with the preprocessing...")

        return filtered_data, epochs, (reject_log if autoreject else None), (exclude if autoreject else None)


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
        ica_n_components: float
        n_components: int
        """
        # Set ICA parameters
        random_state = 42  # ensures ICA is reproducible each time it's run
        ica_n_components = 0.99  # Specify n_components as a decimal to set % explained variance

        # Fit ICA
        print("Fitting ICA...")
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
        ica.fit(epochs[~rejected_epochs], decim=3)  # decim reduces the number of time points to speed up computation
        print("Done with ICA.")

        # Get the number of components
        n_components = ica.n_components_

        print(f"{n_components} components were found to explain {ica_n_components * 100}% of the variance.")

        return ica, ica_n_components, n_components


    def ica_correlation(ica: mne.preprocessing.ICA, epochs: mne.epochs.Epochs):
        """
        Select ICA components semi-automatically using a correlation approach.

        Arguments:
        ---------
        ica: mne.preprocessing.ICA
            The ICA object containing the components to be examined. This should be the output from run_ica().
        epochs: mne.epochs.Epochs
            The epochs data (1 Hz) used for correlating with the ICA components.
            This should be the output from preprocess_eeg().

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
        emg_indices: list
            Indices of the ICA components identified as related to muscle activity.
        emg_scores: list
            Correlation scores for the muscle activity components.
        """
        # Create list of components to exclude
        ica.exclude = []

        # Correlate with EOG channels
        print("Correlating ICs with EOG channels...")
        eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=eog_channels, threshold="auto", measure="zscore")
        print("Number of EOG components identified: " + str(len(eog_indices)))

        # Correlate with ECG channels
        print("Correlating ICs with ECG channels...")
        ecg_indices, ecg_scores = ica.find_bads_ecg(
            epochs, ch_name="ECG", method="correlation", measure="zscore", threshold="auto"
        )
        print("Number of ECG components identified: " + str(len(ecg_indices)))

        # Correlate with muscle ativity
        print("Correlating ICs with muscle activity...")
        emg_indices, emg_scores = ica.find_bads_muscle(epochs, threshold=0.8)
        print("Number of muscle components identified: " + str(len(emg_indices)))

        # Assign the bad EOG components to the ICA.exclude attribute so they can be removed later
        ica.exclude = eog_indices + ecg_indices + emg_indices
        print("Correlation done.")

        return ica, eog_indices, eog_scores, ecg_indices, ecg_scores, emg_indices, emg_scores


    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. LOAD DATA
    # Initialize list to store data of all participants
    list_data_all = {"ECG": [], "PPG": [], "EEG": []}

    # Initialize list to store excluded participants
    excluded_participants = []

    # Loop over all subjects
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
                "ERROR! No event mapping file found. Using default event markers, which means that the different"
                " videos cannot be separated by their names."
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

        # Create participant file with relevant preprocessing metadata
        participant_metadata = {
            "subject": subject,
            "start_time": time.ctime(),
            "steps": steps,
            "manual_cleaning_of_peaks": manual_cleaning,
            "sampling_rate_ecg": sampling_rates["ecg"],
            "sampling_rate_ppg": sampling_rates["ppg"],
            "sampling_rate_eeg": sampling_rates["eeg"],
            "power_line": powerline,
            "low_frequency": low_frequency,
            "high_frequency": high_frequency,
            "low_frequency_ica": low_frequency_ica,
            "scaling": scaling,
            "scale_factor": scale_factor,
            "cut_off_seconds": cut_off_seconds,
            "bad_epochs_threshold": bad_epochs_threshold,
        }

        # Save the metadata of the participant
        with (subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_preprocessing_metadata.json").open(
            "w"
        ) as f:
            json.dump(participant_metadata, f)

        # %% STEP 2. PREPROCESS DATA
        if "Cutting" in steps:
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

        if "Formatting" in steps:
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

            # Reset index
            events_experiment = events_experiment.reset_index(drop=True)
            cropped_ecg_data = cropped_ecg_data.reset_index(drop=True)
            cropped_ppg_data = cropped_ppg_data.reset_index(drop=True)

            # Scale ECG and PPG data
            if scaling:
                print(f"Scaling ECG and PPG data by {scale_factor}...\n")
                cropped_ecg_data["cardiac"] = cropped_ecg_data["cardiac"] * scale_factor
                cropped_ppg_data["ppg"] = cropped_ppg_data["ppg"] * scale_factor

        # %% ---------------------- 2c. Preprocess ECG and PPG data ----------------------
        if "Preprocessing ECG + PPG" in steps:
            print("********** Preprocessing ECG and PPG **********\n")
            # Flip ECG signal (as it is inverted)
            print("Flipping ECG signal...")
            ecg_data_flipped = nk.ecg_invert(
                cropped_ecg_data["cardiac"], sampling_rate=sampling_rates["ecg"], force=True
            )[0]

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
                    corrected_rpeaks = json.load(f)
                # Load corrected PPG-peaks
                with ppg_corr_fpath.open("r") as f:
                    corrected_ppg_peaks = json.load(f)

            print("Calculating IBI and HR from ECG and PPG data...")
            # Calculate inter-beat-intervals (IBI) from peaks
            r_peaks_indices = (
                corrected_rpeaks["ecg"]["corrected_peaks"] if manual_cleaning else info_ecg["ECG_R_Peaks"]
            )
            ibi_ecg = nk.signal_period(peaks=r_peaks_indices, sampling_rate=sampling_rates["ecg"])

            ppg_peaks_indices = (
                corrected_ppg_peaks["ppg"]["corrected_peaks"] if manual_cleaning else info_ppg["PPG_Peaks"]
            )
            ibi_ppg = nk.signal_period(peaks=ppg_peaks_indices, sampling_rate=sampling_rates["ppg"])

            # Calculate heart rate (HR) from peaks
            heart_rate_ecg = nk.ecg_rate(peaks=r_peaks_indices, sampling_rate=sampling_rates["ecg"])
            heart_rate_ppg = nk.ppg_rate(peaks=ppg_peaks_indices, sampling_rate=sampling_rates["ppg"])

            # Interpolate IBI and HR to match the length of the ECG and PPG data
            desired_length = round(len(cleaned_ecg) / sampling_rates["ecg"])
            x_new = np.linspace(0, desired_length - 1, num=desired_length, endpoint=False)
            x_values_ibi_ecg = np.linspace(0, len(ibi_ecg) - 1, num=len(ibi_ecg), endpoint=False)
            ibi_ecg = nk.signal_interpolate(x_values_ibi_ecg, ibi_ecg, x_new, method="linear")
            x_values_hr_ecg = np.linspace(0, len(heart_rate_ecg) - 1, num=len(heart_rate_ecg), endpoint=False)
            heart_rate_ecg = nk.signal_interpolate(x_values_hr_ecg, heart_rate_ecg, x_new, method="linear")
            x_values_ibi_ppg = np.linspace(0, len(ibi_ppg) - 1, num=len(ibi_ppg), endpoint=False)
            ibi_ppg = nk.signal_interpolate(x_values_ibi_ppg, ibi_ppg, x_new, method="linear")
            x_values_hr_ppg = np.linspace(0, len(heart_rate_ppg) - 1, num=len(heart_rate_ppg), endpoint=False)
            heart_rate_ppg = nk.signal_interpolate(x_values_hr_ppg, heart_rate_ppg, x_new, method="linear")

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
            fig.suptitle(
                f"IBI and HR from ECG and PPG data for subject {subject} "
                "(no manual cleaning)"
                if not manual_cleaning
                else "(after manual cleaning)",
                fontsize=16,
            )
            # Set x-axis labels to minutes instead of seconds for all axes
            for ax in axs.flat:
                ax.set_xlabel("Time (min)")
                x_ticks = ax.get_xticks()
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
                # Add vertical lines for event markers
                # Exclude first and last event markers
                # And only use every second event marker to avoid overlap
                for _, row in events_experiment.iloc[0:-1:2].iterrows():
                    ax.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)

            # Save plot to results directory
            plt.savefig(subject_results_folder / f"sub-{subject}_task-{task}_IBI-HR.png")

            if show_plots:
                plt.show()

            plt.close()

            print("Saving preprocessed ECG and PPG data to tsv files...")

            # Append R-peaks, IBI, and HR so that they have the same length as the ECG data
            r_peaks_indices_with_nans = list(r_peaks_indices) + [np.nan] * (len(cleaned_ecg) - len(r_peaks_indices))
            ibi_ecg_with_nans = list(ibi_ecg) + [np.nan] * (len(cleaned_ecg) - len(ibi_ecg))
            heart_rate_ecg_with_nans = list(heart_rate_ecg) + [np.nan] * (len(cleaned_ecg) - len(heart_rate_ecg))

            # Create dataframe with cleaned ECG data, R-peaks, IBI, and HR
            ecg_data_df = pd.DataFrame({"ECG": cleaned_ecg})
            ecg_data_df["R-peaks"] = pd.Series(r_peaks_indices_with_nans)
            ecg_data_df["IBI"] = pd.Series(ibi_ecg_with_nans)
            ecg_data_df["HR"] = pd.Series(heart_rate_ecg_with_nans)
            # Create array with subject id that has the same length as the other series
            subject_array = [subject] * len(cleaned_ecg)
            ecg_data_df["subject"] = pd.Series(subject_array)
            ecg_data_df["timestamp"] = pd.Series(cropped_ecg_data["timestamp"])
            # Make the subject column the first column and the timestamp the second column
            ecg_data_df = ecg_data_df[["subject", "timestamp", "ECG", "R-peaks", "IBI", "HR"]]

            # Attributes for file naming
            if scaling and manual_cleaning:
                attributes_cardiac = "_scaled_manually-cleaned"
            elif scaling and not manual_cleaning:
                attributes_cardiac = "_scaled"
            elif not scaling and manual_cleaning:
                attributes_cardiac = "_manually-cleaned"
            else:
                attributes_cardiac = ""

            # Save ECG data to tsv file
            ecg_data_df.to_csv(
                subject_preprocessed_folder
                / f"sub-{subject}_task-{task}_physio_ecg_preprocessed{attributes_cardiac}.tsv",
                sep="\t",
                index=False,
            )

            # Add ECG data to list_data_all
            list_data_all["ECG"].append(ecg_data_df)

            # Append PPG-peaks, IBI, and HR so that they have the same length as the PPG data
            ppg_peaks_indices_with_nans = list(ppg_peaks_indices) + [np.nan] * (
                len(signals_ppg["PPG_Clean"]) - len(ppg_peaks_indices)
            )
            ibi_ppg_with_nans = list(ibi_ppg) + [np.nan] * (len(signals_ppg["PPG_Clean"]) - len(ibi_ppg))
            heart_rate_ppg_with_nans = list(heart_rate_ppg) + [np.nan] * (
                len(signals_ppg["PPG_Clean"]) - len(heart_rate_ppg)
            )

            # Create dataframe with cleaned PPG data, PPG-peaks, IBI, and HR
            ppg_data_df = pd.DataFrame({"PPG": signals_ppg["PPG_Clean"]})
            ppg_data_df["PPG-peaks"] = pd.Series(ppg_peaks_indices_with_nans)
            ppg_data_df["IBI"] = pd.Series(ibi_ppg_with_nans)
            ppg_data_df["HR"] = pd.Series(heart_rate_ppg_with_nans)
            # Create array with subject ID that has the same length as the other series
            subject_array = [subject] * len(signals_ppg["PPG_Clean"])
            ppg_data_df["subject"] = pd.Series(subject_array)
            ppg_data_df["timestamp"] = pd.Series(cropped_ppg_data["timestamp"])
            # Make the subject column the first column and the timestamp the second column
            ppg_data_df = ppg_data_df[["subject", "timestamp", "PPG", "PPG-peaks", "IBI", "HR"]]

            # Save PPG data to tsv file
            ppg_data_df.to_csv(
                subject_preprocessed_folder
                / f"sub-{subject}_task-{task}_physio_ppg_preprocessed{attributes_cardiac}.tsv",
                sep="\t",
                index=False,
            )

            # Add PPG data to list_data_all
            list_data_all["PPG"].append(ppg_data_df)

            print("Preprocessed ECG and PPG data saved to tsv files.\n")

        # %% ---------------------- 2d. Preprocess EEG data ----------------------
        if "Preprocessing EEG" in steps:
            print("********** Preprocessing EEG **********\n")
            # Set Montage
            print("Set Montage for EEG data...")
            # Set EEG channel layout for topo plots
            montage_filename = data_dir / exp_name / rawdata_name / "CACS-64_REF.bvef"
            if montage_filename.exists():
                montage = mne.channels.read_custom_montage(montage_filename)
                cropped_eeg_data.set_montage(montage)
            else:
                print(
                    "ERROR! No montage file found. Make sure to download the CACS-64_REF.bvef file from Brainvision "
                    "(https://www.brainproducts.com/downloads/cap-montages/) and place it in the rawdata folder."
                )
                # Exit the program if no montage file is found
                sys.exit()

            # Add NaNs to the ECG data to match the EEG data
            if len(cleaned_ecg) < len(cropped_eeg_data.times):
                cleaned_ecg = np.append(cleaned_ecg, [np.nan] * (len(cropped_eeg_data.times) - len(cleaned_ecg)))
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

            # Preprocessing EEG data using preprocess_eeg function (high-pass filter of 0.1 Hz)
            print("Preprocessing EEG data...")
            filtered_data, epochs, _, _ = preprocess_eeg(
                cropped_eeg_data, low_frequency, high_frequency, autoreject=False
            )

            # Preprocessing EEG data using preprocess_eeg function for ICA (high-pass filter of 1 Hz)
            filtered_data_ica, epochs_ica, reject_log_ica, exclude = preprocess_eeg(
                cropped_eeg_data, low_frequency_ica, high_frequency, autoreject=True
            )

            if exclude:
                excluded_participants.append(subject)

            # Plot reject_log
            fig, ax = plt.subplots(figsize=(15, 10))
            reject_log_ica.plot(orientation="horizontal", show_names=1, aspect="auto", ax=ax, show=False)
            ax.set_title(f"Autoreject: Bad epochs and channels for subject {subject}", fontsize=16)

            # Save plot to results directory
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_autoreject.png")

            if show_plots:
                plt.show()

            plt.close()

            # Artifact rejection with ICA using run_ica function
            print("Running ICA for artifact rejection...")
            ica_all, ica_variance, ica_n_components = run_ica(epochs_ica, reject_log_ica.bad_epochs)

            # Semi-automatic selection of ICA components using ica_correlation function
            # Correlates ICA components with EOG, ECG and muscle data
            print("Selecting ICA components semi-automatically...")
            ica, eog_indices, eog_scores, ecg_indices, ecg_scores, emg_indices, emg_scores = ica_correlation(
                ica_all, epochs_ica
            )

            # Get remaining components
            list_all_components = list(range(ica_all.n_components_))
            other_components = set(list_all_components)
            for sublist in [eog_indices, ecg_indices, emg_indices]:
                other_components -= set(sublist)
            list_remaining_components = list(other_components)

            # Number of components identified automatically
            print(f"Number of components identified automatically: {len(ica.exclude)} (out of {ica.n_components_}).")
            print(
                f"Components identified automatically: {ica.exclude} ({len(eog_indices)} EOG components, "
                f"{len(ecg_indices)} ECG components, {len(emg_indices)} EMG components)."
            )

            # Plot components (with properties)
            # EOG, ECG, EMG and others in separate plots
            if eog_indices != []:
                # Combine all EOG plots into one figure
                fig, axs = plt.subplots(len(eog_indices), 5, figsize=(30, len(eog_indices) * 4))
                for index, component in enumerate(eog_indices):
                    ica.plot_properties(filtered_data_ica, picks=component, show=False, axes=axs[index])
                fig.suptitle(f"EOG components for subject {subject}", fontsize=16)
                # Save plot to results directory
                fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_eog_components.png")
                if show_plots:
                    plt.show()
                plt.close()

            if ecg_indices != []:
                # Combine all ECG plots into one figure
                if len(ecg_indices) == 1:
                    fig, axs = plt.subplots(1, 5, figsize=(30, 4))
                    ica.plot_properties(filtered_data_ica, picks=ecg_indices, show=False, axes=axs)
                else:
                    fig, axs = plt.subplots(len(ecg_indices), 5, figsize=(30, len(ecg_indices) * 4))
                    for index, component in enumerate(ecg_indices):
                        ica.plot_properties(filtered_data_ica, picks=component, show=False, axes=axs[index])
                fig.suptitle(f"ECG components for subject {subject}", fontsize=16)
                # Save plot to results directory
                fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ecg_components.png")
                if show_plots:
                    plt.show()
                plt.close()

            if emg_indices != []:
                # Combine all EMG plots into one figure
                fig, axs = plt.subplots(len(emg_indices), 5, figsize=(30, len(emg_indices) * 4))
                for index, component in enumerate(emg_indices):
                    ica.plot_properties(filtered_data_ica, picks=component, show=False, axes=axs[index])
                fig.suptitle(f"EMG components for subject {subject}", fontsize=16)
                # Save plot to results directory
                fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_emg_components.png")
                if show_plots:
                    plt.show()
                plt.close()


            # Plot remaining ICA components
            fig, axs = plt.subplots(len(list_remaining_components), 5,
                figsize=(30, (len(list_remaining_components) * 4)))
            for index, component in enumerate(list_remaining_components):
                ica.plot_properties(filtered_data_ica, picks=component, show=False, axes=axs[index])
            fig.suptitle(f"Remaining ICA components for subject {subject}", fontsize=16)
            # Save plot to results directory
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_remaining_components.png")
            if show_plots:
                plt.show()
            plt.close()

            # Manual rejection of components
            answer = input("Do you want to remove any of the automatically identified components? (Y/n): ")
            if answer == "Y":
                print("EOG components identified: " + str(eog_indices))
                remove = input("Do you want to remove any of the EOG components? (Y/n): ")
                if remove == "Y":
                    manual_rejection_eog = input(
                        "Enter the indices of the EOG components you want to remove (separated by commas): "
                    )
                    manual_rejection_eog = [int(i) for i in manual_rejection_eog.split(",")]
                    eog_indices = set(eog_indices)
                    for sublist in manual_rejection_eog:
                        eog_indices -= {sublist}
                    eog_indices = list(eog_indices)
                    print(f"EOG components {manual_rejection_eog} removed from list of components.")

                print("ECG components identified: " + str(ecg_indices))
                remove = input("Do you want to remove any of the ECG components? (Y/n): ")
                if remove == "Y":
                    manual_rejection_ecg = input(
                        "Enter the indices of the ECG components you want to remove (separated by commas): "
                    )
                    manual_rejection_ecg = [int(i) for i in manual_rejection_ecg.split(",")]
                    ecg_indices = set(ecg_indices)
                    for sublist in manual_rejection_ecg:
                        ecg_indices -= {sublist}
                    ecg_indices = list(ecg_indices)
                    print(f"ECG components {manual_rejection_ecg} removed from list of components.")

                print("EMG components identified: " + str(emg_indices))
                remove = input("Do you want to remove any of the EMG components? (Y/n): ")
                if remove == "Y":
                    manual_rejection_emg = input(
                        "Enter the indices of the EMG components you want to remove (separated by commas): "
                    )
                    manual_rejection_emg = [int(i) for i in manual_rejection_emg.split(",")]
                    emg_indices = set(emg_indices)
                    for sublist in manual_rejection_emg:
                        emg_indices -= {sublist}
                    emg_indices = list(emg_indices)
                    print(f"EMG components {manual_rejection_emg} removed from list of components.")

            print("Remaining components not identified by the correlation approach: " + str(list_remaining_components))
            remove = input("Do you want to add any of the remaining components to the list of rejected components? (Y/n): ")
            if remove == "Y":
                manual_rejection_other = input(
                    "Enter the indices of the remaining components you want to add to the list of rejected components (separated by commas): "
                    )
                manual_rejection_other = [int(i) for i in manual_rejection_other.split(",")]
                print(f"Remaining components {manual_rejection_other} added to the list of rejected components.")

            ica.exclude = eog_indices + ecg_indices + emg_indices + manual_rejection_other
            print(
                f"Number of components identified after manual removal of invalid components: {len(ica.exclude)} "
                f"(out of {ica.n_components_})."
                )
            print(
                f"Components identified after manual removal of invalid components: {ica.exclude} "
                f"({len(eog_indices)} EOG components, {len(ecg_indices)} ECG components, {len(emg_indices)}"
                f" EMG components, {len(manual_rejection_other)} other components)."
                )

            # Finally reject components in the filtered data (0.1 Hz) that are not brain related
            print("Rejecting components in the filtered data (0.1 Hz) that are not brain related...")
            eeg_clean = ica.apply(filtered_data.copy())

            # Get the explained variance of the ICA components
            explained_variance_ratio = ica.get_explained_variance_ratio(epochs_ica, components=ica.exclude)["eeg"]
            print(f"Explained variance ratio of excluded ICA components: {explained_variance_ratio}")


            # Plot results of ICA for the first 2.5s
            fig = ica.plot_overlay(
                eeg_clean,
                picks="eeg",
                title=f"ICA overlay for subject {subject} (red: original, black: cleaned)",
                show=False,
            )

            # Save plot to results directory
            fig.savefig(subject_results_folder / f"sub-{subject}_task-{task}_ica_overlay.png")

            if show_plots:
                plt.show()

            plt.close()

            # Final Check: Segment cleaned data into epochs again and check in how many epochs
            # there are still artifacts of more than 100 µV
            print(
                f"Checking for any remaining bad epochs (max. value above {artifact_threshold} µV) "
                "in the cleaned data..."
            )
            tstep = 10  # in seconds
            events = mne.make_fixed_length_events(eeg_clean, duration=tstep)
            epochs = mne.Epochs(eeg_clean, events, tmin=0, tmax=tstep, baseline=None, preload=True)
            # Pick only EEG channels
            epochs.pick_types(eeg=True, eog=False, ecg=False)

            # Check for bad epochs
            remaining_bad_epochs = []
            for i, epoch in enumerate(epochs):
                if np.max(epoch.data) > artifact_threshold:
                    remaining_bad_epochs.append(i)

            # Calculate the percentage of bad epochs
            percentage_bad_epochs = len(remaining_bad_epochs) / len(epochs) * 100

            # Print the number of remaining bad epochs
            print(f"Number of remaining bad epochs: {len(remaining_bad_epochs)} ({percentage_bad_epochs:.2f}%).")

            # Check if the percentage of bad epochs is above the threshold
            if percentage_bad_epochs > bad_epochs_threshold:
                print("The percentage of bad epochs is above the threshold. Participant should be excluded.")

            answer = input("Do you want to exclude this participant? (Y/n): ")
            if answer == "Y":
                excluded_participants.append(subject)
                print("Participant added to the list of excluded participants.")
            else:
                print("Participant not excluded.")

            attributes_eeg = f"filtered_{low_frequency}-{high_frequency}"

            print("Saving preprocessed EEG data to fif files...")

            # Save the raw data before ICA
            filtered_data.save(
                subject_preprocessed_folder
                / f"sub-{subject}_task-{task}_eeg_preprocessed_{attributes_eeg}_before_ica.fif",
                overwrite=True,
            )

            # Save the clean data after ICA
            eeg_clean.save(
                subject_preprocessed_folder
                / f"sub-{subject}_task-{task}_eeg_preprocessed_{attributes_eeg}_after_ica.fif",
                overwrite=True,
            )

            # Add EEG data to list_data_all
            list_data_all["EEG"].append(eeg_clean)

            # Save the ICA object with the bad components
            ica.save(
                subject_preprocessed_folder / f"sub-{subject}_task-{task}_eeg_{attributes_eeg}_ica.fif", overwrite=True
            )

            print("Preprocessed EEG data saved to fif files.\n")

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

            # Read in the participant metadata json file
            with (subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_preprocessing_metadata.json").open(
                "r"
            ) as f:
                participant_metadata = json.load(f)

            # Append more information to the participant metadata json file
            participant_metadata["number_of_bad_epochs"] = int(np.sum(reject_log_ica.bad_epochs))
            participant_metadata["number_of_all_epochs"] = int(len(epochs_ica))
            participant_metadata["percentage_bad_epochs"] = float(int(np.sum(reject_log_ica.bad_epochs)) / len(epochs_ica))
            participant_metadata["number_of_bad_channels_x_epochs"] = int(np.sum(reject_log_ica.labels == 1))
            participant_metadata["number_of_interpolated_channels_x_epochs"] = int(np.sum(reject_log_ica.labels == 2))  # noqa: PLR2004
            participant_metadata["number_of_all_channels_x_epochs"] = int(
                (len(epochs_ica.info["ch_names"])-len(eog_channels)-1) * len(epochs_ica)
            )
            participant_metadata["percentage_interpolated_channels_x_epochs"] = float(
                int(np.sum(reject_log_ica.labels == 2)) /
                (int(len(epochs_ica.info["ch_names"])-len(eog_channels)-1) * len(epochs_ica)))
            participant_metadata["participant_excluded"] = exclude
            participant_metadata["number_of_all_components"] = int(ica_n_components)
            participant_metadata["number_of_removed_components"] = int(len(ica.exclude))
            participant_metadata["ica_variance"] = float(ica_variance)
            participant_metadata["eog_components"] = [int(index) for index in eog_indices]
            participant_metadata["ecg_components"] = [int(index) for index in ecg_indices]
            participant_metadata["emg_components"] = [int(index) for index in emg_indices]
            participant_metadata["other_components"] = [int(index) for index in list_remaining_components]
            participant_metadata["explained_variance_ratio"] = float(explained_variance_ratio)

            # Include the end time of the preprocessing
            participant_metadata["end_time"] = time.ctime()

            # Save the updated participant metadata json file
            with (subject_preprocessed_folder / f"sub-{subject}_task-{task}_physio_preprocessing_metadata.json").open(
                "w"
            ) as f:
                json.dump(participant_metadata, f)

    # Save event markers to tsv file
    events_experiment.to_csv(
        data_dir / exp_name / derivative_name / preprocessed_name / "events_experiment.tsv", sep="\t",
        index=False
        )

    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS
    if "Averaging" in steps:
        print("********** Averaging over all participants **********\n")
        # ---------------------- 3a. Interpolate data ----------------------
        print("Interpolating data to match the length of the first subject's data...")

        # Create new empty lists to store the interpolated data
        list_data_all_interpolated = {"ECG": [], "PPG": []}

        # Interpolate ECG data
        for i, ecg_data in enumerate(list_data_all["ECG"]):
            # Interpolate ECG data to match the length of the first subject's data
            if i != 0:
                interpolated_ecg = nk.signal_interpolate(
                    np.linspace(0, len(ecg_data["ECG"]) - 1, num=len(ecg_data["ECG"]), endpoint=False),
                    ecg_data["ECG"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["ECG"][0]["ECG"]) - 1,
                        num=len(list_data_all["ECG"][0]["ECG"]),
                        endpoint=False,
                    ),
                    method="linear",
                )
                interpolated_rpeaks = nk.signal_interpolate(
                    np.linspace(0, len(ecg_data["R-peaks"]) - 1, num=len(ecg_data["R-peaks"]), endpoint=False),
                    ecg_data["R-peaks"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["ECG"][0]["R-peaks"]) - 1,
                        num=len(list_data_all["ECG"][0]["R-peaks"]),
                        endpoint=False,
                    ),
                    method="linear",
                )
                interpolated_ibi = nk.signal_interpolate(
                    np.linspace(0, len(ecg_data["IBI"]) - 1, num=len(ecg_data["IBI"]), endpoint=False),
                    ecg_data["IBI"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["ECG"][0]["IBI"]) - 1,
                        num=len(list_data_all["ECG"][0]["IBI"]),
                        endpoint=False,
                    ),
                    method="linear",
                )
                interpoldated_hr = nk.signal_interpolate(
                    np.linspace(0, len(ecg_data["HR"]) - 1, num=len(ecg_data["HR"]), endpoint=False),
                    ecg_data["HR"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["ECG"][0]["HR"]) - 1,
                        num=len(list_data_all["ECG"][0]["HR"]),
                        endpoint=False,
                    ),
                    method="linear",
                )

                # Get the timestamps of the first subject
                timestamps_first_subject = list_data_all["ECG"][0]["timestamp"]

                # Create new array with subject number that has the same length as the interpolated data
                subject_array = [ecg_data["subject"].unique()[0]] * len(interpolated_ecg)

                # Initialize new empty dataframe
                ecg_data_interpolated = pd.DataFrame()

                # Add the interpolated data to the new dataframe
                ecg_data_interpolated["subject"] = subject_array
                ecg_data_interpolated["timestamp"] = timestamps_first_subject
                ecg_data_interpolated["ECG"] = interpolated_ecg
                ecg_data_interpolated["R-peaks"] = interpolated_rpeaks
                ecg_data_interpolated["IBI"] = interpolated_ibi
                ecg_data_interpolated["HR"] = interpoldated_hr

                # Append the interpolated data to the list
                list_data_all_interpolated["ECG"].append(ecg_data_interpolated)

            else:
                list_data_all_interpolated["ECG"].append(ecg_data)

        # Interpolate PPG data
        for i, ppg_data in enumerate(list_data_all["PPG"]):
            # Interpolate PPG data to match the length of the first subject's data
            if i != 0:
                interpolated_ppg = nk.signal_interpolate(
                    np.linspace(0, len(ppg_data["PPG"]) - 1, num=len(ppg_data["PPG"]), endpoint=False),
                    ppg_data["PPG"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["PPG"][0]["PPG"]) - 1,
                        num=len(list_data_all["PPG"][0]["PPG"]),
                        endpoint=False,
                    ),
                    method="linear",
                )
                interpolated_ppg_peaks = nk.signal_interpolate(
                    np.linspace(0, len(ppg_data["PPG-peaks"]) - 1, num=len(ppg_data["PPG-peaks"]), endpoint=False),
                    ppg_data["PPG-peaks"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["PPG"][0]["PPG-peaks"]) - 1,
                        num=len(list_data_all["PPG"][0]["PPG-peaks"]),
                        endpoint=False,
                    ),
                    method="linear",
                )
                interpolated_ppg_ibi = nk.signal_interpolate(
                    np.linspace(0, len(ppg_data["IBI"]) - 1, num=len(ppg_data["IBI"]), endpoint=False),
                    ppg_data["IBI"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["PPG"][0]["IBI"]) - 1,
                        num=len(list_data_all["PPG"][0]["IBI"]),
                        endpoint=False,
                    ),
                    method="linear",
                )
                interpolated_ppg_hr = nk.signal_interpolate(
                    np.linspace(0, len(ppg_data["HR"]) - 1, num=len(ppg_data["HR"]), endpoint=False),
                    ppg_data["HR"].to_numpy(),
                    np.linspace(
                        0,
                        len(list_data_all["PPG"][0]["HR"]) - 1,
                        num=len(list_data_all["PPG"][0]["HR"]),
                        endpoint=False,
                    ),
                    method="linear",
                )

                # Get the timestamps of the first subject
                timestamps_first_subject = list_data_all["PPG"][0]["timestamp"]

                # Create new array with subject number that has the same length as the interpolated data
                subject_array = [ppg_data["subject"].unique()[0]] * len(interpolated_ppg)

                # Initialize new empty dataframe
                ppg_data_interpolated = pd.DataFrame()

                # Add the interpolated data to the new dataframe
                ppg_data_interpolated["subject"] = subject_array
                ppg_data_interpolated["timestamp"] = timestamps_first_subject
                ppg_data_interpolated["PPG"] = interpolated_ppg
                ppg_data_interpolated["PPG-peaks"] = interpolated_ppg_peaks
                ppg_data_interpolated["IBI"] = interpolated_ppg_ibi
                ppg_data_interpolated["HR"] = interpolated_ppg_hr

                # Append the interpolated data to the list
                list_data_all_interpolated["PPG"].append(ppg_data_interpolated)

            else:
                list_data_all_interpolated["PPG"].append(ppg_data)

        # ---------------------- 3b. Concatenate all data ----------------------
        print("Concatenating all data...")
        # Concatenate all ECG data
        all_ecg_data = pd.concat(list_data_all_interpolated["ECG"], ignore_index=True)

        # Concatenate all PPG data
        all_ppg_data = pd.concat(list_data_all_interpolated["PPG"], ignore_index=True)

        # Save concatenated data to tsv files
        print("Saving concatenated data to tsv files...")

        all_ecg_data.to_csv(
            data_dir
            / exp_name
            / derivative_name
            / preprocessed_name
            / averaged_name
            / datatype_name
            / f"all_subjects_task-{task}_physio_preprocessed_ecg.tsv",
            sep="\t",
            index=False,
        )
        all_ppg_data.to_csv(
            data_dir
            / exp_name
            / derivative_name
            / preprocessed_name
            / averaged_name
            / datatype_name
            / f"all_subjects_task-{task}_physio_preprocessed_ppg.tsv",
            sep="\t",
            index=False,
        )

        # ---------------------- 3b. Average over all participants ----------------------
        print("Averaging over all participants' ECG and PPG data...")

        # Drop subject column
        all_ecg_data = all_ecg_data.drop(columns=["subject"])
        all_ppg_data = all_ppg_data.drop(columns=["subject"])

        # Drop peak columns
        all_ecg_data = all_ecg_data.drop(columns=["R-peaks"])
        all_ppg_data = all_ppg_data.drop(columns=["PPG-peaks"])

        # Calculate averaged data
        ecg_data_mean = all_ecg_data.groupby("timestamp").mean()
        ppg_data_mean = all_ppg_data.groupby("timestamp").mean()

        # Reset index
        ecg_data_mean.reset_index(inplace=True)  # noqa: PD002
        ppg_data_mean.reset_index(inplace=True)  # noqa: PD002

        # Save averaged data to tsv files
        print("Saving averaged data to tsv files...")

        ecg_data_mean.to_csv(
            data_dir
            / exp_name
            / derivative_name
            / preprocessed_name
            / averaged_name
            / datatype_name
            / f"avg_task-{task}_physio_preprocessed_ecg.tsv",
            sep="\t",
            index=False,
        )

        ppg_data_mean.to_csv(
            data_dir
            / exp_name
            / derivative_name
            / preprocessed_name
            / averaged_name
            / datatype_name
            / f"avg_task-{task}_physio_preprocessed_ppg.tsv",
            sep="\t",
            index=False,
        )

        # Plot averaged data
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        axs[0, 0].plot(
            ecg_data_mean["timestamp"][:: sampling_rates["ecg"]], ecg_data_mean["IBI"].dropna(), color=colors["ECG"][0]
        )
        axs[0, 0].set_ylabel("IBI from ECG")
        axs[0, 1].plot(
            ecg_data_mean["timestamp"][:: sampling_rates["ecg"]], ecg_data_mean["HR"].dropna(), color=colors["ECG"][1]
        )
        axs[0, 1].set_ylabel("HR from ECG")
        axs[1, 0].plot(
            ppg_data_mean["timestamp"][:: sampling_rates["ppg"]], ppg_data_mean["IBI"].dropna(), color=colors["PPG"][0]
        )
        axs[1, 0].set_ylabel("IBI from PPG")
        axs[1, 1].plot(
            ppg_data_mean["timestamp"][:: sampling_rates["ppg"]], ppg_data_mean["HR"].dropna(), color=colors["PPG"][1]
        )
        axs[1, 1].set_ylabel("HR from PPG")
        fig.suptitle(
            f"Mean IBI and HR from ECG and PPG data for AVR phase 3 (n={len(subjects)}), "
            "(no manual cleaning)"
            if not manual_cleaning
            else "(after manual cleaning)",
            fontsize=16,
        )
        # Set x-axis labels to minutes instead of seconds for all axes
        for ax in axs.flat:
            ax.set_xlabel("Time (min)")
            x_ticks = ax.get_xticks()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{round(x/60)}" for x in x_ticks])
            # Add vertical lines for event markers
            # Exclude first and last event markers
            # And only use every second event marker to avoid overlap
            for _, row in events_experiment.iloc[0:-1:2].iterrows():
                ax.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)

        # Save plot to results directory
        plt.savefig(
            Path(results_dir) / exp_name / averaged_name / datatype_name / f"avg_task-{task}_physio_IBI-HR.png"
        )

        if show_plots:
            plt.show()

        plt.close()

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    preprocess_physiological()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

