"""
Script for quick inspection of behavioral & physiological data from .xdf files & saving them in BIDS-compatible format.

Script to read in behavioral and physiological data from .xdf files, check the available streams,
plot the raw data for quick inspection, and save them in BIDS-compatible format.
Optionally, LSL offset correction can be selected to correct LSL event marker time stamps
for the known LSL-BrainAmp latency.

The following steps are performed:
1. Load the .xdf file and check the available streams.
2. Plot the raw data for quick inspection.
    a. Event markers
    b. Behavioral data (Ratings)
    c. VR data (Head movement, Eye tracking)
    d. Physiological data (EEG, ECG, respiration, PPG)
3. Create BIDS-compatible files & save them in the appropriate directory. TODO
    a.  Create a *_events.tsv file containing the event markers, and a _events.json file containing the event metadata.
        Save in rawdata/sub-<participant>/beh/.
    b.  Create a *_beh.tsv file containing the rating data, and a _beh.json file containing the metadata.
        Save in rawdata/sub-<participant>/beh/.
    c.  Create a *_motion.tsv.gz file containing the headmovement data, and a _motion.json file containing the metadata.
        Save in rawdata/sub-<participant>/motion/.
    d.  Create a *_recording_eyetracking_physio.tsv.gz file containing the eye tracking data, and a _recording_eyetracking_physio.json file containing the metadata.
        Save in rawdata/sub-<participant>/eyetrack/.
    e.  Create a *_eeg.tsv.gz file containing the EEG data, and a _eeg.json file containing the metadata.
        Save in rawdata/sub-<participant>/eeg/.
    f.  Create a *_physio.tsv.gz file containing the physiological data (ECG, respiration, PPG), and a _physio.json file containing the metadata.
        Save in rawdata/sub-<participant>/eeg/.

Required packages: pyxdf, mne

Author: Lucy Roellecke
Contact: lucy.roellecke[at]fu-berlin.de
Created on: 30 April 2024
Last update: 13 June 2024
"""

# %% Import

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from matplotlib import cm

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["pilot001", "pilot002"]  # Adjust as needed
session = "S001"  # Adjust as needed
run = "001"  # Adjust as needed
subject_task_mapping = {"pilot001": "AVRnomov", "pilot002": "AVRmov"}  # For pilot data
# subject 001 and 002 were the same person but once without movement and once with movement

# Specify the data path info (in BIDS format)
# change with the directory of data storage
data_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/")
results_dir = Path("/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/phase3/")
exp_name = "AVR"
sourcedata_name = "sourcedata"  # sourcedata folder
rawdata_name = "rawdata"  # rawdata folder
datatype_names = ["beh", "eeg", "eyetrack", "motion"]  # data type specification

# Create raw_data directories if they do not exist
for subject in subjects:
    subject_folder = data_dir / exp_name / rawdata_name / f"sub-{subject}"
    # Create the subject folder if it does not exist
    subject_folder.mkdir(parents=True, exist_ok=True)
    for datatype_name in datatype_names:
        datatype_folder = subject_folder / datatype_name
        # Create the datatype folder if it does not exist
        datatype_folder.mkdir(parents=True, exist_ok=True)
# Create results directories if they do not exist
    subject_folder_results = results_dir / exp_name / f"sub-{subject}"
    # Create the subject folder if it does not exist
    subject_folder_results.mkdir(parents=True, exist_ok=True)
    for datatype_name in datatype_names:
        datatype_folder_results = subject_folder_results / datatype_name
        # Create the datatype folder if it does not exist
        datatype_folder_results.mkdir(parents=True, exist_ok=True)

# Offset correction for LSL-BrainAmp fixed delay of 20 ms
# Set to "False" if no offset correction is needed
LSLoffset_corr = True
LSLoffset = 0.020  # LSL markers precede the actual stimulus presentation by approx. 20 ms

# Define the streams to be selected for further processing
selected_streams = ["Events", "Rating.CR", "Head.PosRot", "EDIA.Eye.CENTER", "LiveAmpSN-054206-0127"]
stream_modality_mapping = {
    "Events": "events",
    "Rating.CR": "beh",
    "Head.PosRot": "motion",
    "EDIA.Eye.CENTER": "eyetrack",
    "LiveAmpSN-054206-0127": "eeg",
}
stream_sampling_rate = {"Rating.CR": 50, "Head.PosRot": 90, "EDIA.Eye.CENTER": 120, "LiveAmpSN-054206-0127": 500}
stream_dimensions = {
    "Rating.CR": {0: "valence", 1: "arousal"},
    "Head.PosRot": {0: "PosX", 1: "PosY", 2: "PosZ", 3: "RotX", 4: "RotY", 5: "RotZ", 6: "RotW"},
    "EDIA.Eye.CENTER": {
        0: "PosX",
        1: "PosY",
        2: "PosZ",
        3: "Pitch",
        4: "Yaw",
        5: "Roll",
        6: "PupilDiameter",
        7: "Confidence",
        8: "TimestampET",
    },
}
# labels for the motion parameters in BIDS format
motion_bids_labels = {"PosX": "x", "PosY": "y", "PosZ": "z", "RotX": "quat_x", "RotY": "quat_y", "RotZ": "quat_z", "RotW": "quat_w"}
physiological_modalities = ["eeg", "cardiac", "respiratory", "ppg"]

# Names of the channels for each modalitiey
channel_names = {
    "eeg": [
        "Fp1",
        "Fz",
        "F3",
        "F7",
        "FT9",
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
        "FT10",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "Fp2",
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
    "cardiac": ["AUX1"],
    "respiratory": ["AUX2"],
    "ppg": ["AUX3"],
}

channel_indices = {"eeg": range(0, 64), "cardiac": [64], "respiratory": [65], "ppg": [66]}


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def get_stream_indexes(streams, selected_streams):
    """
    Extract indexes of streams with specific names from a list of streams.

    Parameters
    ----------
    streams : list of dict
        A list of dictionaries, each representing a stream. Each dictionary is expected to contain
        a key named "info", which is another dictionary that should include a key named "name".
        The "name" key holds the name(s) of the stream.
    selected_streams : list of str
        A list of strings representing the names of the streams of interest. The function will
        search for these names within the "name" keys of the stream dictionaries in `streams`
        and return their indexes.

    Returns
    -------
    dict
        A dictionary mapping selected stream names to their indexes. The keys are the names of
        the streams (as found in the "name" key of each stream's "info"), and the values are the
        indexes of these streams in the original `streams` list.
    """
    stream_indexes = {}

    for index, stream in enumerate(streams):
        names = stream["info"]["name"]
        if any(name in selected_streams for name in names):
            stream_indexes[names[0]] = index

    return stream_indexes


def plot_raw_data(data, stream_name, sampling_rate, labels):
    """
    Plot the raw data for quick inspection.

    Parameters
    ----------
    data : numpy.ndarray
        The raw data to be plotted, expected to be a 2D array where rows correspond to samples
        over time and columns correspond to different dimensions or channels of the data.
    stream_name : str
        The name of the data stream. This name is used in the figure's title to identify the
        plotted data.
    sampling_rate : float
        The sampling rate of the data in Hertz (samples per second), used to scale the x-axis
        from samples to time.
    labels : dict
        A dictionary mapping the index of each dimension (or channel) to its label, used for
        subplot titles and y-axis labels.

    Returns
    -------
    None
    """
    # Create a figure with subplots for each dimension
    fig, ax = plt.subplots(len(labels), 1, figsize=(20, len(labels) * 2), sharex=True)

    # Generate a sequence of evenly spaced values
    color_values = np.linspace(0, 1, len(labels))

    # Use a colormap to generate a list of colors
    colors = [cm.rainbow(x) for x in color_values]

    # For each dimension defined in labels, create one subplot
    for i, label in labels.items():
        ax[i].plot(data[:, i], color=colors[i])
        ax[i].set_title(f"{label}")
        ax[i].set_ylabel(label)

    # Set the x-axis label for the last subplot
    ax[i].set_xlabel("Time (min)")
    # Convert time in samples to minutes
    ax[i].set_xticklabels([str(round(i / (60 * sampling_rate), 2)) for i in ax[i].get_xticks()])

    # Set the title for the whole figure
    fig.suptitle(f"{stream_name} for {subject_name}")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Iterate through each participant
    for subject in subjects:
        subject_name = "sub-" - subject  # participant ID
        file_name = subject_name + f"_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_eeg.xdf"

        # Merge information into complete datapath
        sourcedata_dir = Path(data_dir) / exp_name / sourcedata_name / subject_name / file_name

        # %% STEP 1: LOAD XDF FILE & CHECK STREAMS
        # Load XDF file
        streams, header = pyxdf.load_xdf(sourcedata_dir)

        # Print available streams in the XDF file
        print("Available streams in the XDF file:")
        for stream in streams:
            print(stream["info"]["name"])

        # List of the available streams in XDF file:
        # 'Head.PosRot':            Head movement from VR HMD
        #                           7 dimensions (PosX, PosY, PosZ, RotX, RotY, RotZ, RotW)
        #                           90 Hz, float32
        #                           Length 145192 samples
        # 'EDIA.Eye.LEFT':          Eye tracking data from left eye
        #                           9 dimensions (PosX, PosY, PosZ, Pitch, Yaw, Roll, PupilDiameter, Confidence,
        #                           TimestampET)
        #                           120 Hz, float32
        #                           Length 193546 samples
        # 'RightHand.PosRot':       Right hand movement from VR controller
        #                           7 dimensions (PosX, PosY, PosZ, RotX, RotY, RotZ, RotW)
        #                           90 Hz, float32
        #                           Length 145191 samples
        # 'Events':                 Event markers
        #                           1 dimension
        #                           string
        #                           Length 63 samples
        # 'EDIA.Eye.RIGHT':         Eye tracking data from right eye
        #                           9 dimensions (PosX, PosY, PosZ, Pitch, Yaw, Roll, PupilDiameter, Confidence,
        #                           TimestampET)
        #                           120 Hz, float32
        #                           Length 193546 samples
        # 'EDIA.Eye.CENTER':        Eye tracking data from center eye
        #                           9 dimensions (PosX, PosY, PosZ, Pitch, Yaw, Roll, PupilDiameter, Confidence,
        #                           TimestampET)
        #                           120 Hz, float32
        #                           Length 193546 samples
        # 'Rating.CR':              Continuous rating data from VR controller
        #                           2 dimensions (x: valence, y: arousal)
        #                           50 Hz, float32
        #                           Length 80631 samples
        # 'LiveAmpSN-054206-0127':  ExG data from BrainVision LiveAmp
        #                           70 dimensions (64 EEG channels, 3 AUX channels (ECG, RESP, PPG), ACC_X, ACC_Y,
        #                           ACC_Z)
        #                           500 Hz, float32
        #                           Length 807278 samples
        # 'Rating.SR':              Summary rating data from VR controller
        #                           2 dimensions (x: valence, y: arousal)
        #                           float32
        #                           Length 2 samples (one rating from training, one from experiment)
        # 'LeftHand.PosRot':        Left hand movement from VR controller
        #                           7 dimensions (PosX, PosY, PosZ, RotX, RotY, RotZ, RotW)
        #                           90 Hz, float32
        #                           Length 145192 samples

        # Extract indexes corresponding to certain streams
        indexes_info = get_stream_indexes(streams, selected_streams)
        selected_indexes = list(indexes_info.values())

        # %% STEP 2: PLOT RAW DATA FOR QUICK INSPECTION
        for stream in selected_streams:
            # Extract the stream data
            stream_data = streams[indexes_info[stream]]

            # STEP 2a: --------- EVENT MARKERS -----------
            if stream_modality_mapping[stream] == "events":
                # Extract the event markers
                event_markers = stream_data["time_series"]
                # Extract the event timestamps
                event_timestamps = stream_data["time_stamps"]

                # Correct the LSL event marker time stamps for the known LSL-BrainAmp latency
                if LSLoffset_corr:
                    event_timestamps_corrected = event_timestamps + LSLoffset

                # Combine event markers and timestamps into a dataframe in BIDS format
                event_data = pd.DataFrame(
                    data={"onset": event_timestamps_corrected,
                    "duration": "n/a",
                    "trial_type": (marker[0] for marker in event_markers)})

                # Print the event markers and timestamps
                print(event_data)

                # Plot the event markers and timestamps as a vertical lines
                figure, axis = plt.subplots(1, 1, figsize=(20, 2))
                axis.vlines(event_data["onset"], ymin=0, ymax=1, color="black")
                plt.title("Event Markers")
                axis.set_yticks([])
                axis.set_xlabel("Time (s)")
                plt.legend(event_data["trial_type"], loc="upper right")
                plt.show()

                results_behav_dir = Path(results_dir) / exp_name / f"sub-{subject}" / "beh"

                # Save the plot as a .png file
                event_plot_name = f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_events.png"
                event_plot_file = results_behav_dir / event_plot_name
                plt.savefig(event_plot_file)

            # STEP 2b: --------- RATING DATA -----------
            elif stream_modality_mapping[stream] == "beh":
                # Extract the rating data
                rating_data = stream_data["time_series"]
                # Extract the timestamps
                rating_timestamps = stream_data["time_stamps"]

                # Combine rating data and timestamps into a dataframe with three columns in BIDS format
                rating_data_dataframe = pd.DataFrame(data=rating_data, columns=["valence", "arousal"])
                # Make the timestamps column the first column
                rating_data_dataframe.insert(0, "onset", rating_timestamps)
                # Add a second column for the duration of the rating
                rating_data_dataframe.insert(1, "duration", "n/a")

                # Plot the rating data
                plot_raw_data(rating_data, stream, stream_sampling_rate[stream], stream_dimensions[stream])

                results_behav_dir = Path(results_dir) / exp_name / f"sub-{subject}" / "beh"

                # Save the plot as a .png file
                rating_plot_name = f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_beh.png"
                rating_plot_file = results_behav_dir / rating_plot_name
                plt.savefig(rating_plot_file)

            # STEP 2c: --------- VR DATA (HEAD MOVEMENT & EYETRACKING) -----------
            elif stream_modality_mapping[stream] == "VR":
                # Extract the VR data
                vr_data = stream_data["time_series"]
                # Extract the timestamps
                vr_timestamps = stream_data["time_stamps"]

                # Exclude the timestamp dimension for the eyetracking data
                if stream == "EDIA.Eye.CENTER":
                    vr_data = vr_data[:, :-1]
                    stream_dimensions[stream] = {
                        key: value for key, value in stream_dimensions[stream].items() if key != 8
                    }

                # TODO: here I was !!!
                # Combine VR data and timestamps into a dataframe in BIDS format
                vr_data_dataframe = pd.DataFrame(data=vr_data, columns=list(motion_bids_labels[stream_dimensions[stream].values()]))
                # Make the timestamps column the first column
                vr_data_dataframe.insert(0, "timestamps", vr_timestamps)

                # Plot the VR data
                plot_raw_data(vr_data, stream, stream_sampling_rate[stream], stream_dimensions[stream])

                results_behav_dir = os.path.join(results_dir, modality_name)

                # Define stream-specific file names
                if stream == "Head.PosRot":
                    stream_name = "HeadMovement"
                elif stream == "EDIA.Eye.CENTER":
                    stream_name = "EyeTracking"

                # Save the plot as a .png file
                vr_plot_file = os.path.join(
                    results_behav_dir,
                    f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_{stream_name}.png",
                )
                plt.savefig(vr_plot_file)

                # Save the VR data in a tsv file
                vr_file = os.path.join(
                    data_dir,
                    exp_name,
                    rawdata_name,
                    datatype_name,
                    f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_{stream_name}.tsv",
                )
                vr_data_dataframe.to_csv(vr_file, sep="\t", index=False)

                # ------------ Sanity Check for Eye Tracking Data ------------
                if stream == "EDIA.Eye.CENTER":
                    # Read in event markers from the events.tsv file
                    events_file = os.path.join(
                        data_dir,
                        exp_name,
                        rawdata_name,
                        modality_name,
                        f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_events.tsv",
                    )
                    events_data = pd.read_csv(events_file, sep="\t")
                    # Get the timestamps of the White Flash event markers
                    white_flash_timestamps_start = events_data[events_data["markers"] == "WhiteOn"]["timestamps"]
                    white_flash_timestamps_stop = events_data[events_data["markers"] == "WhiteOff"]["timestamps"]

                    # Reset the timestamps of the flashes to start at 0
                    white_flash_timestamps_start = white_flash_timestamps_start - vr_data_dataframe["timestamps"].iloc[0]
                    white_flash_timestamps_stop = white_flash_timestamps_stop - vr_data_dataframe["timestamps"].iloc[0]

                    # Reset the timestamps of the eye tracking data to start at 0
                    vr_data_dataframe["timestamps"] = vr_data_dataframe["timestamps"] - vr_data_dataframe["timestamps"].iloc[0]

                    # Create a plot of the Pupil Diameter with the White Flash event markers
                    figure, axis = plt.subplots(1, 1, figsize=(20, 2))
                    axis.plot(vr_data_dataframe["timestamps"], vr_data_dataframe["PupilDiameter"], color="darkorange")
                    plt.title("Pupil Diameter")
                    axis.set_ylabel("Pupil Diameter")
                    # Add lines for the White Flash event markers and fill the area between them with transparent grey
                    for start, stop in zip(white_flash_timestamps_start, white_flash_timestamps_stop, strict=True):
                        plt.axvline(x=start, color="grey")
                        plt.axvline(x=stop, color="grey")
                        plt.fill_betweenx(y=[0, 5], x1=start, x2=stop, color="grey", alpha=0.1)

                    # Convert time in seconds to minutes
                    axis.set_xticklabels([str(round(i / 60, 2)) for i in axis.get_xticks()])
                    axis.set_xlabel("Time (min)")

                    # Save the plot as a .png file
                    eye_tracking_plot_file = os.path.join(
                        results_behav_dir,
                        f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_WhiteFlash.png",
                    )
                    plt.savefig(eye_tracking_plot_file)
                    plt.show()

                    # Create a separate plot for each white flash event and 5s before and after
                    for i, (start, stop) in enumerate(zip(white_flash_timestamps_start, white_flash_timestamps_stop)):
                        # Get the eye tracking data for the current flash event
                        flash_data = vr_data_dataframe[
                            (vr_data_dataframe["timestamps"] >= start - 5) & (vr_data_dataframe["timestamps"] <= stop + 5)
                        ]

                        # Create a plot of the Pupil Diameter with the White Flash event markers
                        figure, axis = plt.subplots(1, 1, figsize=(5, 4))
                        axis.plot(flash_data["timestamps"], flash_data["PupilDiameter"], color="darkorange")
                        plt.title("Pupil Diameter for White Flash Event " + str(i + 1))
                        axis.set_ylabel("Pupil Diameter")
                        # Add lines for the White Flash event markers and fill the area between them with transparent grey
                        plt.axvline(x=start, color="grey", alpha=0.1)
                        plt.axvline(x=stop, color="grey", alpha=0.1)
                        plt.fill_betweenx(y=[0, 5], x1=start, x2=stop, color="grey", alpha=0.1)

                        # Convert time in seconds to minutes
                        axis.set_xticklabels([str(round(i / 60, 2)) for i in axis.get_xticks()])
                        axis.set_xlabel("Time (min)")

                        # Save the plot as a .png file
                        eye_tracking_plot_file = os.path.join(
                            results_behav_dir,
                            f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_WhiteFlash_{i+1}.png",
                        )
                        plt.savefig(eye_tracking_plot_file)
                        plt.show()

            # STEP 2d: --------- PHYSIOLOGICAL DATA (EEG, ECG, RESPIRATION, PPG) -----------
            elif stream_modality_mapping[stream] == "physiological":
                # Loop over physiological modalities
                for modality in physiological_modalities:
                    # Get channel names for the current modality
                    channels_modality = channel_names[modality]
                    # Get channel indices for the current modality
                    channel_indices_modality = channel_indices[modality]

                    # Extract the time series data
                    data = stream_data["time_series"][:, channel_indices_modality].T
                    # Extract the timestamps
                    timestamps = stream_data["time_stamps"]

                    # Combine data and timestamps into a dataframe
                    dataframe = pd.DataFrame(
                        data=stream_data["time_series"][:, channel_indices_modality], columns=channels_modality
                    )
                    # Make the timestamps column the first column
                    dataframe.insert(0, "timestamps", timestamps)

                    # Get the sampling frequency
                    sampling_frequency = float(stream_data["info"]["nominal_srate"][0])
                    # Create MNE info object
                    info = mne.create_info(ch_names=channels_modality, sfreq=sampling_frequency)
                    # Create MNE raw object
                    raw = mne.io.RawArray(data, info)
                    # Plot the raw data
                    raw.plot(n_channels=32, title=f"{stream} {modality} data", duration=20, start=14)

        # %% STEP 3: CREATE BIDS-COMPATIBLE FILES & SAVE THEM IN APPROPRIATE DIRECTORY
        for datatype in datatype_names:

            # STEP 3a: --------- EVENT MARKERS -----------
            if datatype == "beh":
                # Create a *_events.tsv file containing the event markers
                events_filename = f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_events.tsv"
                events_file = Path(data_dir) / exp_name / rawdata_name / datatype / events_filename
                # Save the event markers in a tsv file
                event_data.to_csv(events_file, sep="\t", index=False)

                # Create a *_events.json file containing the event metadata
                events_metadata = {
                    "onset": {
                        "LongName": "Event onset",
                        "Description": "Time of the event in seconds",
                        "Units": "s"},
                    "duration": {"LongName": "Event duration",
                        "Description": "Duration of the event in seconds",
                        "Units": "s"},
                    "trial_type": {"LongName": "Event name",
                                "Description": "Name of the event"}}

                events_metadata_filename = f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_events.json"
                events_metadata_file = Path(data_dir) / exp_name / rawdata_name / datatype / events_metadata_filename
                # Save the event metadata in a json file
                with open(events_metadata_file, "w") as f:
                    json.dump(events_metadata, f, indent=4)
            
                # STEP 3b: --------- RATING DATA -----------
                # Create a *_beh.tsv file containing the rating data
                rating_filename = f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_beh.tsv"
                rating_file = Path(data_dir) / exp_name / rawdata_name / datatype / rating_filename
                # Save the rating data in a tsv file
                rating_data_dataframe.to_csv(rating_file, sep="\t", index=False)

                # Create a *_beh.json file containing the rating metadata
                rating_metadata = {
                    "onset": {
                        "LongName": "Rating onset",
                        "Description": "Time of the rating in seconds",
                        "Units": "s",
                        "SamplingRate": stream_sampling_rate["Rating.CR"]},
                    "duration": {"LongName": "Rating duration",
                        "Description": "Duration of the rating in seconds",
                        "Units": "s"},
                    "valence": {"LongName": "Valence rating",
                                "Description": "Valence rating from the VR controller",
                                "Range": [-1, 1],
                                "SamplingRate": stream_sampling_rate["Rating.CR"]},
                    "arousal": {"LongName": "Arousal rating",
                                "Description": "Arousal rating from the VR controller",
                                "Range": [-1, 1],
                                "SamplingRate": stream_sampling_rate["Rating.CR"]}}
                
                rating_metadata_filename = f"{subject_name}_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_beh.json"
                rating_metadata_file = Path(data_dir) / exp_name / rawdata_name / datatype / rating_metadata_filename
                # Save the rating metadata in a json file
                with open(rating_metadata_file, "w") as f:
                    json.dump(rating_metadata, f, indent=4)

# %%
