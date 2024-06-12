"""
Script for quick inspection of physiological data from .xdf files and saving them in BIDS-compatible format.

Script to read in physiological data from .xdf files, check the available streams,
plot the raw data for quick inspection, and save them in BIDS-compatible format.
LSL event markers are saved in both BIDS-compatible _events.tsv format and in MNE-compatible _events.txt format.
Optionally, LSL offset correction can be selected to correct LSL event marker time stamps
for the known LSL-BrainAmp latency.

The following steps are performed:
1. Load the .xdf file and check the available streams.
2. Plot the raw data for quick inspection.
3. Create BIDS-compatible files: TODO
    a. Create a _physio.tsv.gz file containing the physiological data.
    b. Create an _events.tsv file containing the event markers.
4. Save the BIDS-compatible files in the appropriate directory. TODO

Required packages: pyxdf, mne

Author: Lucy Roellecke (Largely based on Marta Gerosa's script for the BBSIG project)
Contact: lucy.roellecke[at]fu-berlin.de
Created on: 30 April 2024
Last update: 12 June 2024
"""

# %% Import

import os

import matplotlib.pyplot as plt
import mne
import pyxdf

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["001"]  # Adjust as needed
# "002"
session = "S001"  # Adjust as needed
run = "001"  # Adjust as needed
subject_task_mapping = {"001": "AVR_nomov", "002": "AVR_mov"}  # For pilot data
# subject 001 and 002 were the same person but once without movement and once with movement

# Specify the data path info (in BIDS format)
# change with the directory of data storage
wd = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/"
exp_name = "AVR"
sourcedata_name = "sourcedata"  # sourcedata folder
rawdata_name = "rawdata"  # rawdata folder
datatype_name = "physio"  # data type specification
modality_name = "beh"  # modality folder

# Offset correction for LSL-BrainAmp fixed delay of 56 ms
# Set to "False" if no offset correction is needed
LSLoffset_corr = False
LSLoffset = 0.055  # LSL markers precede the actual stimulus presentation by approx. 56 ms
# TODO: define offset correction after doing Photodiode check  # noqa: FIX002

# Define the streams to be selected for further processing
selected_streams = ["Events", "Rating.CR", "Head.PosRot", "EDIA.Eye.CENTER", "LiveAmpSN-054206-0127"]
stream_modality_mapping = {"Events": "events", "Rating.CR": "rating", "Head.PosRot": "VR", 
                            "EDIA.Eye.CENTER": "VR", "LiveAmpSN-054206-0127": "physiological"}
stream_sampling_rate = {"Rating.CR": 50, "Head.PosRot": 90, "EDIA.Eye.CENTER": 120,
                        "LiveAmpSN-054206-0127": 500}
stream_dimensions = {"Rating.CR": {0: "valence", 1:"arousal"}, "Head.PosRot": {0: "PosX", 1: "PosY", 2: "PosZ",
                    3: "RotX", 4: "RotY", 5: "RotZ", 6: "RotW"}, "EDIA.Eye.CENTER": {0: "PosX", 1: "PosY", 2: "PosZ",
                    3: "Pitch", 4: "Yaw", 5: "Roll", 6: "PupilDiameter", 7: "Confidence", 8: "TimestampET"}}

physiological_modalities = ["eeg", "ecg", "ppg", "respiration"]

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
    "ecg": ["AUX1"],
    "respiration": ["AUX2"],
    "ppg": ["AUX3"]}

channel_indices = {
    "eeg": range(0, 64),
    "ecg": [64],
    "respiration": [65],
    "ppg": [66]}


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def get_stream_indexes(streams, selected_streams):
    """
    Extract indexes of streams with specific names from a list of streams.

    Parameters
    ----------
    streams (list): list of dictionaries, each corresponding to a stream
    selected_streams (list): list of target stream names

    Returns
    -------
    Dict: A dictionary mapping selected stream names to their indexes
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
    data (numpy.ndarray): raw data
    stream_name (str): name of the stream
    sampling_rate (float): sampling rate of the data
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    # For each dimension defined in labels, plot one line
    for key, label in labels.items():
        ax.plot(data[:, key], label=label)
    ax.set_title(f"{stream_name} for {subject_name}")
    ax.set_xlabel("Time (min)")
    # Convert time in samples to minutes
    ax.set_xticklabels([str(round(i / (60 * sampling_rate), 2)) for i in ax.get_xticks()])
    ax.set_ylabel(stream_modality_mapping[stream_name])
    ax.legend()
    plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Iterate through each participant
    for subject in subjects:
        subject_name = "sub-P" + str(subject)  # participant ID
        file_name = subject_name + f"_ses-{session}_task-{subject_task_mapping[subject]}_run-{run}_eeg.xdf"

        # Merge information into complete datapath
        physiodata_dir = os.path.join(wd, exp_name, sourcedata_name, file_name)

        # %% STEP 1: LOAD XDF FILE & CHECK STREAMS
        # Load XDF file
        streams, header = pyxdf.load_xdf(physiodata_dir)

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
        #                           9 dimensions (PosX, PosY, PosZ, Pitch, Yaw, Roll, PupilDiameter, Confidence, TimestampET)
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
        #                           9 dimensions (PosX, PosY, PosZ, Pitch, Yaw, Roll, PupilDiameter, Confidence, TimestampET)
        #                           120 Hz, float32
        #                           Length 193546 samples
        # 'EDIA.Eye.CENTER':        Eye tracking data from center eye
        #                           9 dimensions (PosX, PosY, PosZ, Pitch, Yaw, Roll, PupilDiameter, Confidence, TimestampET)
        #                           120 Hz, float32
        #                           Length 193546 samples
        # 'Rating.CR':              Continuous rating data from VR controller
        #                           2 dimensions (x: valence, y: arousal)
        #                           50 Hz, float32
        #                           Length 80631 samples
        # 'LiveAmpSN-054206-0127':  ExG data from BrainVision LiveAmp
        #                           70 dimensions (64 EEG channels, 3 AUX channels (ECG, RESP, PPG), ACC_X, ACC_Y, ACC_Z)
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
        selected_indexes = [index for index in indexes_info.values()]

        # %% STEP 2: PLOT RAW DATA FOR QUICK INSPECTION

        for stream in selected_streams:
            # Extract the stream data
            stream_data = streams[indexes_info[stream]]

            # --------- EVENT MARKERS -----------
            if stream_modality_mapping[stream] == "events":
                # Extract the event markers
                event_markers = stream_data["time_series"]
                # Print the event markers
                print(f"Event markers: {event_markers}")

                # TODO:Save the event markers in a text file
            
            # --------- RATING DATA -----------
            elif stream_modality_mapping[stream] == "rating":
                # Extract the rating data
                rating_data = stream_data["time_series"]

                # Plot the rating data
                plot_raw_data(rating_data, stream, stream_sampling_rate[stream], stream_dimensions[stream])

            # --------- VR DATA (HEAD MOVEMENT & EYETRACKING) -----------
            elif stream_modality_mapping[stream] == "VR":
                # Extract the VR data
                vr_data = stream_data["time_series"]

                # Plot the VR data
                plot_raw_data(vr_data, stream, stream_sampling_rate[stream], stream_dimensions[stream])
                # TODO: eyetracking data looks weird -> check

            # --------- PHYSIOLOGICAL DATA (EEG, ECG, RESPIRATION, PPG) -----------
            elif stream_modality_mapping[stream] == "physiological":
                # Loop over physiological modalities
                for modality in physiological_modalities:
                    # Get channel names for the current modality
                    channels_modality = channel_names[modality]
                    # Get channel indices for the current modality
                    channel_indices_modality = channel_indices[modality]

                    # Extract the time series data
                    data = stream_data["time_series"][:, channel_indices_modality].T

                    # Get the sampling frequency
                    sampling_frequency = float(stream_data["info"]["nominal_srate"][0])
                    # Create MNE info object
                    info = mne.create_info(ch_names=channels_modality, sfreq=sampling_frequency)
                    # Create MNE raw object
                    raw = mne.io.RawArray(data, info)
                    # Plot the raw data
                    raw.plot(n_channels=32, title=f"{stream} {modality} data", duration=20, start=14)

        # %% STEP 3a: CREATE BIDS _PHYSIO.TSV.GZ FILE

        # %% STEP 3b: CREATE BIDS _EVENTS.TSV FILE

# %%
