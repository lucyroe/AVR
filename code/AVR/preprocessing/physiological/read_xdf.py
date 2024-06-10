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
4. Save the BIDS-compatible files in the appropriate directory.

Required packages: pyxdf, mne

Author: Lucy Roellecke (Largely based on Marta Gerosa's script for the BBSIG project)
Contact: lucy.roellecke[at]fu-berlin.de
Created on: 30 April 2024
Last update: 10 June 2024
"""

# %% Import

import os

import mne
import pyxdf

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

subjects = ["001"]  # Adjust as needed
session = "S001"  # Adjust as needed
run = "001"  # Adjust as needed

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
selected_streams = ["LiveAmpSN-054206-0127"]  # TODO: define the streams to be selected  # noqa: FIX002

physiological_modalities = ["eeg", "ecg"]
# "ppg", "respiration", "eyetracking"]

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
    "ppg": ["AUX2"],
    "respiration": ["AUX3"],
    "eyetracking": ["AUX4"],
}
# TODO: define channel names for each modality  # noqa: FIX002

channel_indices = {
    "eeg": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
    ],
    "ecg": [64],
    "ppg": [65],
    "respiration": [66],
    "eyetracking": [67],
}


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


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Iterate through each participant
    for subject in subjects:
        subject_name = "sub-P" + str(subject)  # participant ID
        file_name = subject_name + f"_ses-{session}_task-Default_run-{run}_eeg.xdf"

        # Merge information into complete datapath
        physiodata_dir = os.path.join(wd, exp_name, sourcedata_name, file_name)

        # %% STEP 1: LOAD XDF FILE & CHECK STREAMS
        # Load XDF file
        streams, header = pyxdf.load_xdf(physiodata_dir)

        # Print available streams in the XDF file
        print("Available streams in the XDF file:")
        for stream in streams:
            print(stream["info"]["name"])

        # List of the available streams in XDF file (not necessarily in this order):
        # TODO: list the streams in the XDF file

        # Print length of time series for each stream
        for stream in streams:
            print(f"Length of time series for stream '{stream['info']['name'][0]}': {len(stream['time_series'])}")

        # Check whether time stamps are the same
        # Print first 10 time stamps for each stream
        for stream in streams:
            print(f"First 10 time stamps for stream '{stream['info']['name'][0]}': {stream['time_stamps'][:10]}")

        # Check whether the data makes sense
        # Print first 2 time series points for each stream
        for stream in streams:
            print(f"First 2 time series points for stream '{stream['info']['name'][0]}': {stream['time_series'][:2]}")

        # Extract indexes corresponding to certain streams
        indexes_info = get_stream_indexes(streams, selected_streams)
        selected_indexes = [index for index in indexes_info.values()]

        # %% STEP 2: PLOT RAW DATA FOR QUICK INSPECTION

        for stream in selected_streams:
            # Extract the stream data
            stream_data = streams[indexes_info[stream]]

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
                raw.plot(n_channels=5, title=f"{stream} {modality} data", duration=1, start=14)

        # %% STEP 3a: CREATE BIDS _PHYSIO.TSV.GZ FILE

        # %% STEP 3b: CREATE BIDS _EVENTS.TSV FILE

# %%
