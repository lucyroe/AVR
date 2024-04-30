# -*- coding: utf-8 -*-
"""
Name: read_xdf.py

Author: Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
        Largely based on Marta Gerosa's script for the BBSIG project
Created on: 30 April 2024
Last updated: 30 April 2024

Initial pipeline for importing physio data file (in .xdf format) and checking the streams available.
Relevant streams are then extracted and saved into a BIDS-compatible file in .tsv.gz format.
LSL event markers are saved in both BIDS-compatible _events.tsv format and in MNE-compatible _events.txt format.

Optionally, LSL offset correction can be selected to correct LSL event marker time stamps for the known LSL-BrainAmp latency.

The following steps are performed:
    - Load .xdf file streams (sampled data) and headers (metadata)
    - List available streams in the .xdf file
    - TODO: define next steps

Requires: pyxdf

"""

#%% PREPARATION

############## Import modules ##############

import pyxdf # for loading XDF files
import numpy as np
import pandas as pd
import os
import json
from collections import OrderedDict

############## Functions ##############

# get_stream_indexes: function to get indexes of selected streams names 
def get_stream_indexes(streams, selected_streams):
    """
    Extract indexes of streams with specific names from a list of streams.

    Parameters:
    - streams (list): list of dictionaries, each corresponding to a stream
    - selected_streams (list): list of target stream names

    Returns:
    - dict: A dictionary mapping selected stream names to their indexes
    """
    stream_indexes = {}
    
    for index, stream in enumerate(streams):
        names = stream["info"]["name"]
        if any(name in selected_streams for name in names):
            stream_indexes[names[0]] = index
    
    return stream_indexes


############## Offset correction setup ##############

# Offset correction for LSL-BrainAmp fixed delay of 56 ms
# Set to "False" if no offset correction is needed
LSLoffset_corr = False
LSLoffset = 0.055 # LSL markers precede the actual stimulus presentation by approx. 56 ms
# TODO: define offset correction after doing Photodiode check

#%% DATA EXTRACTION

############## Define path for physio data ##############

participant_ids = ["002"]  # Adjust as needed

# Specify the data path info (in BIDS format)
wd = '/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/' # change with the directory of data storage
exp_name = 'AVR'
sourcedata_name = 'sourcedata' # sourcedata folder
rawdata_name = 'rawdata' # rawdata folder 
datatype_name = 'physio' # data type specification
modality_name = 'beh' # modality folder 


# Iterate through each participant
for subjID in participant_ids:
    subj_name = 'sub-P' + str(subjID) # participant ID (in BIDS format)
    file_name = subj_name + '_ses-S001_task-Default_run-001_eeg.xdf'

    # Merge information into complete datapath
    physiodata_dir = os.path.join(wd, exp_name, sourcedata_name, file_name)

    ############## Load physio streams from xdf file ##############

    # Load XDF file 
    streams, header = pyxdf.load_xdf(physiodata_dir)

    ###################### Check data streams #####################

    # Print available streams in the XDF file
    print("Available streams in the XDF file:")
    for stream in streams:
        print(stream["info"]["name"])

    # List of the available streams in XDF file (not necessarily in this order):
    # - Unity.PosRot.Head - Position and rotation of the head in Unity (7 channels: Pos X, Pos Y, Pos Z, Rot X, Rot Y, Rot Z, Rot W)
    # - Unity.Pose - Pose of the Unity scene (7 channels: Pos X, Pos Y, Pos Z, Rot X, Rot Y, Rot Z, Rot W)
    # - Unity.PosRot.Left - Position and rotation of the left hand in Unity (7 channels: Pos X, Pos Y, Pos Z, Rot X, Rot Y, Rot Z, Rot W)
    # - Unity.PosRot.Right - Position and rotation of the right hand in Unity (7 channels: Pos X, Pos Y, Pos Z, Rot X, Rot Y, Rot Z, Rot W)
    # - Unity.MatColour - Color changes in unity, serving as event markers (1 channel: RGBA value)
    # - LiveAmpSN-054206-0127-DeviceTrigger - Device trigger markers (empty)
    # - LiveAmpSN-054206-0127-STETriggerIn - STE trigger markers (empty)
    # - LiveAmpSN-054206-0127 - ExG data (67-68 channels: 64 EEG, (AUX_1), ACC_X, ACC_Y, ACC_Z)

    # TODO: Test whether annotation data is recorded in another stream

    # Print length of time series for each stream
    for stream in streams:
        print(f"Length of time series for stream '{stream['info']['name'][0]}': {len(stream['time_series'])}")

    # All Unity timeseries have the same length (except for the event marker stream), two of the LiveAmp streams are empty, and the ExG LiveAmp stream has a length x10 of the Unity streams

    # Check whether time stamps are the same
    # Print first 10 time stamps for each stream
    for stream in streams:
        print(f"First 10 time stamps for stream '{stream['info']['name'][0]}': {stream['time_stamps'][:10]}")
    
    # Time stamps are slightly different for each stream (and don't start at 0)

    # Check whether the data makes sense
    # Print first 2 time series points for each stream
    for stream in streams:
        print(f"First 2 time series points for stream '{stream['info']['name'][0]}': {stream['time_series'][:2]}")

    ############## Physio data extraction ##############
    # Extract indexes corresponding to 'LiveAmpSN-054206-0127' & 'Unity.MatColour' streams
    selected_streams = ['LiveAmpSN-054206-0127', 'Unity.MatColour']
    indexes_info = get_stream_indexes(streams, selected_streams)
    selected_indexes = [index for index in indexes_info.values()]



#%% CREATE BIDS _PHYSIO.TSV.GZ FILE


#%% CREATE BIDS _EVENTS.TSV FILE


#%% CREATE MNE-COMPATIBLE ANNOTATIONS
