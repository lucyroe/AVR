########################################################################################################################
# Script to preprocess CR data from CASE dataset
#
# Inputs:       Interpolated annotation data from CASE dataset (download here: https://springernature.figshare.com/collections/A_dataset_of_continuous_affect_annotations_and_physiological_signals_for_emotion_analysis/4260668)
# Outputs:      Preprocessed annotation data from CASE dataset for each participant as csv files in "preprocessed/" directory
#
# Author:       Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
# Last version: 24.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import numpy as np
import os

# ------------------------- SETUP ------------------------------
# change the datapath to the path where you saved the physiological CASE data
main_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CASE/data/"
datapath = os.path.join(main_path, "interpolated/annotations/")
preprocessed_path = os.path.join(main_path, "preprocessed/")

# check if preprocessed path exists, if not create it
if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)

quadrant = ["HP", "LP", "LN", "HN"]

# get the file path for each file of the CASE dataset
subject_list_CASE = os.listdir(datapath)

# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if ".DS_Store" in subject_list_CASE:
    subject_list_CASE.remove(".DS_Store")

# remove the README file from the list
if "README_annotations.txt" in subject_list_CASE:
    subject_list_CASE.remove("README_annotations.txt")

# sort the list in ascending order
subject_list_CASE = sorted(
    subject_list_CASE, key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Create a dictionary mapping video values to quadrant labels
video_quadrant_mapping = {
    1: "HP",
    2: "HP",
    3: "LN",
    4: "LN",
    5: "LP",
    6: "LP",
    7: "HN",
    8: "HN",
}

# ------------------------- MAIN ------------------------------
# read and preprocess annotation CASE data
# add preprocessed data to dataframe
if __name__ == "__main__":
    # loop over subject files
    for i, file in enumerate(subject_list_CASE):
        print(
            "Preprocessing subject "
            + str(i + 1)
            + " of "
            + str(len(subject_list_CASE))
            + "..."
        )

        # read in subject csv file
        data = pd.read_csv(datapath + file)

        # insert subject ID column
        data.insert(0, "sj_id", i + 1)
        # add empty 'quadrant' column
        data["quadrant"] = np.nan

        # rename columns
        data = data.rename(
            columns={
                "jstime": "time",
                "video": "video_id",
                "valence": "cr_v",
                "arousal": "cr_a",
            }
        )

        # filter out rows with start, end, and blue videos [10, 11, 12]
        data = data[~data["video_id"].isin([10, 11, 12])]

        # map the video values to quadrant labels using the dictionary
        data["quadrant"] = data["video_id"].map(video_quadrant_mapping)

        # rescale valence and arousal ratings from 0 to 10 range to -1 to 1 range
        data["cr_v"] = data["cr_v"] / 5 - 1
        data["cr_a"] = data["cr_a"] / 5 - 1

        # Group the data by 'video_id' and calculate the minimum 'time' for each video
        min_time_per_video = data.groupby("video_id")["time"].transform("min")

        # Subtract the minimum 'time' from each 'time' value
        data["time"] = data["time"] - min_time_per_video

        # rescale time from ms to s
        # round to 2 numbers after decimal
        data["time"] = round(data["time"] / 1000, 3)

        # remove the first 0.05 seconds of the data
        data = data.drop(data[data["time"] < 0.05].index)

        # the physiological data of the CASE dataset has a sampling rate of 1000 Hz
        # (one measurement per millisecond)
        # the annotation data of the CASE dataset has a sampling rate of 20 Hz
        # (one measurement per 50 milliseconds)
        # this leads to many more datapoints in the physiological data than in the annotation data
        # TODO: is this a problem for changepoint analysis? should we downsample the physiological data?

        # save preprocessed data to csv file
        filename, _ = os.path.splitext(file)
        new_filename = filename + "_preprocessed.csv"
        data.to_csv(os.path.join(preprocessed_path, new_filename), index=False)

        """
        data is saved as a csv file with the following columns (in that order):
            "sj_id": subject ID
            "time": time (seconds, s), sampling rate = 20 Hz, starts at 0.05 seconds
            "cr_v": valence rating (-1-1)
            "cr_a": arousal rating (-1-1)
            "video_id": video id (1-8)
            "quadrant": quadrant (HP, LP, LN, HN)
        """
