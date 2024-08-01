########################################################################################################################
# Script to preprocess CR data from CEAP dataset
#
# Inputs:       Raw data from CEAP dataset (download here: https://github.com/cwi-dis/CEAP-360VR-Dataset)
# Outputs:      Preprocessed data from CEAP dataset for each participant as csv files in "Preprocessed" directory
#
# Author:       Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
# Last version: 24.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import numpy as np
import os
import json

# ------------------------- SETUP ------------------------------
# change the data_path to the path where you saved the CEAP dataset
main_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/"
CEAP_datapath = main_path + "CEAP/data/CEAP-360VR/3_AnnotationData/"

# use the raw data of CEAP as the data in the "transformed" directory is transformed into square space
# and data in the "frame" directory is converted to a scale from 1 to 9 in accordance with the SAM
# (which we also don't want, we want it in a scale from -1 to 1)

# get the file path for each file of the CEAP dataset
subject_list_CEAP = os.listdir(CEAP_datapath + "Raw/")

# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if ".DS_Store" in subject_list_CEAP:
    subject_list_CEAP.remove(".DS_Store")

# sort the list in ascending order
subject_list_CEAP = sorted(
    subject_list_CEAP, key=lambda x: int(x.split(".")[0].split("_")[0].split("P")[1])
)

# video frame rate
video_frame_rate = 30
# frame rate of all videos was 30 except for video 1 (25)
# for analysis reasons, we pretend that video 1 also had a frame rate of 30
# and interpolate it in the same way as the other videos
# otherwise it is not possible to average over the videos of the same quadrant
# however, for an accurate analysis this value should be adjusted for video 1

length_videos = 60  # length of videos in seconds
number_timepoints = video_frame_rate * length_videos  # number of timepoints per video

# ------------------------- MAIN ------------------------------
# read and preprocess CEAP data
# add preprocessed data to dataframe
if __name__ == "__main__":
    # loop over subject files
    for i, file in enumerate(subject_list_CEAP):
        print(
            "Preprocessing subject "
            + str(i + 1)
            + " of "
            + str(len(subject_list_CEAP))
            + "..."
        )

        # initiate empty dataframe to fill with CEAP annotation data
        variables = {
            "sj_id": [],  # subject ID
            "time": [],  # time
            "cr_v": [],  # valence rating
            "cr_a": [],  # arousal rating
            "video_id": [],  # video id (V1-V8)
            "quadrant": [],  # quadrant (HP, LP, LN, HN)
        }
        CEAP_data = pd.DataFrame(variables)

        # read in subject json file
        with open(CEAP_datapath + "Raw/" + file, "r") as json_file:
            data = json.load(json_file)

        # access the annotation data inside the json file
        annotation_data = data["ContinuousAnnotation_RawData"][0][
            "Video_Annotation_RawData"
        ]

        # loop over videos (V1-V8)
        for video in range(0, len(annotation_data)):
            # get video ID
            video_id = annotation_data[video]["VideoID"]
            # get corresponding quadrant
            if video_id == "V1" or video_id == "V5":
                video_quadrant = "HP"
            elif video_id == "V3" or video_id == "V7":
                video_quadrant = "HN"
            elif video_id == "V2" or video_id == "V6":
                video_quadrant = "LP"
            else:
                video_quadrant = "LN"

            # get the timed annotation data
            timed_annotation_data = annotation_data[video]["TimeStamp_Xvalue_Yvalue"]

            # the annotation timepoints and the times between them are different for all participants in the CEAP dataset
            # this is because different devices and sensors with different frequencies were used (see CEAP Dataset description)
            # in order to be able to compare them to the AffectiveVR data, we need to align and synchronize the timepoints
            # we will first calculate the new sampling timestamps in each frame for each video
            # if the sampling frequency of the raw data is less than the video frame rate, a linear interpolation is performed
            # to determine the values at new sampling timestamps by fitting a line using the correspoding discrete samples in the raw data
            # for the raw sampling frequency higher than the video frame rate, we select the maximum value less than the specified timestamp of the re-sampled data
            # this approach was copied by the original authors (see CEAP Dataset description and GitHub repository)

            # create empty lists to fill with original values
            seconds_list = []
            valence_list = []
            arousal_list = []

            # loop over all timepoints and add them to the respective list
            for time_stamp in range(0, len(timed_annotation_data)):
                seconds_list.append(timed_annotation_data[time_stamp]["TimeStamp"])
                valence_list.append(timed_annotation_data[time_stamp]["X_Value"])
                arousal_list.append(timed_annotation_data[time_stamp]["Y_Value"])

            # create new lists with interpolated values
            frame_seconds_list = np.linspace(0, 60, 60 * video_frame_rate)
            frame_valence_list = np.interp(
                frame_seconds_list, seconds_list, valence_list
            )
            frame_arousal_list = np.interp(
                frame_seconds_list, seconds_list, arousal_list
            )

            # create empty list to fill with interpolated values
            interpolated_annotation_data = []
            # put interpolated values in dataframe
            for frame in range(0, len(frame_seconds_list)):
                interpolated_annotation_data_timeframe = {
                    "TimeStamp": round(frame_seconds_list[frame], 3),
                    "X_Value": round(frame_valence_list[frame], 3),
                    "Y_Value": round(frame_arousal_list[frame], 3),
                }
                interpolated_annotation_data.append(
                    interpolated_annotation_data_timeframe
                )

            # loop over all interpolated timepoints (1.800 timepoints per video)
            for t, timepoint in enumerate(interpolated_annotation_data):
                time = timepoint["TimeStamp"]  # get timepoint
                valence = timepoint["X_Value"]  # get valence rating
                arousal = timepoint["Y_Value"]  # get arousal rating

                # add everything to the dataframe we created
                CEAP_data.loc[video * number_timepoints + t, "sj_id"] = str(i + 1)
                CEAP_data.loc[video * number_timepoints + t, "time"] = time
                CEAP_data.loc[video * number_timepoints + t, "cr_v"] = valence
                CEAP_data.loc[video * number_timepoints + t, "cr_a"] = arousal
                CEAP_data.loc[video * number_timepoints + t, "video_id"] = video_id[1]
                CEAP_data.loc[
                    video * number_timepoints + t, "quadrant"
                ] = video_quadrant

        # group the data by 'video' and calculate the minimum 'time' for each video
        min_time_per_video = CEAP_data.groupby("video_id")["time"].transform("min")

        # subtract the minimum 'time' from each 'time'
        CEAP_data["time"] = CEAP_data["time"] - min_time_per_video

        # round to 2 numbers after decimal
        CEAP_data["time"] = round(CEAP_data["time"], 2)

        # remove the first 0.05 seconds of the data
        CEAP_data = CEAP_data.drop(CEAP_data[CEAP_data["time"] < 0.05].index)

        # save preprocessed data in preprocessed file
        filename, _ = os.path.splitext(file)
        new_filename = filename + "_preprocessed.csv"
        preprocessed_path = os.path.join(CEAP_datapath, "Preprocessed/")
        # create new directory if folder does not exist yet
        if not os.path.exists(preprocessed_path):
            os.mkdir(preprocessed_path)
        # save preprocessed data to csv file
        CEAP_data.to_csv(os.path.join(preprocessed_path, new_filename), index=False)
