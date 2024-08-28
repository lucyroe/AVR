########################################################################################################################
# Script to preprocess physiological data from CEAP dataset
#
# Inputs:       Frame physiological data from CEAP dataset (download here: https://github.com/cwi-dis/CEAP-360VR-Dataset)
# Outputs:      Preprocessed physiological data from CEAP dataset for each participant as csv files in "Preprocessed" directory
#
# Author:       Lucy Roellecke (lucy.roellecke[at]tuta.com)
# Last version: 24.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import os
import json

# ------------------------- SETUP ------------------------------
# change the datapath to the path where you saved the physiological CEAP data
main_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CEAP/data/CEAP-360VR/5_PhysioData/"
preprocessed_path = os.path.join(main_path, "Preprocessed/")

# check if preprocessed path exists, if not create it
if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)

# get the file path for each file of the CEAP dataset
subject_list_CEAP = os.listdir(main_path + "Frame/")

# use the "frame" data of CEAP as this data is already filtered, normalized and interpolated
# the approach used can be found in the CEAP dataset description and GitHub repository

# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if ".DS_Store" in subject_list_CEAP:
    subject_list_CEAP.remove(".DS_Store")

# sort the list in ascending order
subject_list_CEAP = sorted(
    subject_list_CEAP, key=lambda x: int(x.split(".")[0].split("_")[0].split("P")[1])
)

# list of physiological measures taken
physiological_measures = ["ACC", "BVP", "EDA", "HR", "IBI", "SKT"]

# ------------------------- MAIN ------------------------------
# read and preprocess physiological CEAP data
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

        # initiate empty dataframe to fill with CEAP physiological data
        variables = {
            "sj_id": [],  # subject ID
            "time": [],  # time
            "acc_x": [],  # acceleration in x direction (64 Hz)
            "acc_y": [],  # acceleration in y direction (64 Hz)
            "acc_z": [],  # acceleration in z direction (64 Hz)
            "bvp": [],  # blood volume pulse (64 Hz)
            "eda": [],  # electrodermal activity in uS (4 Hz)
            "skt": [],  # skin temperature in degrees Celsius (4 Hz)
            "hr": [],  # heart rate (1 Hz)
            "ibi": [],  # interbeat interval (missing from participants 2 and 12)
            "video_id": [],  # video id (V1-V8)
            "quadrant": [],  # quadrant (HP, LP, LN, HN)
        }
        CEAP_data = pd.DataFrame(variables)

        # read in subject json file
        with open(main_path + "Frame/" + file, "r") as json_file:
            data = json.load(json_file)

        # access the data inside the json file
        physiological_data = data["Physio_FrameData"][0]["Video_Physio_FrameData"]

        # loop over videos (V1-V8)
        for video in range(0, len(physiological_data)):
            # get video ID
            video_id = physiological_data[video]["VideoID"]
            # get corresponding quadrant
            if video_id == "V1" or video_id == "V5":
                video_quadrant = "HP"
            elif video_id == "V3" or video_id == "V7":
                video_quadrant = "HN"
            elif video_id == "V2" or video_id == "V6":
                video_quadrant = "LP"
            else:
                video_quadrant = "LN"

            if video == 0:
                start_point = 0  # first video has 1500 timepoints
                number_timepoints = 0
            else:
                start_point = 1500
                number_timepoints = 1800  # all other videos have 1800 timepoints

            # loop over all physiological measures that were taken
            for measure in physiological_measures:
                # get the timed physiological data
                measure_data = physiological_data[video][measure + "_FrameData"]

                # loop over all timepoints
                for t, timepoint in enumerate(measure_data):
                    # IBI has a lot less datapoints than the other measures
                    # we need to add the missing timepoints and fill them with NaN
                    if measure == "IBI":
                        # round time to 1 number after decimal
                        ibi_time = round(timepoint["TimeStamp"], 1)
                        # find the index of the timepoint in the dataframe that matches ibi_time
                        CEAP_data_video = CEAP_data[
                            CEAP_data["video_id"] == str(video + 1)
                        ]
                        index = CEAP_data_video.index[
                            round(CEAP_data_video["time"], 1) == ibi_time
                        ].tolist()
                        # if there is no timepoint in the dataframe that matches ibi_time
                        # print error message
                        if len(index) == 0:
                            print("No timepoint found for time " + str(ibi_time))

                        else:
                            # get interbeat interval and round to 3 numbers after decimal
                            ibi = round(timepoint["IBI"], 3)
                            # add interbeat interval to the dataframe we created
                            CEAP_data.loc[index[0], "ibi"] = ibi

                    # for all other measures
                    else:
                        time = timepoint["TimeStamp"]
                        # add time to the dataframe we created
                        CEAP_data.loc[
                            start_point + (video - 1) * number_timepoints + t, "time"
                        ] = time
                        # add subject number to the dataframe we created
                        CEAP_data.loc[
                            start_point + (video - 1) * number_timepoints + t, "sj_id"
                        ] = str(i + 1)

                        # add video ID and quadrant to the dataframe we created
                        CEAP_data.loc[
                            start_point + (video - 1) * number_timepoints + t,
                            "video_id",
                        ] = video_id[1]
                        CEAP_data.loc[
                            start_point + (video - 1) * number_timepoints + t,
                            "quadrant",
                        ] = video_quadrant

                        if measure == "ACC":
                            # get acceleration in x, y, and z direction
                            acc_x = timepoint["ACC_X"]
                            acc_y = timepoint["ACC_Y"]
                            acc_z = timepoint["ACC_Z"]

                            # add acceleration in x, y, and z direction to the dataframe we created
                            CEAP_data.loc[
                                start_point + (video - 1) * number_timepoints + t,
                                "acc_x",
                            ] = acc_x
                            CEAP_data.loc[
                                start_point + (video - 1) * number_timepoints + t,
                                "acc_y",
                            ] = acc_y
                            CEAP_data.loc[
                                start_point + (video - 1) * number_timepoints + t,
                                "acc_z",
                            ] = acc_z

                        else:
                            # get physiological data and round to 3 numbers after decimal
                            value = round(timepoint[measure], 3)
                            # add physiological data to the dataframe we created
                            CEAP_data.loc[
                                start_point + (video - 1) * number_timepoints + t,
                                measure.lower(),
                            ] = value

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

        # save preprocessed data to csv file
        CEAP_data.to_csv(os.path.join(preprocessed_path, new_filename), index=False)
