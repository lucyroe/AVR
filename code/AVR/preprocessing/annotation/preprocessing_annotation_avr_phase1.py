########################################################################################################################
# Script to preprocess CR data from Affective VR phase 1
#
# Inputs:       Raw data from Affective VR phase 1
# Outputs:      Preprocessed data from AVR data for each participant as csv files in "Preprocessed" directory
#
# Author:       Lucy Roellecke (lucy.roellecke[at]tuta.com)
#
# Last version: 27.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

# ------------------------- SETUP ------------------------------
# change the data_path to the path where you saved the AVR data
main_path = (
    "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/AffectiveVR/data/"
)
preprocessed_path = main_path + "preprocessed/"

# create new directory if folder does not exist yet
if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

# Set experimental parameters
blocks = ["Practice", "Experiment", "Assessment"]
logging_freq = ["CR", "SR"]
test_site = ["Torino", "Berlin"]  # Torino (even SJ nb) = 0, Berlin (odd SJ nb) = 1
rating_methods = ["Grid", "Flubber", "Proprioceptive", "Baseline"]
quadrants = ["HP", "LP", "LN", "HN"]
questionnaire = ["SUS", "invasive_presence", "Kunin"]
frequency = 1 / 0.05  # sampling frequency CR in Hz
video_length = 60  # length of the videos in s
session = "S000"  # session of recording

# get list of participants
subject_list = os.listdir(main_path + "AVR/")
# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if ".DS_Store" in subject_list:
    subject_list.remove(".DS_Store")
# sort participants in ascending order
subject_list.sort()
# exclude participants because no Grid CR log recorded
to_be_removed = {"01", "02", "03", "04", "05", "07", "09", "11", "13", "15", "17"}
subject_list = [item for item in subject_list if item not in to_be_removed]

# ------------------------- MAIN ------------------------------
# read and preprocess data
# add preprocessed data to dataframe
if __name__ == "__main__":
    for subject, subject_id in enumerate(subject_list):
        print("Processing subject " + subject_id + " ...")

        # initialize variables
        sj_id = []
        site = []
        rating_method = []
        quadrant = []
        cr_v = []
        cr_a = []
        cr_dist = []
        cr_angle = []
        time = []

        # read results in csv file
        subject_path = main_path + "AVR/" + subject_id + "/" + session + "/"
        trial_results_filename = subject_path + "trial_results.csv"
        trials_results = pd.read_csv(trial_results_filename)

        # loop over rating methods
        for index, rat_method in enumerate(rating_methods):
            # select data according to rating method
            trials_rating_method = trials_results[
                trials_results["method_descriptor"] == rat_method
            ]
            # select experiment data
            trials_experiment = trials_rating_method[
                trials_rating_method["block_name"] == "Experiment"
            ]

            # lpop over quadrants/videos
            for video_index, video in enumerate(quadrants):
                # get CR from file
                # rating methods other than Baseline
                if rat_method != "Baseline":
                    # location of CR file
                    cr_location = "rating_" + rat_method + "_CR_location_0"
                    cr_path = (
                        main_path
                        + trials_experiment.loc[
                            trials_experiment["quadrant_descriptor"] == video,
                            cr_location,
                        ].item()
                    )
                    cr = pd.read_csv(cr_path)  # load CR file
                    # rescale time from 0 to end of video
                    cr["time"] = cr["time"] - cr["time"][0]
                    # a little bit of cleaning name of column
                    cr.rename(columns={"arousal ": "arousal"}, inplace=True)
                    cr_valence = cr["valence"].values  # valence
                    cr_arousal = cr["arousal"].values  # arousal
                    cr_distance = np.hypot(cr_valence, cr_arousal)  # distance
                    cr_ang = np.arctan2(cr_arousal, cr_valence)  # angle
                    cr_time = cr["time"].values  # times of samples
                    number_cr = cr.__len__()  # number of samples

                    # resampling
                    cr_time = np.arange(
                        1 / frequency, video_length + 1 / frequency, 1 / frequency
                    )
                    cr_valence_interpolated = interp1d(
                        cr["time"],
                        cr_valence,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(cr_valence[0], cr_valence[-1]), # type: ignore
                    )
                    cr_arousal_interpolated = interp1d(
                        cr["time"],
                        cr_arousal,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(cr_arousal[0], cr_arousal[-1]), # type: ignore
                    )
                    cr_distance_interpolated = interp1d(
                        cr["time"],
                        cr_distance,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(cr_distance[0], cr_distance[-1]), # type: ignore
                    )
                    cr_angle_interpolated = interp1d(
                        cr["time"],
                        cr_ang,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(cr_ang[0], cr_ang[-1]), # type: ignore
                    )
                    cr_valence = cr_valence_interpolated(cr_time)
                    cr_arousal = cr_arousal_interpolated(cr_time)
                    cr_distance = cr_distance_interpolated(cr_time)
                    cr_ang = cr_angle_interpolated(cr_time)

                    number_samples = cr_time.__len__()  # number of samples
                    # save CR data of this trial in variables
                    sj_id = np.append(sj_id, np.tile(subject_id, number_samples))
                    rating_method = np.append(
                        rating_method, np.tile(rat_method, number_samples)
                    )
                    quadrant = np.append(quadrant, np.tile(video, number_samples))
                    # sjs with even number are in the first testing site, odd in the second
                    if int(subject_id) % 2 == 0:
                        site = np.append(site, np.tile(test_site[0], number_samples))
                    else:
                        site = np.append(site, np.tile(test_site[1], number_samples))

                    cr_v = np.append(cr_v, cr_valence)
                    cr_a = np.append(cr_a, cr_arousal)
                    cr_dist = np.append(cr_dist, cr_distance)
                    cr_angle = np.append(cr_angle, cr_ang)
                    time = np.append(time, cr_time)

        # create dataframe with CR data
        data_cr = {
            "sj_id": sj_id,
            "test_site": site,
            "rating_method": rating_method,
            "quadrant": quadrant,
            "cr_v": cr_v,
            "cr_a": cr_a,
            "cr_dist": cr_dist,
            "cr_angle": cr_angle,
            "time": time,
        }
        dataframe_cr = pd.DataFrame(data=data_cr)

        # round all values to two decimals
        dataframe_cr["time"] = round(dataframe_cr["time"], 2)

        # save dataframe in csv file
        filename_cr = preprocessed_path + f"/sub_{subject_id}_cr_preprocessed.csv"
        dataframe_cr.to_csv(filename_cr, na_rep="NaN", index=False)
