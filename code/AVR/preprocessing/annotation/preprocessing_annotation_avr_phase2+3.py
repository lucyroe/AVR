########################################################################################################################
# Script to preprocess CR data from Affective VR phase 2 or 3

# Inputs:       Raw data from Affective VR phase 2 or 3
# Outputs:      Preprocessed data from AVR data for each participant as csv files in "Preprocessed" directory
#
# Author:       Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
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
    "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase2/data/"
)
preprocessed_path = main_path + "preprocessed/"

# create new directory if folder does not exist yet
if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

# Set experimental parameters
blocks = ["Practice", "Experiment"]
logging_freq = ["CR", "SR"]
test_site = ["Torino", "Berlin"]  # Torino (even SJ nb) = 0, Berlin (odd SJ nb) = 1
questionnaire = ["SUS", "invasive_presence", "Kunin"]
frequency = 1 / 0.05  # sampling frequency CR in Hz
video_length = (
    90 * 4 + 253 + 390 + 381
)  # length of the sequence of videos in s: 4xScifi + Invasion + Asteroids + Underwood
session = "S000"  # session of recording
resample = True  # resample CR to cr_fs (if samples are not even). Note: does not interpolate NA values

# get list of participants
subject_list = os.listdir(main_path + "AVR/")
# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if ".DS_Store" in subject_list:
    subject_list.remove(".DS_Store")
# sort participants in ascending order
subject_list.sort()

# ------------------------- MAIN ------------------------------
# read and preprocess data
# add preprocessed data to dataframe
if __name__ == "__main__":

    # create empty list for data of all participants
    list_data_all = []
    
    for subject, subject_id in enumerate(subject_list):
        print("Processing subject " + subject_id + " ...")

        # initialize variables
        sj_id = []
        site = []
        cr_v = []
        cr_a = []
        cr_dist = []
        cr_angle = []
        time = []

        # read results in csv file
        subject_path = main_path + "AVR/" + subject_id + "/" + session + "/"
        trial_results_filename = subject_path + "trial_results.csv"
        trials_results = pd.read_csv(trial_results_filename)

        # select experiment data
        trials_experiment = trials_results[trials_results["block_name"] == "Experiment"]

        # get CR from file
        # location of CR file
        cr_location = 'rating_CR_location_0'
        cr_path = os.path.join(main_path, trials_experiment[cr_location].iloc[0])
        cr = pd.read_csv(cr_path)  # load CR file
        # rescale time from 0 to end of video
        cr["time"] = cr["time"] - cr["time"][0]
        # a little bit of cleaning name of column
        cr.rename(columns={"arousal ": "arousal"}, inplace=True)
        cr_valence = np.array(cr["valence"].values)  # valence
        cr_arousal = np.array(cr["arousal"].values)  # arousal
        cr_distance = np.hypot(cr_valence, cr_arousal)  # distance
        cr_ang = np.arctan2(cr_arousal, cr_valence)  # angle
        cr_time = np.array(cr["time"].values)  # times of samples
        number_cr = cr.__len__()  # number of samples

        if resample:
            # TODO: decide how to do resampling and adapt code
            cr_time = np.arange(
                1 / frequency, video_length + 1 / frequency, 1 / frequency
            )
            cr_valence_interpolated = interp1d(
                cr["time"],
                cr_valence,
                kind="linear",
                bounds_error=False,
                fill_value=(cr_valence[0], cr_valence[-1]),  # type: ignore
            )
            cr_arousal_interpolated = interp1d(
                cr["time"],
                cr_arousal,
                kind="linear",
                bounds_error=False,
                fill_value=(cr_arousal[0], cr_arousal[-1]),  # type: ignore
            )
            cr_distance_interpolated = interp1d(
                cr["time"],
                cr_distance,
                kind="linear",
                bounds_error=False,
                fill_value=(cr_distance[0], cr_distance[-1]),  # type: ignore
            )
            cr_angle_interpolated = interp1d(
                cr["time"],
                cr_ang,
                kind="linear",
                bounds_error=False,
                fill_value=(cr_ang[0], cr_ang[-1]),  # type: ignore
            )
            cr_valence = cr_valence_interpolated(cr_time)
            cr_arousal = cr_arousal_interpolated(cr_time)
            cr_distance = cr_distance_interpolated(cr_time)
            cr_ang = cr_angle_interpolated(cr_time)

        number_samples = cr_time.__len__()  # number of samples

        # save CR data of this trial in variables
        sj_id = np.append(sj_id, np.tile(subject_id, number_samples))

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
        if resample:
            filename_cr = (
                preprocessed_path + f"/sub_{subject_id}_cr_preprocessed_rs.csv"
            )
        else:
            filename_cr = preprocessed_path + f"/sub_{subject_id}_cr_preprocessed.csv"

        dataframe_cr.to_csv(filename_cr, na_rep="NaN", index=False)

        # add participant's data to dataframe for all participants
        list_data_all.append(dataframe_cr)
    
    # concatenate data of all participants into one dataframe
    data_all = pd.concat(list_data_all, ignore_index=True)

    # save dataframe with all participants in csv file
    if resample:
        filename_all = preprocessed_path + "/all_cr_preprocessed_rs.csv"
    else:
        filename_all = preprocessed_path + "/all_cr_preprocessed.csv"
    
    data_all.to_csv(filename_all, na_rep="NaN", index=False)

    # calculate mean CR data
    data_mean = data_all.groupby("time").mean()
    # add time column
    data_mean["time"] = list_data_all[0]["time"].values

    # TODO: this does not work when the data is not resampled
    # different timepoints for different participants
    # cannot be averaged

    # save dataframe with mean CR data in csv file
    if resample:
        filename_mean = preprocessed_path + "/mean_cr_preprocessed_rs.csv"
    else:
        filename_mean = preprocessed_path + "/mean_cr_preprocessed.csv"
    
    data_mean.to_csv(filename_mean, na_rep="NaN", index=False)
