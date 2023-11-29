########################################################################################################################
# Script to plot CR data from Affective VR phase 2 or 3 (as a visual check whether the data is clean)

# Inputs:       Raw or preprocessed data from Affective VR phase 2 or 3
# Outputs:      Descriptive plots of individual timeseries of CR data
#
# Author:       Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
# Last version: 27.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import os
import matplotlib.pyplot as plt

# ------------------------- SETUP ------------------------------
# change the data_path to the path where you saved the preprocessed AVR data
main_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase2/data/preprocessed/"
resultpath = (
    "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase2/results/"
)

# Set experimental parameters
test_site = ["Berlin"]  # Torino (even SJ nb) = 0, Berlin (odd SJ nb) = 1
# "Torino"
frequency = 1 / 0.05  # sampling frequency CR in Hz
video_length = (
    90 * 4 + 253 + 390 + 381
)  # length of the sequence of videos in s: 4xScifi + Invasion + Asteroids + Underwood
session = "S000"  # session of recording

# get list of preprocessed participants files
subject_list = os.listdir(main_path)
# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if ".DS_Store" in subject_list:
    subject_list.remove(".DS_Store")
# exclude any files that are not individual participant's data
for item in subject_list:
    if item.startswith("sub_") is False:
        subject_list.remove(item)
# sort participants in ascending order
subject_list.sort()

# ------------------------- MAIN ------------------------------
# read data and plot timeseries
if __name__ == "__main__":
    # loop over all participants
    for index, subject_file in enumerate(subject_list):
        subject = subject_file.split("_")[1]
        print("Plotting the data of subject " + subject + " ...")

        # read data
        data = pd.read_csv(main_path + subject_file)

        # set result path
        resultpath_subject = os.path.join(resultpath, "sub_" + subject)
        # create folder for results of subject
        if not os.path.exists(resultpath_subject):
            os.makedirs(resultpath_subject)

        # get valence and arousal ratings
        valence_data = data["cr_v"]
        arousal_data = data["cr_a"]

        # plot valence data
        plt.figure(figsize=(12, 6))
        plt.plot(valence_data)

        # set ticks to minutes
        x_ticks = plt.xticks()[0]
        plt.xticks(x_ticks, [int((xtick / frequency) / 60) for xtick in x_ticks])

        # set limits of x-axis
        plt.xlim(0, len(valence_data) + 100)
        # set limits of y-axis
        plt.ylim(-1.5, 1.5)

        # add title and axis labels
        plt.title("Valence Ratings over Time for Subject " + subject)
        plt.xlabel("Time in minutes")
        plt.ylabel("Valence")

        # show plot
        # plt.show()

        # save plot to subject result folder
        plt.savefig(
            os.path.join(
                resultpath_subject,
                f"sub_{subject}_valence.pdf",
            )
        )

        plt.close()

        # plot arousal data
        plt.figure(figsize=(12, 6))
        plt.plot(arousal_data)

        # set ticks to minutes
        x_ticks = plt.xticks()[0]
        plt.xticks(x_ticks, [int((xtick / frequency) / 60) for xtick in x_ticks])

        # set limits of x-axis
        plt.xlim(0, len(arousal_data) + 100)
        # set limits of y-axis
        plt.ylim(-1.5, 1.5)

        # add title and axis labels
        plt.title("Arousal Ratings over Time for Subject " + subject)
        plt.xlabel("Time in minutes")
        plt.ylabel("Arousal")

        # show plot
        # plt.show()

        # save plot to subject result folder
        plt.savefig(
            os.path.join(
                resultpath_subject,
                f"sub_{subject}_arousal.pdf",
            )
        )

        plt.close()

    # plot valence and arousal data averaged across all participants
    # get valence and arousal ratings
    print("Plotting the data averaged across all subjects ...")

    # read data
    data = pd.read_csv(main_path + "mean_cr_preprocessed_rs.csv")
    valence_data_all = data["cr_v"]
    arousal_data_all = data["cr_a"]

    # plot valence data
    plt.figure(figsize=(12, 6))
    plt.plot(valence_data_all)

    # set ticks to minutes
    x_ticks = plt.xticks()[0]
    plt.xticks(x_ticks, [int((xtick / frequency) / 60) for xtick in x_ticks])

    # set limits of x-axis
    plt.xlim(0, len(valence_data_all) + 100)
    # set limits of y-axis
    plt.ylim(-1, 1)

    # add title and axis labels
    plt.title("Valence Ratings over Time averaged across N = " + str(len(subject_list)) + " Subjects")
    plt.xlabel("Time in minutes")
    plt.ylabel("Valence")

    # show plot
    # plt.show()

    # save plot to result folder
    plt.savefig(
        os.path.join(
            resultpath,
            "valence_avg.pdf",
        )
    )

    plt.close()

    # plot arousal data
    plt.figure(figsize=(12, 6))
    plt.plot(arousal_data_all)

    # set ticks to minutes
    x_ticks = plt.xticks()[0]
    plt.xticks(x_ticks, [int((xtick / frequency) / 60) for xtick in x_ticks])

    # set limits of x-axis
    plt.xlim(0, len(arousal_data_all) + 100)
    # set limits of y-axis
    plt.ylim(-1, 1)

    # add title and axis labels
    plt.title("Arousal Ratings over Time averaged across N = " + str(len(subject_list)) + " Subjects")
    plt.xlabel("Time in minutes")
    plt.ylabel("Arousal")

    # show plot
    # plt.show()

    # save plot to result folder
    plt.savefig(
        os.path.join(
            resultpath,
            "arousal_avg.pdf",
        )
    )

    plt.close()
