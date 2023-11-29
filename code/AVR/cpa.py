########################################################################################################################
# Script to perform a change point analysis (CPA) on continuous annotation data
#
# Script includes functions to perform a CPA, to plot the results of a CPA, and to test the significance of the results
# for participants from a given dataset individually.
# If you want to perform a CPA for averaged data across participants, use the script cpa_averaged.py
# (script needs to be in the same directory as this one as it imports functions from this script).
#
# This script is based on the code used by McClay et al. (2023), adapted from their Google Collab on 08.11.2023
# Access:           https://www.nature.com/articles/s41467-023-42241-2
# Citation:         McClay, M., Sachs, M.E. & Clewett, D.
#                   Dynamic emotional states shape the episodic structure of memory. Nat Commun 14, 6533 (2023).
# OSF Page:         https://osf.io/s8g5n/
# Google Collab:    https://colab.research.google.com/drive/1msf01IgCTwi3VcDyGzx56KFjNpDouSJJ?authuser=1#scrollTo=OXrgDOpIAQvV
#
# Inputs:           preprocessed data for all timepoints and participants separately, for both valence and arousal, of a given dataset
# Outputs:          table with valence and arousal changepoints for each participant and all videos (sub_x_changepoint_data_model=x_jump=x.csv)
#                   plots of changepoints for each participant and each video, for valence and arousal separately (sub_x_changepoints_Vx_valence.pdf, sub_x_changepoints_Vx_arousal.pdf)
#
# Functions:        get_changepoints(signal, model, pen, jump) -> returns changepoints for a given signal
#                   plot_elbow(signal, model, list_penalties, jump) -> plots elbow plot to determine optimal penalty value
#                   plot_changepoints(changepoints, signal, sampling_rate, title, xlabel, ylabel) -> plots changepoints for a given signal
#                   test_changepoints(TODO: specify this) -> tests changepoints for significance
#
# Author:           Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
# Last version:     16.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ruptures as rpt

# ------------------------- SETUP ------------------------------
# datasets to perform CPA on
datasets = ["CASE"]
# other possible datasets that this code is going to be tested on: (TODO) "CEAP", "AVR"

# annotation sampling rate of datasets
sampling_rates = {"CASE": 20, "CEAP": 10, "AVR": 20}

# change to where you saved the preprocessed data
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/"

# cost function model to use for CPA
model = "l2"  # we use the least squared deviation model as a default as this was also used by McClay et al.
# this model detects shifts in the mean of the signal
# other possible models to use: "l1", "normal", "rbf", "cosine", "linear", "clinear", "rank", "ml", "ar"

# penalty parameter for the cost function
# the higher the penalty, the fewer change points are detected
pen = 1

# create list of possible penalty values to test
list_penalties = list(range(11))

# jump parameter for the cost function
jump = 5
# jump controls the grid of possible change points (by default jump = 5)
# the higher jump is, the faster is the computation (at the expense of precision)


# ------------------------- FUNCTIONS ------------------------------
# function that returns the changepoints for a given dataset
def get_changepoints(signal: np.ndarray, model: str, pen: int, jump: int) -> list:
    # .fit() takes a signal as input and fits the algorithm to the data
    algorithm = rpt.Pelt(model=model, jump=jump).fit(signal)
    # other algorithms to use are: "Dynp", "Binseg", "Window"

    # .predict() returns a list of indexes corresponding to the end of each regime
    result = algorithm.predict(pen=pen)
    # by design, the last element of this list is the number of samples

    # store the changepoint timestamps
    changepoints = [int(change) for change in result]

    return changepoints


# function that creates an elbow plot to determine the optimal penalty
def plot_elbow(signal: np.ndarray, model: str, list_penalties: list[int], jump: int):
    # create an empty list to store the number of changepoints for each penalty value
    list_number_of_changepoints = []
    # perform a cpa for each possible penalty value specified in list_penalties
    for penalty_value in list_penalties:
        changepoints = get_changepoints(signal, model, penalty_value, jump)
        number_of_changepoints = len(changepoints)
        list_number_of_changepoints.append(number_of_changepoints)

    # plot elbow plot
    plt.figure(figsize=(12, 6))
    plt.plot(list_penalties, list_number_of_changepoints)
    plt.xlabel("Penalty")
    plt.ylabel("Number of changepoints")
    # show plot
    #plt.show()


# function that plots results of changepoint analysis
def plot_changepoints(
    changepoints: list[int],  # changepoints need to be in seconds
    signal: np.ndarray,  # signal needs to be in seconds
    sampling_rate: int,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt.figure(figsize=(12, 6))
    # plot annotation data
    plt.plot(signal)

    # TODO: make plot prettier? e.g. with shaded areas instead of lines for changepoints? using seaborn?

    # plot changepoints as red vertical lines
    for changepoint in changepoints:
        plt.axvline(changepoint, color="r", linestyle="--")

    # set ticks to seconds
    x_ticks = plt.xticks()[0]
    plt.xticks(x_ticks, [int((xtick / sampling_rate)) for xtick in x_ticks])

    # set limits of x-axis
    plt.xlim(0, len(signal) + 100)
    # set limits of y-axis
    plt.ylim(-1, 1)

    # add title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # add number of changepoints
    plt.text(
        5 / 6 * len(signal),
        0.9,
        f"Number of changepoints: {len(changepoints)}",
        ha="center",
        va="center",
        size=12,
        color="r",
        bbox=dict(facecolor=(1, 1, 1, 0.8), edgecolor="r"),
    )


# function that tests changepoints for significance
def test_changepoints():
    # TODO: write this function
    return ...


# ------------------------- MAIN ------------------------------
if __name__ == "__main__":
    # loop over dataset
    for dataset in datasets:
        # set data path
        datapath_set = os.path.join(datapath, "Phase1/{}/data/".format(dataset))
        # set result path
        resultpath_set = os.path.join(datapath, "Phase1/{}/results/cpa/".format(dataset))
        # create result folder if it doesn't exist yet
        if not os.path.exists(resultpath_set):
            os.makedirs(resultpath_set)

        if dataset == "CASE":
            # CASE dataset has 30 participants
            subjects = [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
            ]

            # create empty list to store changepoints
            changepoints_all = []

            # loop over subjects
            for subject in subjects:
                print("Processing subject ", subject, " of ", str(len(subjects)), "...")

                # read in annotation data from excel file
                annotation_data = pd.read_csv(
                    os.path.join(
                        datapath_set, f"preprocessed/sub_{subject}_preprocessed.csv"
                    )
                )

                # group data by video
                grouped_data = annotation_data.groupby("video")

                # create empty list to store changepoints
                changepoint_data = []

                # create subject result folder
                resultpath_subject = os.path.join(resultpath_set, f"sub_{subject}")
                if f"sub_{subject}" not in os.listdir(resultpath_set):
                    os.mkdir(resultpath_subject)
                else:
                    print("Result folder already exists. Caution: Files might be overwritten.")

                # loop over videos
                for video, group_data in grouped_data:
                    valence_data = group_data["cr_v"].values
                    arousal_data = group_data["cr_a"].values

                    # reshape data to fit the input format of the algorithm
                    valence_data = np.array(valence_data).reshape(-1, 1)
                    arousal_data = np.array(arousal_data).reshape(-1, 1)

                    # ELBOW PLOT TO DETERMINE OPTIMAL PENALTY VALUE
                    # if you've already decided which penalty value to use or if you want to use the same penalty value for all videos,
                    # comment out the following lines and change the pen value at the top of the script
                    """
                    # plot elbow plot to determine the optimal penalty value for valence data
                    plot_elbow(valence_data, model, list_penalties, jump)
                    # ask for input of the best penalty value to use for subsequent analysis of valence data
                    valence_pen = int(input("Please enter the penalty value you want to use for valence and press Enter: "))
                    plt.close()

                    # plot elbow plot to determine the optimal penalty value for arousal data
                    plot_elbow(arousal_data, model, list_penalties, jump)
                    # ask for input of the best penalty value to use for subsequent analysis of arousal data
                    arousal_pen = int(input("Please enter the penalty value you want to use for arousal and press Enter: "))
                    plt.close()
                    """

                    # perform changepoint analysis on valence
                    valence_changepoints = get_changepoints(
                        valence_data, model, pen, jump
                    )
                    # change third parameter to valence_pen when using elbow method and pen when not
                    # delete last element of the list (which is the number of samples)
                    valence_changepoints.pop()

                    # perform changepoint analysis on arousal
                    arousal_changepoints = get_changepoints(
                        arousal_data, model, pen, jump
                    )
                    # change third parameter to arousal_pen when using elbow method and pen when not
                    # delete last element of the list (which is the number of samples)
                    arousal_changepoints.pop()

                    # visualize changepoints for valence
                    plot_changepoints(
                        valence_changepoints,
                        valence_data,
                        sampling_rates[dataset],
                        f"Changepoint Analysis for Valence (Subject: {subject}, Video: {video})",
                        "Time (seconds)",
                        "Valence",
                    )
                    # show plot
                    # plt.show()

                    # save plot to subject result folder
                    plt.savefig(
                        os.path.join(
                            resultpath_subject,
                            f"sub_{subject}_changepoints_V{video}_valence.pdf",
                        )
                    )
                    plt.close()

                    # visualize changepoints for arousal
                    plot_changepoints(
                        arousal_changepoints,
                        arousal_data,
                        sampling_rates[dataset],
                        f"Changepoint Analysis for Arousal (Subject: {subject}, Video: {video})",
                        "Time (seconds)",
                        "Arousal",
                    )
                    # show plot
                    # plt.show()

                    # save plot to subject result folder
                    plt.savefig(
                        os.path.join(
                            resultpath_subject,
                            f"sub_{subject}_changepoints_V{video}_arousal.pdf",
                        )
                    )
                    plt.close()

                    # convert changepoints to seconds (rounded to two decimals)
                    valence_changepoints_seconds = [
                        round((changepoint / sampling_rates[dataset]), 2)
                        for changepoint in valence_changepoints
                    ]
                    arousal_changepoints_seconds = [
                        round((changepoint / sampling_rates[dataset]), 2)
                        for changepoint in arousal_changepoints
                    ]

                    # add changepoints to changepoint_data
                    changepoint_data.append(
                        {
                            "subject": subject,
                            "video": video,
                            "valence_changepoints": valence_changepoints_seconds,
                            "number_valence_changepoints": len(
                                valence_changepoints_seconds
                            ),
                            "arousal_changepoints": arousal_changepoints_seconds,
                            "number_arousal_changepoints": len(
                                arousal_changepoints_seconds
                            ),
                            "model": model,
                            "jump_value": jump,
                            "penalty_value": pen,
                        }
                    )

                # create dataframe from changepoint_data
                changepoint_df = pd.DataFrame(changepoint_data)

                # add changepoint_df to changepoints_all
                changepoints_all.append(changepoint_df)

                # display the changepoint dataframe
                # print(changepoint_df)

                # save changepoint dataframe to csv
                # change name to include the two parameters model & jump (so that when we test different values, we save different files)
                changepoint_df.to_csv(
                    os.path.join(
                        resultpath_subject,
                        f"sub_{subject}_changepoint_data_model={model}_jump={jump}.csv",
                    ),
                    index=False,
                )

            # create dataframe from changepoints_all
            changepoints_all_df = pd.concat(changepoints_all)

            # create new folder for all changepoints
            resultpath_set_all = os.path.join(resultpath_set, "all")
            if resultpath_set_all not in os.listdir(resultpath_set):
                os.mkdir(resultpath_set_all)

            # save changepoints_all_df to csv
            changepoints_all_df.to_csv(
                os.path.join(
                    resultpath_set_all,
                    f"changepoint_data_model={model}_jump={jump}.csv",
                ),
                index=False,
            )

        else:
            print("Dataset not available.")
            continue
