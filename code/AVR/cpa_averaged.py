########################################################################################################################
# Script to perform a change point analysis (CPA) on continuous annotation data
#
# Script performs a CPA for averaged data across participants.
# If you want to perform a CPA for participants individually, use the script cpa.py
# (this script needs to be in the same directory as cpa.py as it imports functions from this script).
#
# Both scripts are based on the code used by McClay et al. (2023), adapted from their Google Collab on 08.11.2023
# Access:           https://www.nature.com/articles/s41467-023-42241-2
# Citation:         McClay, M., Sachs, M.E. & Clewett, D.
#                   Dynamic emotional states shape the episodic structure of memory. Nat Commun 14, 6533 (2023).
# OSF Page:         https://osf.io/s8g5n/
# Google Collab:    https://colab.research.google.com/drive/1msf01IgCTwi3VcDyGzx56KFjNpDouSJJ?authuser=1#scrollTo=OXrgDOpIAQvV
#
# Inputs:           preprocessed data for all timepoints and participants, for both valence and arousal, of a given dataset
# Outputs:          table with valence and arousal changepoints for all videos averaged across participants (changepoint_data_model=x_jump=x.csv)
#                   plots of changepoints for each video, for valence and arousal separately (changepoints_Vx_valence.pdf, changepoints_Vx_arousal.pdf)
#
# Functions:        get_changepoints(signal, model, pen, jump) -> returns changepoints for a given signal
#                   plot_elbow(signal, model, list_penalties, jump) -> plots elbow plot to determine optimal penalty value
#                   plot_changepoints(changepoints, signal, sampling_rate, title, xlabel, ylabel) -> plots changepoints for a given signal
#                   test_changepoints(TODO: specify this) -> tests changepoints for significance
#
# Author:           Lucy Roellecke (lucy.roellecke[at]fu-berlin.de)
# Last version:     22.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES -------------------------
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import ast
from cpa import (
    get_changepoints,
    plot_changepoints,
)  # cpa.py and cpa_averaged.py need to be in the same directory!

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
                "30"
            ]

            # create color maps for plots
            number_colors = len(subjects)
            colormaps = [matplotlib.colors.ListedColormap(plt.get_cmap('gist_rainbow')(np.linspace(0,1,number_colors)))] # type: ignore

            # create empty list to store data from all participants
            grouped_data_all = []

            # loop over subjects
            for subject in subjects:
                # read in annotation data from excel file
                annotation_data = pd.read_csv(
                    os.path.join(
                        datapath_set, f"preprocessed/sub_{subject}_preprocessed.csv"
                    )
                )

                # create a unique identifier for each row within each video
                annotation_data["row_id"] = annotation_data.groupby("video").cumcount()

                # add grouped data to list
                grouped_data_all.append(annotation_data)

            # concatenate all dataframes in the list
            concatenated_data = pd.concat(grouped_data_all)

            # group data by video and row_id
            grouped_data = concatenated_data.groupby("video")
            # group data by video and row_id, then calculate mean
            grouped_data_avg = concatenated_data.groupby(["video", "row_id"]).mean()
            
            # create empty list to store changepoints
            changepoint_data_avg = []

           
            # ---------------------- PLOT AVERAGED CPs + AVERAGED DATA --------------------------
            # loop over videos
            for video, group_data in grouped_data_avg.groupby("video"):
                valence_data = group_data["cr_v"].values
                arousal_data = group_data["cr_a"].values

                # reshape data to fit the input format of the algorithm
                valence_data = np.array(valence_data).reshape(-1, 1)
                arousal_data = np.array(arousal_data).reshape(-1, 1)

                # ELBOW PLOT TO DETERMINE OPTIMAL PENALTY VALUE
                # if you've already decided which penalty value to use or if you want to use the same penalty value for all videos,
                # comment out the following lines and change the pen value at the top of the script
                '''
                # plot elbow plot to determine the optimal penalty value for valence data
                plot_elbow(valence_data, model, list_penalties, jump)
                # ask for input of the best penalty value to use for subsequent analysis of valence data
                valence_pen = int(input("Please enter the penalty value you want to use for valence and press Enter: "))
                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set,
                        f"elbow_V{video}_valence.pdf",
                    )
                )
                plt.close()
                
                # plot elbow plot to determine the optimal penalty value for arousal data
                plot_elbow(arousal_data, model, list_penalties, jump)
                # ask for input of the best penalty value to use for subsequent analysis of arousal data
                arousal_pen = int(input("Please enter the penalty value you want to use for arousal and press Enter: "))
                                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set,
                        f"elbow_V{video}_arousal.pdf",
                    )
                )
                plt.close()
                '''

                # perform changepoint analysis on valence
                valence_changepoints = get_changepoints(valence_data, model, pen, jump)
                # change third parameter to valence_pen when using elbow method and pen when not
                # delete last element of the list (which is the number of samples)
                valence_changepoints.pop()

                # perform changepoint analysis on arousal
                arousal_changepoints = get_changepoints(arousal_data, model, pen, jump)
                # change third parameter to arousal_pen when using elbow method and pen when not
                # delete last element of the list (which is the number of samples)
                arousal_changepoints.pop()

                # visualize changepoints for valence
                plot_changepoints(
                    valence_changepoints,
                    valence_data,
                    sampling_rates[dataset],
                    f"Changepoint Analysis for Valence Averaged across Participants (Video: {video})",
                    "Time (seconds)",
                    "Valence",
                )
                # show plot
                # plt.show()

                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set, "avg",
                        f"changepoints_V{video}_valence_avg.pdf",
                    )
                )
                plt.close()

                # visualize changepoints for arousal
                plot_changepoints(
                    arousal_changepoints,
                    arousal_data,
                    sampling_rates[dataset],
                    f"Changepoint Analysis for Arousal Averaged across Participants (Video: {video})",
                    "Time (seconds)",
                    "Arousal",
                )
                # show plot
                # plt.show()

                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set, "avg",
                        f"changepoints_V{video}_arousal_avg.pdf",
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
                changepoint_data_avg.append(
                    {
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
            
            # ---------------------- PLOT AVERAGED CPs + INDIVIDUAL DATA --------------------------
            # loop over videos
            for video, group_data in grouped_data:
                
                # create empty list to store valence data for each subject
                valence_data_all = []
                # create empty list to store arousal data for each subject
                arousal_data_all = []

                # get annotation data for each subject
                for subject, subject_data in group_data.groupby("sj_id"):

                    valence_data = subject_data["cr_v"].values
                    arousal_data = subject_data["cr_a"].values

                    # reshape data to fit the input format of the algorithm
                    valence_data = np.array(valence_data).reshape(-1, 1)
                    arousal_data = np.array(arousal_data).reshape(-1, 1)

                    # add valence data to list
                    valence_data_all.append(valence_data)
                    # add arousal data to list
                    arousal_data_all.append(arousal_data)
                
                # get averaged changepoints
                for data in changepoint_data_avg:
                    if data["video"] == video:
                        valence_changepoints = data["valence_changepoints"]
                        arousal_changepoints = data["arousal_changepoints"]
                
                # get averaged timeseries
                for video_avg, data in grouped_data_avg.groupby("video"):
                    if video_avg == video:
                        valence_data_avg = data["cr_v"].values
                        arousal_data_avg = data["cr_a"].values
                
                plt.figure(figsize=(12, 6))
                # plot valence data for each subject in the same plot
                for index, subject in enumerate(subjects):
                    # plot valence data of subject
                    plt.plot(valence_data_all[index], label=subject, alpha=0.5)
                
                sampling_rate = sampling_rates[dataset]

                # plot averaged timeseries
                plt.plot(valence_data_avg, label="mean", color="black", linewidth=2)
                
                # visualize changepoints for valence averaged across participants
                for valence_changepoint in valence_changepoints:
                    plt.axvline(valence_changepoint*sampling_rate, color="black", linestyle="--", linewidth=2, label="cp")

                # add legend
                legend = subjects + ["mean"] + ["cp"]
                plt.legend(legend, fontsize='x-small')

                # set ticks to seconds
                x_ticks = plt.xticks()[0]
                plt.xticks(x_ticks, [int((xtick / sampling_rate)) for xtick in x_ticks])

                # set limits of x-axis
                plt.xlim(0, len(valence_data_all[0]) + 350)
                # set limits of y-axis
                plt.ylim(-1, 1)

                # add title and axis labels
                plt.title(f"Changepoint Analysis for Valence (Video: {video})")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Valence")

                # show plot
                # plt.show()

                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set, "all_data",
                        f"changepoints_V{video}_valence_avg_all_data.pdf",
                    )
                )
                plt.close()

                plt.figure(figsize=(12, 6))
                
                # plot arousal data for each subject in the same plot
                for index, subject in enumerate(subjects):
                    # plot arousal data of subject
                    plt.plot(arousal_data_all[index], label=subject, alpha=0.5)
                
                # plot averaged timeseries
                plt.plot(arousal_data_avg, label="mean", color="black", linewidth=2)

                # visualize changepoints for arousal averaged across participants
                for arousal_changepoint in arousal_changepoints:
                    plt.axvline(arousal_changepoint*sampling_rate, color="black", linestyle="--", linewidth=2, label="cp")

                # add legend
                plt.legend(legend, fontsize='x-small')

                # set ticks to seconds
                x_ticks = plt.xticks()[0]
                plt.xticks(x_ticks, [int((xtick / sampling_rate)) for xtick in x_ticks])

                # set limits of x-axis
                plt.xlim(0, len(arousal_data_all[0]) + 350)
                # set limits of y-axis
                plt.ylim(-1, 1)

                # add title and axis labels
                # add title and axis labels
                plt.title(f"Changepoint Analysis for Arousal (Video: {video})")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Arousal")

                # show plot
                # plt.show()

                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set, "all_data",
                        f"changepoints_V{video}_arousal_avg_all_data.pdf",
                    )
                )
                plt.close()
            
           
            
            # ----------------------- PLOT INDIVIDUAL CPs + AVERAGED DATA ------------------------
            # get individual changepoints
            changepoints_all = pd.read_csv(
                os.path.join(
                    resultpath_set, "all",
                    f"changepoint_data_model={model}_jump={jump}.csv"
                )
            )

            # get averaged changepoints
            changepoints_avg = pd.read_csv(
                os.path.join(
                    resultpath_set, "avg",
                    f"changepoint_data_model={model}_jump={jump}_avg.csv"
                )
            )

            # loop over videos
            for video, group_data in grouped_data_avg.groupby("video"):
                valence_data = group_data["cr_v"].values
                arousal_data = group_data["cr_a"].values

                # reshape data to fit the input format of the algorithm
                valence_data = np.array(valence_data).reshape(-1, 1)
                arousal_data = np.array(arousal_data).reshape(-1, 1)

                # get changepoints of all participants for that video
                changepoints_video = changepoints_all.loc[changepoints_all["video"] == video]
                valence_changepoints = changepoints_video["valence_changepoints"]
                arousal_changepoints = changepoints_video["arousal_changepoints"]

                # visualize changepoints for valence
                plt.figure(figsize=(12, 6))

                sampling_rate = sampling_rates[dataset]
    
                # plot averaged timeseries
                plt.plot(valence_data, label="mean", color="grey", linewidth=1)

                # create empty list for legend entries
                legend_subjects = []
                # plot changepoints for each subject in the same plot
                for index, subject in enumerate(subjects):
                    # Check if the DataFrame is empty
                    if not valence_changepoints.empty:
                        # Reset the index of the DataFrame
                        valence_changepoints = valence_changepoints.reset_index(drop=True)
                        for changepoint in ast.literal_eval(valence_changepoints[index]):
                            plt.axvline(changepoint*sampling_rate, color=colormaps[0](index/number_colors), alpha=0.3)
                    subject_line = mlines.Line2D([], [], color=colormaps[0](index/number_colors), alpha=0.3, label='cp ' + subject)
                    legend_subjects.append(subject_line)
                
                # plot averaged changepoints
                changepoints_avg_video_valence = changepoints_avg.loc[changepoints_avg["video"] == video]["valence_changepoints"]
                # check if the series is empty
                if not changepoints_avg_video_valence.empty:
                    # Reset the index of the Series
                    changepoints_avg_video_valence = changepoints_avg_video_valence.reset_index(drop=True)
                    for changepoint_avg in ast.literal_eval(changepoints_avg_video_valence[0]):
                        plt.axvline(changepoint_avg*sampling_rate, color="black", linewidth=2, linestyle="--")
                
                # add legend
                # create Line2D instances for the legend
                cp_line = mlines.Line2D([], [], color='grey', alpha=0.5, label='cp')
                mean_legend = mlines.Line2D([], [], color='grey', lw=1, label='mean')
                avg_legend = mlines.Line2D([], [], color='black', lw=2, linestyle="--", label='cp avg')
                legend_subjects.extend([avg_legend, mean_legend])
                plt.legend(handles=legend_subjects, fontsize='x-small')

                # set ticks to seconds
                x_ticks = plt.xticks()[0]
                plt.xticks(x_ticks, [int((xtick / sampling_rate)) for xtick in x_ticks])

                # set limits of x-axis
                plt.xlim(0, len(valence_data) + 350)
                # set limits of y-axis
                plt.ylim(-1, 1)

                # add title and axis labels
                plt.title(f"Changepoint Analysis for Valence (Video: {video})")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Valence")

                # show plot
                # plt.show()

                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set, "all",
                        f"changepoints_V{video}_valence_avg.pdf",
                    )
                )
                plt.close()

                # visualize changepoints for aroual
                plt.figure(figsize=(12, 6))
     
                # plot averaged timeseries
                plt.plot(arousal_data, label="mean", color="grey", linewidth=1)
                
                # create empty list for legend entries
                legend_subjects = []
                # plot changepoints for each subject in the same plot
                for index, subject in enumerate(subjects):
                    # Check if the DataFrame is empty
                    if not arousal_changepoints.empty:
                        # Reset the index of the DataFrame
                        arousal_changepoints = arousal_changepoints.reset_index(drop=True)
                        for changepoint in ast.literal_eval(arousal_changepoints[index]):
                            plt.axvline(changepoint*sampling_rate, color=colormaps[0](index/number_colors), alpha=0.3)
                    subject_line = mlines.Line2D([], [], color=colormaps[0](index/number_colors), alpha=0.3, label='cp ' + subject)
                    legend_subjects.append(subject_line)
                
                # plot averaged changepoints
                changepoints_avg_video_arousal = changepoints_avg.loc[changepoints_avg["video"] == video]["arousal_changepoints"]
                # check if the series is empty
                if not changepoints_avg_video_arousal.empty:
                    # Reset the index of the Series
                    changepoints_avg_video_arousal = changepoints_avg_video_arousal.reset_index(drop=True)
                    for changepoint_avg in ast.literal_eval(changepoints_avg_video_arousal[0]):
                        plt.axvline(changepoint_avg*sampling_rate, color="black", linewidth=2, linestyle="--")
                
                # add legend
                # create Line2D instances for the legend
                cp_line = mlines.Line2D([], [], color='grey', alpha=0.5, label='cp')
                mean_legend = mlines.Line2D([], [], color='grey', lw=1, label='mean')
                avg_legend = mlines.Line2D([], [], color='black', lw=2, linestyle="--", label='cp avg')
                legend_subjects.extend([avg_legend, mean_legend])
                plt.legend(handles=legend_subjects, fontsize='x-small')

                # set ticks to seconds
                x_ticks = plt.xticks()[0]
                plt.xticks(x_ticks, [int((xtick / sampling_rate)) for xtick in x_ticks])

                # set limits of x-axis
                plt.xlim(0, len(arousal_data) + 350)
                # set limits of y-axis
                plt.ylim(-1, 1)

                # add title and axis labels
                plt.title(f"Changepoint Analysis for Arousal (Video: {video})")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Arousal")

                # show plot
                # plt.show()

                # save plot to result folder
                plt.savefig(
                    os.path.join(
                        resultpath_set, "all",
                        f"changepoints_V{video}_arousal_avg.pdf",
                    )
                )
                plt.close()
            
            
            # create dataframe from changepoint_data
            changepoint_df = pd.DataFrame(changepoint_data_avg)

            # display the changepoint dataframe
            # print(changepoint_df)

            # save changepoint dataframe to csv
            # change name to include the two parameters model & jump (so that when we test different values, we save different files)
            changepoint_df.to_csv(
                os.path.join(
                    resultpath_set, "avg",
                    f"changepoint_data_model={model}_jump={jump}_avg.csv",
                ),
                index=False,
            )
            
        else:
            print("Dataset not available.")
            continue
