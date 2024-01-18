"""
Script to perform a changepoint analysis (CPA) on a given time series.

Script performs a CPA for averaged data across participants.
If you want to perform a CPA for participants individually, use the script cpa.py
(this script needs to be in the same directory as cpa.py as it imports functions from this script).

Inputs: Preprocessed time series data as .csv files averaged across participants of a given dataset

Outputs:

Functions:

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: January 17th, 2024
"""

# TODO: (Status 17.01.2024)  # noqa: FIX002
# - CPA for all three datasets for annotation data averaged across participants' cps and timeseries DONE

# - repeat 1st CPA after subject 89 data is there
# - CPA for all three datasets for annotation data averaged across participants' timeseries
# - CPA for all three datasets for annotation data averaged across participants' cps
# - CPA for all three datasets for physiological data averaged across participants' timeseries
# - CPA for all three datasets for physiological data averaged across participants' cps
# - CPA for all three datasets for physiological data averaged across participants' cps and timeseries

# %% Import
import os
import re
import ast
import warnings
from pathlib import Path
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
from cpa import (
    get_changepoints,
    plot_changepoints,
)  # cpa.py and cpa_averaged.py need to be in the same directory!


warnings.filterwarnings("ignore", category=FutureWarning)   # ignore future warnings from seaborn

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# datasets to perform CPA on
datasets = ["AVR"]
# "CASE", "CEAP"

# dataset modalities
modalities = {"CASE": ["annotations"], "CEAP": ["annotations"], "AVR": ["annotations_phase2"]}
# "CASE": ["physiological"],  # noqa: ERA001
# "CEAP": ["physiological"],  # noqa: ERA001
# "AVR": ["annotations_phase1"],  # noqa: ERA001

# physiological modalities
physiological_modalities = {"CEAP": ["ibi"]}
# "CASE": ["ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"],  # noqa: ERA001
# "CEAP": ["acc_x", "acc_y", "acc_z", "bvp", "eda", "skt", "hr"]

# modalities sampling frequencies
sampling_rates = {
    "CASE": [20, 1000],
    "CEAP": [30, 1],  # physiological CEAP data -> IBI: sampling rate of 1 Hz; rest: sampling rate of 25 Hz
    "AVR": [20, 20],
}
# TODO: adjust sampling rates / downsample physiological data to match sampling rate of annotations?  # noqa: FIX002

# number of videos per dataset
number_of_videos = {"CASE": 8, "CEAP": 8, "AVR": [4, 1]}

# video to quadrant mapping for each dataset
# defines which video corresponds to which quadrant of the circumplex model of emotions
video_quadrant_mapping = {
    "CASE": {"1": "HP", "2": "HP", "3": "LN", "4": "LN", "5": "LP", "6": "LP", "7": "HN", "8": "HN"},
    "CEAP": {"1": "HP", "2": "LP", "3": "HN", "4": "LN", "5": "HP", "6": "LP", "7": "HN", "8": "LN"},
    "AVR": {"1": "HP", "2": "LP", "3": "LN", "4": "HN"},
}

# Wong color-blind safe color palette for plotting
color_palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# black, orange, sky-blue, bluish green, yellow, blue, vermillion, reddish purple
# make color palette into Matplotlib colormap
wong_colormap = LinearSegmentedColormap.from_list("Wong", color_palette, N=8)

shade_colors = [
    color_palette[2],
    color_palette[7],
]  # colors to use for shading between change points (sky-blue & reddish purple)
change_point_color = color_palette[6]  # color to use for vertical lines at change points (vermillion)
timeseries_color = color_palette[0]  # color to use for plotting the time series (black)
elbow_color = color_palette[3]  # color to use for plotting the elbow plot (bluish green)

# change to where you saved the preprocessed data
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"

# change to where you want to save the results
resultpath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/"

# analysis steps to perform
steps = ["cpa"]
# "summary statistics", "test"

# averaging modes
averaging_modes = ["all"]
# "timeseries", "changepoints"
# "all": average both across all participants' timeseries and their changepoints
# "timeseries": average only across all participants' timeseries, plot changepoints separately
# "changepoints": average only across all participants' changepoints, plot timeseries separately

# turn on debug mode (if True, only two subjects are processed)
debug = False

# %% Set CPA parameters >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# cost function model to use for CPA
model = "l2"  # we use the least squared deviation model as a default as this was also used by McClay et al.
# this model detects shifts in the mean of the signal
# other possible models to use: "l1", "normal", "rbf", "cosine", "linear", "clinear", "rank", "ml", "ar"

# penalty parameter for the cost function
# the higher the penalty, the fewer change points are detected
pen = 1

# minimum number of samples between two change points
# although the technical sampling frequency of an input device might be higher,
# humans are limited in their ability to rate their emotions on the input device
# usually, humans are able to rate their emotions at a maximum frequency of 1-2 Hz
# (see Ian D Loram et al. (2011): Human control of an inverted pendulum)
min_size = sampling_rates  # if we assume a maximal motor control frequency of 1 Hz
# results in a minimum distance between two change points of 1 second

# create list of possible penalty values to test
list_penalties = list(range(11))

# jump parameter for the cost function
jump = 5
# jump controls the grid of possible change points (by default jump = 5)
# the higher jump is, the faster is the computation (at the expense of precision)

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Loop through datasets
    for dataset in datasets:
        # Loop through modalities
        for modality_index, modality in enumerate(modalities[dataset]):
            print(f"Performing changepoint analysis for {modality} data of {dataset} dataset...")

            # Set datapath for dataset
            # Set resultpath for dataset
            if dataset == "AVR":
                datapath_dataset = Path(datapath) / modality.split("_")[1] / "preprocessed" / modality.split("_")[0]
                resultpath_dataset = Path(resultpath) / modality.split("_")[1] / "cpa" / modality.split("_")[0] / "avg"
            else:
                datapath_dataset = Path(datapath) / dataset / "preprocessed" / modality
                resultpath_dataset = Path(resultpath) / dataset / "cpa" / modality / "avg"

            # Create resultpath if it doesn't exist yet
            if not resultpath_dataset.exists():
                resultpath_dataset.mkdir(parents=True)

            # List data files
            data_files = os.listdir(datapath_dataset)
            # Remove hidden files
            data_files = [file for file in data_files if not file.startswith(".")]

            # Create subject list from files
            subjects = []
            for file in data_files:
                # find the participant number in the filename
                subject_number = re.findall(r"\d+", file)[0]
                subjects.append(subject_number)
            # sort subject list
            subjects.sort()
            # process only two subjects if debug mode is on
            if debug:
                subjects = ["1", "2"]
                if (dataset == "AVR") & (
                    modality == "annotations_phase1"
                ):  # AVR dataset for phase 1 doesn't have annotations for subject 1 and 2
                    subjects = ["06", "08"]
            
            # Create color maps for plots
            number_colors = len(subjects)
            # Create a new colormap from the Wong color palette with one color for each subject
            subjects_colormap = LinearSegmentedColormap.from_list(
                'Wong_colormap_subjects', 
                [wong_colormap(i / (len(color_palette) - 1)) for i in range(len(color_palette))],
                N=number_colors  # Number of colors in the new colormap
                )

            # Create empty list to store data from all participants
            grouped_data_all = []
            
            # Loop through subjects
            for subject in subjects:
                # Load data
                # check if datafile contains subject number
                pattern = r"\D" + str(subject) + r"\D"
                right_datafile = next(data_file for data_file in data_files if re.search(pattern, data_file))
                data = pd.read_csv(datapath_dataset / right_datafile)

                # if dataset is AVR, use only "Flubber" as rating_method
                # drop the rows containing "Grid" or "Proprioceptive" values
                if modality == "annotations_phase1":
                    data = data[data["rating_method"] == "Flubber"]

                # drop rows containing NaN values
                data = data.dropna()

                # create a unique identifier for each row within each video for CASE and CEAP dataset
                if dataset in ("CASE", "CEAP"):
                    group_variable = "video_id"
                    data["row_id"] = data.groupby(group_variable).cumcount()
                    # drop columns with quadrant (as they're not numeric and cannot be averaged over)
                    data = data.drop(columns=["quadrant"])
                elif modality == "annotations_phase1":  # for quadrant for phase 1 of AVR dataset
                    group_variable = "quadrant"
                    data["row_id"] = data.groupby(group_variable).cumcount()
                    # drop columns with test site and rating_method (as they're not numeric and cannot be averaged over)
                    data = data.drop(columns=["test_site", "rating_method"])
                else:   # create a unique identifier for each row for phase 2 of AVR dataset
                    group_variable = "sj_id"
                    data["row_id"] = data.groupby(group_variable).cumcount()
                    # drop columns with test site (as they're not numeric and cannot be averaged over)
                    data = data.drop(columns=["test_site"])
                
                # add grouped data to list
                grouped_data_all.append(data)
            
            # concatenate all dataframes in the list
            concatenated_data = pd.concat(grouped_data_all)

            if dataset in ("CASE", "CEAP") or modality == "annotations_phase1":
                # group data by video and row_id for CASE and CEAP dataset, then calculate mean
                # group data by quadrant and row_id for phase 1 of AVR dataset, then calculate mean
                grouped_data = concatenated_data.groupby(group_variable)
                grouped_data_avg = concatenated_data.groupby([group_variable, "row_id"]).mean()

                # drop column with sj_id
                grouped_data_avg = grouped_data_avg.drop(columns=["sj_id"])

            else:
                # calculate mean for phase 2 of AVR dataset
                grouped_data = concatenated_data
                grouped_data_avg = concatenated_data.groupby("row_id").mean()
                # set sj_id to 1 as dummy (as there is only one video in phase 2 of AVR dataset)
                grouped_data_avg["sj_id"] = 1

            # Loop through analysis steps
            for step_number, step in enumerate(steps):
                print(f"Performing '{step}' (step {(step_number + 1)!s} of {len(steps)!s})...")
                
                # %% step 1: cpa >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
                if step == "cpa":

                    for averaging_mode in averaging_modes:
                        print(f"Averaging across '{averaging_mode}' (mode {(averaging_modes.index(averaging_mode) + 1)!s} of {len(averaging_modes)!s})...")

                        # Create result folder
                        resultpath_averaging_mode = (resultpath_dataset / averaging_mode)
                        if averaging_mode not in os.listdir(resultpath_dataset):
                            Path.mkdir(resultpath_averaging_mode)

                        if averaging_mode == "all": # average across both timeseries and changepoints
                            
                            # create empty list to store changepoints
                            changepoint_data = []

                            # Loop over videos
                            for video, group_data in grouped_data_avg.groupby(group_variable):
                                if "annotations" in modality:
                                    valence_data = group_data["cr_v"].to_numpy()
                                    arousal_data = group_data["cr_a"].to_numpy()

                                    # reshape data to fit the input format of the algorithm
                                    valence_data = np.array(valence_data).reshape(-1, 1)
                                    arousal_data = np.array(arousal_data).reshape(-1, 1)

                                    # perform changepoint analysis on valence
                                    valence_changepoints = get_changepoints(
                                        valence_data, model, pen, jump, min_size[dataset][modality_index]
                                    )
                                    # delete last element of the list (which is the number of samples)
                                    valence_changepoints.pop()

                                    # perform changepoint analysis on arousal
                                    # perform changepoint analysis on arousal
                                    arousal_changepoints = get_changepoints(
                                        arousal_data, model, pen, jump, min_size[dataset][modality_index]
                                    )
                                    # delete last element of the list (which is the number of samples)
                                    arousal_changepoints.pop()

                                    # determine size of figure depending on length of stimuli
                                    figsize = (
                                        (18, 5) if (dataset == "AVR") & (modality == "annotations_phase2") else (12, 6)
                                    )

                                    # set title and axes of plot
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        title_valence = (f"Changepoint Analysis of annotation data of Phase 2 of "
                                            f"{dataset} dataset for Valence averaged across participants")
                                        title_arousal = (f"Changepoint Analysis of annotation data of Phase 2 of "
                                            f"{dataset} dataset for Arousal averaged across participants)")
                                        x_axis_label = "Time (minutes)"
                                    elif (dataset == "AVR") & (modality == "annotations_phase1"):
                                        title_valence = (f"Changepoint Analysis of annotation data of Phase 1 of "
                                            f"{dataset} dataset for Valence averaged across participants (Video: {video})")
                                        title_arousal = (f"Changepoint Analysis of annotation data of Phase 1 of "
                                            f"{dataset} dataset for Arousal averaged across participants (Video: {video})")
                                        x_axis_label = "Time (seconds)"
                                    else:
                                        title_valence = (f"Changepoint Analysis of annotation data of "
                                            f"{dataset} dataset for Valence averaged across participants (Video: {video})")
                                        title_arousal = (f"Changepoint Analysis of annotation data of "
                                            f"{dataset} dataset for Arousal averaged across participants (Video: {video})")
                                        x_axis_label = "Time (seconds)"

                                    # visualize changepoints for valence
                                    plot_changepoints(
                                        valence_changepoints,
                                        valence_data,
                                        sampling_rates[dataset][modality_index],
                                        title_valence,
                                        x_axis_label,
                                        "Valence",
                                        figsize,
                                    )

                                    # change x-ticks to minutes if dataset is AVR phase 2
                                    # (because stimulus video is more than 20 minutes long)
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        x_ticks = plt.xticks()[0]
                                        plt.xticks(
                                            x_ticks,
                                            [
                                                int(xtick / sampling_rates[dataset][modality_index] / 60)
                                                for xtick in x_ticks
                                            ],
                                        )
                                        # set limits of x-axis
                                        plt.xlim(0, len(valence_data))

                                    # show plot
                                    # plt.show()  # noqa: ERA001

                                    # define name of file
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        name = "averaged_changepoints_time_valence.pdf"
                                    else:
                                        name = f"averaged_changepoints_time_V{video}_valence.pdf"

                                    # save plot to subject result folder
                                    plt.savefig(
                                            resultpath_averaging_mode /
                                            name
                                    )
                                    plt.close()

                                    # visualize changepoints for arousal
                                    plot_changepoints(
                                        arousal_changepoints,
                                        arousal_data,
                                        sampling_rates[dataset][modality_index],
                                        title_arousal,
                                        x_axis_label,
                                        "Arousal",
                                        figsize,
                                    )

                                    # change x-ticks to minutes if dataset is AVR phase 2
                                    # (because stimulus video is more than 20 minutes long)
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        x_ticks = plt.xticks()[0]
                                        plt.xticks(
                                            x_ticks,
                                            [
                                                int(xtick / sampling_rates[dataset][modality_index] / 60)
                                                for xtick in x_ticks
                                            ],
                                        )
                                        # set limits of x-axis
                                        plt.xlim(0, len(valence_data))

                                    # show plot
                                    # plt.show()  # noqa: ERA001

                                    # define name of file
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        name = "averaged_changepoints_time_arousal.pdf"
                                    else:
                                        name = f"averaged_changepoints_time_V{video}_arousal.pdf"

                                    # save plot to subject result folder
                                    plt.savefig(
                                            resultpath_averaging_mode /
                                            name
                                        )
                                    plt.close()

                                    # convert changepoints to seconds (rounded to two decimals)
                                    valence_changepoints_seconds = [
                                        round((changepoint / sampling_rates[dataset][modality_index]), 2)
                                        for changepoint in valence_changepoints
                                    ]
                                    arousal_changepoints_seconds = [
                                        round((changepoint / sampling_rates[dataset][modality_index]), 2)
                                        for changepoint in arousal_changepoints
                                    ]

                                    # add changepoints to changepoint_data
                                    changepoint_data.append(
                                        {
                                            "video": video,
                                            "valence_changepoints": valence_changepoints_seconds,
                                            "number_valence_changepoints": len(valence_changepoints_seconds),
                                            "arousal_changepoints": arousal_changepoints_seconds,
                                            "number_arousal_changepoints": len(arousal_changepoints_seconds),
                                            "model": model,
                                            "jump_value": jump,
                                            "penalty_value": pen,
                                            "min_size": min_size[dataset][modality_index],
                                        }
                                    )

                                else:  # if modality is physiological data (TODO)
                                    for physio_index, physiological_modality in enumerate(
                                        physiological_modalities[dataset]
                                    ):
                                        # get physiological data of that modality
                                        original_data = group_data[physiological_modality].to_numpy()
                                        ...
                            
                            # create dataframe from changepoint_data
                            changepoint_df = pd.DataFrame(changepoint_data)

                            # save changepoint dataframe to csv
                            # change name to include the two parameters model & jump (so that when we test different values, we save different files)
                            changepoint_df.to_csv(
                                Path(resultpath_dataset) / "all"/
                                f"annotations_changepoint_data_model={model}_jump={jump}_avg.csv",
                                index=False,
                            )

                        elif averaging_mode == "timeseries":    # average across timeseries only
                            
                            # get individual changepoints
                            changepoints_all = pd.read_csv(
                                Path(resultpath) / modality.split("_")[1] / "cpa" / modality.split("_")[0] /"all"/
                                    f"annotations_{modality.split('_')[1]}_changepoint_data_model={model}_jump={jump}.csv"
                                )
                            
                            # get averaged changepoints
                            changepoints_avg = pd.read_csv(
                                Path(resultpath_dataset) / "all"/
                                    f"annotations_{modality.split('_')[1]}_changepoint_data_model={model}_jump={jump}_avg.csv"
                                )

                            # Loop over videos
                            for video, group_data in grouped_data_avg.groupby(group_variable):
                                if "annotations" in modality:
                                    valence_data = group_data["cr_v"].to_numpy()
                                    arousal_data = group_data["cr_a"].to_numpy()

                                    # reshape data to fit the input format of the algorithm
                                    valence_data = np.array(valence_data).reshape(-1, 1)
                                    arousal_data = np.array(arousal_data).reshape(-1, 1)

                                    # get changepoints for all participants for that video
                                    changepoints_video = changepoints_all.loc[changepoints_all[group_variable] == video]
                                    valence_changepoints = changepoints_video["valence_changepoints"]
                                    arousal_changepoints = changepoints_video["arousal_changepoints"]


                                    # determine size of figure depending on length of stimuli
                                    figsize = (
                                        (18, 5) if (dataset == "AVR") & (modality == "annotations_phase2") else (12, 6)
                                    )

                                    # visualize changepoints for valence
                                    plt.figure(figsize=figsize)

                                    # plot averaged timeseries
                                    plt.plot(valence_data, label="mean", color=timeseries_color, linewidth=1)

                                    # create empty list for legend entries
                                    legend_subjects = []
                                    # plot changepoints for each subject in the same plot
                                    for index, subject in enumerate(subjects):
                                        # Check if the DataFrame is empty
                                        if not valence_changepoints.empty:
                                            # Reset the index of the DataFrame
                                            valence_changepoints = valence_changepoints.reset_index(drop=True)
                                            for changepoint in ast.literal_eval(valence_changepoints[index]):
                                                plt.axvline(changepoint*sampling_rates[dataset][modality_index], color=subjects_colormap[0](index/number_colors), alpha=0.3)
                                        subject_line = mlines.Line2D([], [], color=subjects_colormap[0](index/number_colors), alpha=0.3, label='cp ' + subject)
                                        legend_subjects.append(subject_line)
                                    
                                    # plot averaged changepoints
                                    changepoints_avg_video_valence = changepoints_avg.loc[changepoints_avg[group_variable] == video]["valence_changepoints"]
                                    # check if the series is empty
                                    if not changepoints_avg_video_valence.empty:
                                        # Reset the index of the Series
                                        changepoints_avg_video_valence = changepoints_avg_video_valence.reset_index(drop=True)
                                        for changepoint_avg in ast.literal_eval(changepoints_avg_video_valence[0]):
                                            plt.axvline(changepoint_avg*sampling_rates[dataset][modality_index], color=change_point_color, linestyle="--", linewidth=2, label="cp mean")

                                    # add legend
                                    # create Line2D instances for the legend
                                    cp_line = mlines.Line2D([], [], color=change_point_color, alpha=0.5, label='cp')
                                    mean_legend = mlines.Line2D([], [], color=timeseries_color, linewidth=1, label='mean')
                                    avg_legend = mlines.Line2D([], [], color=change_point_color, linestyle="--", linewidth=2, label='cp mean')
                                    legend_subjects.extend([avg_legend, mean_legend])
                                    plt.legend(handles=legend_subjects, fontsize='x-small')

                                    # set ticks to seconds
                                    x_ticks = plt.xticks()[0]
                                    plt.xticks(x_ticks, [int((xtick / sampling_rates[dataset][modality_index])) for xtick in x_ticks])

                                    # set limits of x-axis
                                    plt.xlim(0, len(valence_data))
                                    # set limits of y-axis
                                    plt.ylim(-1, 1)

                                    # change x-ticks to minutes if dataset is AVR phase 2
                                    # (because stimulus video is more than 20 minutes long)
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        x_ticks = plt.xticks()[0]
                                        plt.xticks(
                                            x_ticks,
                                            [
                                                int(xtick / sampling_rates[dataset][modality_index] / 60)
                                                for xtick in x_ticks
                                            ],
                                        )
                                        # set limits of x-axis
                                        plt.xlim(0, len(valence_data))

                                    # set title and axes of plot
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        title_valence = (f"Changepoint Analysis of annotation data of Phase 2 of "
                                            f"{dataset} dataset for Valence")
                                        title_arousal = (f"Changepoint Analysis of annotation data of Phase 2 of "
                                            f"{dataset} dataset for Arousal")
                                        x_axis_label = "Time (minutes)"
                                    elif (dataset == "AVR") & (modality == "annotations_phase1"):
                                        title_valence = (f"Changepoint Analysis of annotation data of Phase 1 of "
                                            f"{dataset} dataset for Valence (Video: {video})")
                                        title_arousal = (f"Changepoint Analysis of annotation data of Phase 1 of "
                                            f"{dataset} dataset for Arousal (Video: {video})")
                                        x_axis_label = "Time (seconds)"
                                    else:
                                        title_valence = (f"Changepoint Analysis of annotation data of "
                                            f"{dataset} dataset for Valence (Video: {video})")
                                        title_arousal = (f"Changepoint Analysis of annotation data of "
                                            f"{dataset} dataset for Arousal(Video: {video})")
                                        x_axis_label = "Time (seconds)"
                                    
                                    plt.title(title_valence)
                                    plt.xlabel(x_axis_label)
                                    plt.ylabel("Valence")

                                    # show plot
                                    # plt.show()  # noqa: ERA001

                                    # define name of file
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        name = "all_changepoints_time_valence.pdf"
                                    else:
                                        name = f"all_changepoints_time_V{video}_valence.pdf"

                                    # save plot to result folder
                                    plt.savefig(
                                            resultpath_averaging_mode /
                                            name
                                    )
                                    plt.close()

                                    # visualize changepoints for arousal
                                    plt.figure(figsize=figsize)

                                    # plot averaged timeseries
                                    plt.plot(arousal_data, label="mean", color=timeseries_color, linewidth=1)

                                    # create empty list for legend entries
                                    legend_subjects = []
                                    # plot changepoints for each subject in the same plot
                                    for index, subject in enumerate(subjects):
                                        # Check if the DataFrame is empty
                                        if not arousal_changepoints.empty:
                                            # Reset the index of the DataFrame
                                            arousal_changepoints = arousal_changepoints.reset_index(drop=True)
                                            for changepoint in ast.literal_eval(arousal_changepoints[index]):
                                                plt.axvline(changepoint*sampling_rates[dataset][modality_index], color=subjects_colormap[0](index/number_colors), alpha=0.3)
                                        subject_line = mlines.Line2D([], [], color=subjects_colormap[0](index/number_colors), alpha=0.3, label='cp ' + subject)
                                        legend_subjects.append(subject_line)
                                    
                                    # plot averaged changepoints
                                    changepoints_avg_video_arousal = changepoints_avg.loc[changepoints_avg[group_variable] == video]["arousal_changepoints"]
                                    # check if the series is empty
                                    if not changepoints_avg_video_arousal.empty:
                                        # Reset the index of the Series
                                        changepoints_avg_video_arousal = changepoints_avg_video_arousal.reset_index(drop=True)
                                        for changepoint_avg in ast.literal_eval(changepoints_avg_video_arousal[0]):
                                            plt.axvline(changepoint_avg*sampling_rates[dataset][modality_index], color=change_point_color, linestyle="--", linewidth=2, label="cp mean")
                                    
                                    # add legend
                                    # create Line2D instances for the legend
                                    cp_line = mlines.Line2D([], [], color=change_point_color, alpha=0.5, label='cp')
                                    mean_legend = mlines.Line2D([], [], color=timeseries_color, linewidth=1, label='mean')
                                    avg_legend = mlines.Line2D([], [], color=change_point_color, linestyle="--", linewidth=2, label='cp mean')
                                    legend_subjects.extend([avg_legend, mean_legend])
                                    plt.legend(handles=legend_subjects, fontsize='x-small')

                                    # set ticks to seconds
                                    x_ticks = plt.xticks()[0]
                                    plt.xticks(x_ticks, [int((xtick / sampling_rates[dataset][modality_index])) for xtick in x_ticks])

                                    # set limits of x-axis
                                    plt.xlim(0, len(arousal_data))
                                    # set limits of y-axis
                                    plt.ylim(-1, 1)

                                    # change x-ticks to minutes if dataset is AVR phase 2
                                    # (because stimulus video is more than 20 minutes long)
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        x_ticks = plt.xticks()[0]
                                        plt.xticks(
                                            x_ticks,
                                            [
                                                int(xtick / sampling_rates[dataset][modality_index] / 60)
                                                for xtick in x_ticks
                                            ],
                                        )
                                        # set limits of x-axis
                                        plt.xlim(0, len(arousal_data))
                                    
                                    plt.title(title_arousal)
                                    plt.xlabel(x_axis_label)
                                    plt.ylabel("Arousal")

                                    # show plot
                                    # plt.show()  # noqa: ERA001

                                    # define name of file
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        name = "all_changepoints_time_arousal.pdf"
                                    else:
                                        name = f"all_changepoints_time_V{video}_arousal.pdf"
                                    
                                    # save plot to result folder
                                    plt.savefig(
                                            resultpath_averaging_mode /
                                            name
                                        )
                                    plt.close()

                                    
                                else:  # if modality is physiological data (TODO)
                                    for physio_index, physiological_modality in enumerate(
                                        physiological_modalities[dataset]
                                    ):
                                        # get physiological data of that modality
                                        original_data = group_data[physiological_modality].to_numpy()
                                        ...

                        else:   # averaging_mode == "changepoints"  # average across changepoints only
                            ...


'''
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


                # %% step 3: summary statistics >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
                elif step in ("summary statistics", "test"):
                    ...

                else:
                    print("Error: Step not recognized. Please specify a valid step.")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

'''