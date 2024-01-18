"""
Script to perform a changepoint analysis (CPA) on a given time series.

Script includes functions to perform a CPA, to plot the results of a CPA, and to test the significance of the results
for participants from a given dataset individually.
If you want to perform a CPA for averaged data across participants, use the script cpa_averaged.py
(script needs to be in the same directory as this one as it imports functions from this script).

Inputs: Preprocessed time series data as .csv files for each participant of a given dataset

Outputs:

Functions:

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: January 18th, 2024
"""

# TODO: (Status 18.01.2024)  # noqa: FIX002
# - CPA for all three datasets for annotation data DONE
# - Elbow plots for all three datasets for annotation data DONE -> pen = 1 always optimal
# - make CPA plots for phase 1 of AVR longer / reduce number of change points (maybe with min_size parameter?) DONE
#   -> min_size = 20 or 30 so that there is max. one changepoint per second
# - Make plots prettier DONE

# - Check all elbow plots and see whether pen should always be 1
# - NO physiological data CPA working atm

# %% Import
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)   # ignore future warnings from seaborn

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# datasets to perform CPA on
datasets = ["AVR"]
# "CASE", "CEAP"    # noqa: ERA001

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
# "elbow", "summary statistics", "test"

# turn on debug mode (if True, only one subject is processed)
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


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_changepoints(signal: np.ndarray, model: str, pen: int, jump: int, min_size: int) -> list:
    """Return the changepoints for a given dataset."""
    # .fit() takes a signal as input and fits the algorithm to the data
    algorithm = rpt.Pelt(model=model, jump=jump, min_size=min_size).fit(signal)
    # other algorithms to use are: "Dynp", "Binseg", "Window"

    # .predict() returns a list of indexes corresponding to the end of each regime
    result = algorithm.predict(pen=pen)
    # by design, the last element of this list is the number of samples

    # store the changepoint timestamps
    return [int(change) for change in result]


def plot_elbow(signal: np.ndarray, model: str, list_penalties: list[int], jump: int, min_size: int):
    """Create an elbow plot to determine the optimal penalty."""
    # create an empty list to store the number of changepoints for each penalty value
    list_number_of_changepoints = []
    # perform a cpa for each possible penalty value specified in list_penalties
    for penalty_value in list_penalties:
        changepoints = get_changepoints(signal, model, penalty_value, jump, min_size)
        number_of_changepoints = len(changepoints)-1  # subtract 1 because last element of list is number of samples
        list_number_of_changepoints.append(number_of_changepoints)

    # plot elbow plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    # concatenate the two lists to create a dataframe
    data = pd.DataFrame({"Penalty": list_penalties, "Number of changepoints": list_number_of_changepoints})
    sns.lineplot(data=data, x="Penalty", y="Number of changepoints", color=elbow_color, linewidth=2)


def plot_changepoints(  # noqa: PLR0913
    changepoints: list[int],  # changepoints need to be in seconds
    signal: np.ndarray,  # signal needs to be in seconds
    sampling_rate: int,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple,
):
    """Plot results of changepoint analysis."""
    plt.figure(figsize=figsize)

    # plot annotation data
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=signal, palette=[timeseries_color], linewidth=1.5)  # black line

    # consecutively plot changepoints as vertical lines and shade area between changepoints
    counter = 0
    for index, changepoint in enumerate(changepoints):
        # plot vertical line at changepoint location
        plt.axvline(changepoint, color=change_point_color, linestyle="--", linewidth=2)  # vermillion dashed line

        # determine color and x-coordinate of shade area
        # reddish-purple shade if index is even, sky-blue shade if index is odd
        shade_color = shade_colors[0] if index % 2 == 0 else shade_colors[1]
        start_x = 0 if index == 0 else changepoints[index - 1]

        # shade area between changepoints
        plt.axvspan(start_x, changepoint, color=shade_color, alpha=0.5)

        counter += 1

    # shade area between last changepoint and end of signal
    if len(changepoints) != 0:
        shade_color_last = shade_colors[0] if counter % 2 == 0 else shade_colors[1]
        plt.axvspan(changepoints[-1], len(signal), color=shade_color_last, alpha=0.5)
    else:
        plt.axvspan(0, len(signal), color=shade_colors[0], alpha=0.5)

    # set ticks to seconds
    x_ticks = plt.xticks()[0]
    plt.xticks(x_ticks, [int(xtick / sampling_rate) for xtick in x_ticks])

    # set limits of x-axis
    plt.xlim(0, len(signal))

    # set limits of y-axis
    plt.ylim(-1.1, 1.1)

    # add title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # remove legend
    plt.legend().remove()

    # add number of changepoints
    plt.text(
        5 / 6 * len(signal),
        1,
        f"Number of changepoints: {len(changepoints)}",
        ha="center",
        va="center",
        size=12,
        color=change_point_color,
        bbox=dict(facecolor=(1, 1, 1, 0.8), edgecolor=change_point_color),
    )


# function that tests changepoints for significance
def test_changepoints():
    """Test changepoints for significance."""
    # TODO: write this function  # noqa: FIX002
    return ...


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
                resultpath_dataset = Path(resultpath) / modality.split("_")[1] / "cpa" / modality.split("_")[0]
            else:
                datapath_dataset = Path(datapath) / dataset / "preprocessed" / modality
                resultpath_dataset = Path(resultpath) / dataset / "cpa" / modality

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
            # process only one subject if debug mode is on
            if debug:
                subjects = ["1"]
                if (dataset == "AVR") & (
                    modality == "annotations_phase1"
                ):  # AVR dataset for phase 1 doesn't have annotations for subject 1
                    subjects = ["06"]

            # Loop through analysis steps
            for step_number, step in enumerate(steps):
                print(f"Performing '{step}' (step {(step_number + 1)!s} of {len(steps)!s})...")

                # %% step 1: elbow plots >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
                if step == "elbow":
                    # Loop through subjects
                    for subject in subjects:
                        print("Processing subject ", subject, " of ", str(len(subjects)), "...")

                        # Load data
                        pattern = r"\D" + str(subject) + r"\D"
                        right_datafile = next(data_file for data_file in data_files if re.search(pattern, data_file))
                        data = pd.read_csv(datapath_dataset / right_datafile)

                        # if dataset is AVR, use only "Flubber" as rating_method
                        # drop the rows containing "Grid" or "Proprioceptive" values
                        if modality == "annotations_phase1":
                            data = data[data["rating_method"] == "Flubber"]

                        # drop rows containing NaN values
                        data = data.dropna()

                        # Create subject result folder
                        resultpath_subject = resultpath_dataset / f"sub_{subject}" / "elbow_plots"
                        if f"sub_{subject}" not in os.listdir(resultpath_dataset):
                            Path.mkdir(resultpath_dataset / f"sub_{subject}")

                        if "elbow_plots" not in os.listdir(resultpath_dataset / f"sub_{subject}"):
                            Path.mkdir(resultpath_subject)

                        # Group data by video
                        if modality == "annotations_phase2":
                            grouped_data = data.groupby("sj_id")
                        else:
                            grouped_data = (
                                data.groupby("quadrant")
                                if modality == "annotations_phase1"
                                else data.groupby("video_id")
                            )

                        # Loop over videos
                        for video, group_data in grouped_data:
                            if "annotations" in modality:
                                valence_data = group_data["cr_v"].to_numpy()
                                arousal_data = group_data["cr_a"].to_numpy()

                                # reshape data to fit the input format of the algorithm
                                valence_data = np.array(valence_data).reshape(-1, 1)
                                arousal_data = np.array(arousal_data).reshape(-1, 1)

                                # plot elbow plot to determine the optimal penalty value for valence data
                                plot_elbow(
                                    valence_data, model, list_penalties, jump, min_size[dataset][modality_index]
                                )

                                # show elbow plot
                                # plt.show()  # noqa: ERA001

                                # define name of file
                                if (dataset == "AVR") & (modality == "annotations_phase2"):
                                    name = f"elbow_plot_valence_sub_{subject}.pdf"
                                else:
                                    name = f"elbow_plot_valence_sub_{subject}_video_{video}.pdf"

                                # save elbow plot
                                plt.savefig(
                                    resultpath_subject / name
                                )
                                plt.close()

                                # plot elbow plot to determine the optimal penalty value for arousal data
                                plot_elbow(
                                    arousal_data, model, list_penalties, jump, min_size[dataset][modality_index]
                                )

                                # show elbow plot
                                # plt.show()  # noqa: ERA001

                                # define name of file
                                if (dataset == "AVR") & (modality == "annotations_phase2"):
                                    name = f"elbow_plot_arousal_sub_{subject}.pdf"
                                else:
                                    name = f"elbow_plot_arousal_sub_{subject}_video_{video}.pdf"

                                # save elbow plot
                                plt.savefig(
                                    resultpath_subject / name
                                )
                                plt.close()

                            else:  # if modality is physiological data
                                for physiological_modality in physiological_modalities[dataset]:
                                    # get physiological data of that modality
                                    data = group_data[physiological_modality].to_numpy()

                                    # reshape data to fit the input format of the algorithm
                                    data = np.array(data).reshape(-1, 1)

                                    # drop NaN values
                                    data = data[~np.isnan(data)]

                                    # plot elbow plot to determine the optimal penalty value for physiological data
                                    plot_elbow(data, model, list_penalties, jump, min_size[dataset][modality_index])

                                    # show elbow plot
                                    # plt.show()  # noqa: ERA001

                                    # define name of file
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        name = f"elbow_plot_{physiological_modality}_sub_{subject}.pdf"
                                    else:
                                        name = (
                                            f"elbow_plot_{physiological_modality}_sub_{subject}_video_{video}.pdf"
                                        )

                                    # save elbow plot
                                    plt.savefig(
                                        resultpath_subject /
                                        name
                                    )
                                    plt.close()

                # %% step 2: cpa >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
                elif step == "cpa":
                    # Create empty dataframe to store changepoints
                    changepoints_all = []

                    # Loop through subjects
                    for subject in subjects:
                        print("Processing subject ", subject, " of ", str(len(subjects)), "...")

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

                        # Create subject result folder
                        resultpath_subject = (resultpath_dataset / f"sub_{subject}")
                        if f"sub_{subject}" not in os.listdir(resultpath_dataset):
                            Path.mkdir(resultpath_subject)

                        # Group data by video
                        if modality == "annotations_phase2":
                            grouped_data = data.groupby("sj_id")
                        else:
                            grouped_data = (
                                data.groupby("quadrant")
                                if modality == "annotations_phase1"
                                else data.groupby("video_id")
                            )

                        # Create empty list to store changepoints
                        changepoint_data = []

                        # Loop over videos
                        for video, group_data in grouped_data:
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
                                        f"{dataset} dataset for Valence (Subject: {subject})")
                                    title_arousal = (f"Changepoint Analysis of annotation data of Phase 2 of "
                                        f"{dataset} dataset for Arousal (Subject: {subject})")
                                    x_axis_label = "Time (minutes)"
                                elif (dataset == "AVR") & (modality == "annotations_phase1"):
                                    title_valence = (f"Changepoint Analysis of annotation data of Phase 1 of "
                                        f"{dataset} dataset for Valence (Subject: {subject}, Video: {video})")
                                    title_arousal = (f"Changepoint Analysis of annotation data of Phase 1 of "
                                        f"{dataset} dataset for Arousal (Subject: {subject}, Video: {video})")
                                    x_axis_label = "Time (seconds)"
                                else:
                                    title_valence = (f"Changepoint Analysis of annotation data of "
                                        f"{dataset} dataset for Valence (Subject: {subject}, Video: {video})")
                                    title_arousal = (f"Changepoint Analysis of annotation data of "
                                        f"{dataset} dataset for Arousal (Subject: {subject}, Video: {video})")
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
                                    name = f"sub_{subject}_changepoints_valence.pdf"
                                else:
                                    name = f"sub_{subject}_changepoints_V{video}_valence.pdf"

                                # save plot to subject result folder
                                plt.savefig(
                                        resultpath_subject /
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
                                    name = f"sub_{subject}_changepoints_arousal.pdf"
                                else:
                                    name = f"sub_{subject}_changepoints_V{video}_arousal.pdf"

                                # save plot to subject result folder
                                plt.savefig(
                                        resultpath_subject /
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
                                        "subject": subject,
                                        "video": "1" if (dataset == "AVR") & (modality == "annotations_phase2") else video,
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

                            else:  # if modality is physiological data
                                for physio_index, physiological_modality in enumerate(
                                    physiological_modalities[dataset]
                                ):
                                    # get physiological data of that modality
                                    original_data = group_data[physiological_modality].to_numpy()

                                    # drop NaN values
                                    data = original_data[~np.isnan(original_data)]

                                    # reshape data to fit the input format of the algorithm
                                    data = np.array(data).reshape(-1, 1)

                                    # perform changepoint analysis on physiological data
                                    physiological_changepoints = get_changepoints(
                                        data, model, pen, jump, min_size[dataset][modality_index]
                                    )
                                    # delete last element of the list (which is the number of samples)
                                    physiological_changepoints.pop()

                                    # sanity check: plot physiological timeseries
                                    plt.figure(figsize=(12, 6))
                                    physiological_timepoints = group_data["time"][~np.isnan(original_data)]
                                    # reshape x data
                                    physiological_timepoints = np.array(physiological_timepoints).reshape(-1, 1)
                                    plt.plot(physiological_timepoints, data)

                                    # set ticks to seconds
                                    # x_ticks = plt.xticks()[0]  # noqa: ERA001
                                    # plt.xticks(x_ticks,
                                    # [int((xtick / sampling_rates[dataset][modality_index])) for xtick in x_ticks])

                                    # set limits of x-axis
                                    # plt.xlim(0, len(data) + 100)  # noqa: ERA001

                                    plt.show()
                                    plt.close()

                                    # visualize changepoints for physiological data
                                    plot_changepoints(
                                        physiological_changepoints,
                                        data,
                                        sampling_rates[dataset][modality_index],
                                        (f"Changepoint Analysis of {physiological_modality} data of "
                                            f"{dataset} dataset (Subject: {subject}, Video: {video})"),
                                        "Time (seconds)",
                                        physiological_modality,
                                        figsize,
                                    )

                                    # show plot
                                    # plt.show()  # noqa: ERA001

                                    # define name of file
                                    if (dataset == "AVR") & (modality == "annotations_phase2"):
                                        name = f"sub_{subject}_changepoints_{physiological_modality}.pdf"
                                    else:
                                        name = (
                                            f"sub_{subject}_changepoints_V{video}_{physiological_modality}.pdf"
                                        )

                                    # save plot to subject result folder
                                    plt.savefig(
                                            resultpath_subject /
                                            name
                                        )

                                    # convert changepoints to seconds (rounded to two decimals)
                                    physiological_changepoints_seconds = [
                                        round((changepoint / sampling_rates[dataset][modality_index]), 2)
                                        for changepoint in physiological_changepoints
                                    ]

                                    # add changepoints to changepoint_data
                                    if physio_index == 0:
                                        changepoint_data.append(
                                            {
                                                "subject": subject,
                                                "video": "1" if (dataset == "AVR") & (modality == "annotations_phase2") else video,
                                                physiological_modality: physiological_changepoints_seconds,
                                                f"number_{physiological_modality}_changepoints": len(
                                                    physiological_changepoints_seconds
                                                ),
                                                "model": model,
                                                "jump_value": jump,
                                                "penalty_value": pen,
                                            }
                                        )
                                    else:
                                        changepoint_data.append(
                                            {
                                                "video": "1" if (dataset == "AVR") & (modality == "annotations_phase2") else video,
                                                physiological_modality: physiological_changepoints_seconds,
                                                f"number_{physiological_modality}_changepoints": len(
                                                    physiological_changepoints_seconds
                                                ),
                                            }
                                        )

                        # Create dataframe from changepoint_data
                        changepoint_df = pd.DataFrame(changepoint_data)

                        # Add changepoint_df to changepoints_all
                        changepoints_all.append(changepoint_df)

                        # Display the changepoint dataframe
                        # print(changepoint_df)  # noqa: ERA001

                        # Save changepoint dataframe to csv
                        # Change name to include the two parameters model & jump
                        # (so that when we test different values, we save different files)
                        changepoint_df.to_csv(
                                resultpath_subject /
                                f"sub_{subject}_{modality}_changepoint_data_model={model}_jump={jump}.csv",
                            )

                    # Create dataframe from changepoints_all
                    changepoints_all_df = pd.concat(changepoints_all)

                    # Create new folder for all changepoints
                    resultpath_set_all = (resultpath_dataset / "all")
                    if "all" not in os.listdir(resultpath_dataset):
                        Path.mkdir(resultpath_set_all)

                    # save changepoints_all_df to csv
                    changepoints_all_df.to_csv(
                            resultpath_set_all /
                            f"{modality}_changepoint_data_model={model}_jump={jump}.csv",
                        )

                # %% step 3: summary statistics >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
                elif step in ("summary statistics", "test"):
                    ...

                else:
                    print("Error: Step not recognized. Please specify a valid step.")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
