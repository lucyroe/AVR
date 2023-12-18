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
Last update: December 13, 2023
"""

# %% Import
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns


# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# datasets to perform CPA on
datasets = ["CASE"]
# "CEAP", "AVR"]

# dataset modalities
modalities = {"CASE": ["annotations", "physiological"],
                "CEAP": ["annotations", "physiological"],
                "AVR": ["annotations_phase1", "annotations_phase2"]}

# modalities sampling frequencies
sampling_rates = {"CASE": [20, 1000],
                    "CEAP": [30, 30],
                    "AVR": [20, 20]}

# number of videos per dataset
number_of_videos = {"CASE": 8,
                    "CEAP": 8,
                    "AVR": [4, 1]}

# video to quadrant mapping for each dataset
# defines which video corresponds to which quadrant of the circumplex model of emotions
video_quadrant_mapping = {"CASE": {"1": "HP", "2": "HP", "3": "LN", "4": "LN", "5": "LP", "6": "LP", "7": "HN", "8": "HN"},
                            "CEAP": {"1": "HP", "2": "LP", "3": "HN", "4": "LN", "5": "HP", "6": "LP", "7": "HN", "8": "LN"},
                            "AVR": {"1": "HP", "2": "LP", "3": "LN", "4": "HN"}}

# change to where you saved the preprocessed data
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"

# change to where you want to save the results
resultpath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/"

# analysis steps to perform
steps = ["elbow", "cpa", "summary statistics", "test"]

# %% Set CPA parameters >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

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


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def get_changepoints(signal: np.ndarray, model: str, pen: int, jump: int) -> list:
    """Function that returns the changepoints for a given dataset."""
    # .fit() takes a signal as input and fits the algorithm to the data
    algorithm = rpt.Pelt(model=model, jump=jump).fit(signal)
    # other algorithms to use are: "Dynp", "Binseg", "Window"

    # .predict() returns a list of indexes corresponding to the end of each regime
    result = algorithm.predict(pen=pen)
    # by design, the last element of this list is the number of samples

    # store the changepoint timestamps
    changepoints = [int(change) for change in result]

    return changepoints


def plot_elbow(signal: np.ndarray, model: str, list_penalties: list[int], jump: int) -> plt.figure:
    """Function that creates an elbow plot to determine the optimal penalty"""
    # create an empty list to store the number of changepoints for each penalty value
    list_number_of_changepoints = []
    # perform a cpa for each possible penalty value specified in list_penalties
    for penalty_value in list_penalties:
        changepoints = get_changepoints(signal, model, penalty_value, jump)
        number_of_changepoints = len(changepoints)
        list_number_of_changepoints.append(number_of_changepoints)

    # plot elbow plot
    figure, axes = plt.subplot(figsize=(12, 6))
    plt.plot(list_penalties, list_number_of_changepoints)
    plt.xlabel("Penalty")
    plt.ylabel("Number of changepoints")
    plt.axvline(pen, color="r", linestyle="--")
    # show plot
    plt.show()

    return figure


def plot_changepoints(
    changepoints: list[int],  # changepoints need to be in seconds
    signal: np.ndarray,  # signal needs to be in seconds
    sampling_rate: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.figure:
    """Function that plots results of changepoint analysis"""
    figure, axes = plt.subplot(figsize=(12, 6))
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

    return figure


# function that tests changepoints for significance
def test_changepoints():
    # TODO: write this function
    return ...


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":

    # Loop through datasets
    for dataset in datasets:

        # Loop through modalities
        for modality_index, modality in enumerate(modalities[dataset]):
            print(f"Performing changepoint analysis for {modality} data of {dataset} dataset...")

            # Set datapath for dataset
            if dataset == "AVR":
                datapath_dataset = os.path.join(datapath, modality.split("_")[1:], 'preprocessed', modality.split("_")[0])
            else:
                datapath_dataset = os.path.join(datapath, dataset, 'preprocessed', modality)

            # Set resultpath for dataset
            resultpath_dataset = os.path.join(resultpath, dataset, 'cpa', modality)
            # Create resultpath if it doesn't exist yet
            if not os.path.exists(resultpath_dataset):
                os.makedirs(resultpath_dataset)

            # List data files
            data_files = os.listdir(datapath_dataset)
            # Remove hidden files
            data_files = [file for file in data_files if not file.startswith(".")]

            # Create subject list from files
            subjects = [str(subject_number) for subject_number in range(1, len(data_files)+1)]

            # Loop through analysis steps
            for step_number, step in enumerate(steps):
                print(f"Performing '{step}' (step {(step_number + 1)!s} of {len(steps)!s})...")

                if step == "elbow":
                    # Loop through subjects
                    for subject_index, subject in enumerate(subjects):
                        print("Processing subject ", subject, " of ", str(len(subjects)), "...")

                        # Load data
                        data = pd.read_csv(os.path.join(datapath_dataset, data_files[subject_index]))

                        # Create subject result folder
                        resultpath_subject = os.path.join(resultpath_dataset, f"sub_{subject}", "elbow_plots")
                        if f"sub_{subject}" not in os.listdir(resultpath_dataset):
                            os.mkdir(os.path.join(resultpath_dataset, f"sub_{subject}"))
                        
                        if "elbow_plots" not in os.listdir(os.path.join(resultpath_dataset, f"sub_{subject}")):
                            os.mkdir(resultpath_subject)
                        else:
                            print("Caution: Result folder already exists. Files may be overwritten.")

                        # Group data by video
                        grouped_data = data.groupby("quadrant") if dataset == "AVR" else data.groupby("video_id")

                        # Loop over videos
                        for video, group_data in grouped_data:
                            if "annotations" in modality:
                                valence_data = group_data["cr_v"].values
                                arousal_data = group_data["cr_a"].values

                                # reshape data to fit the input format of the algorithm
                                valence_data = np.array(valence_data).reshape(-1, 1)
                                arousal_data = np.array(arousal_data).reshape(-1, 1)

                                # plot elbow plot to determine the optimal penalty value for valence data
                                elbow_valence = plot_elbow(valence_data, model, list_penalties, jump)
                                # save elbow plot
                                elbow_valence.savefig(os.path.join(resultpath_subject, f"elbow_plot_valence_sub_{subject}_video_{video}.png"))
                                plt.close()

                                # plot elbow plot to determine the optimal penalty value for arousal data
                                elbow_arousal = plot_elbow(arousal_data, model, list_penalties, jump)
                                # save elbow plot
                                elbow_arousal.savefig(os.path.join(resultpath_subject, f"elbow_plot_arousal_sub_{subject}_video_{video}.png"))
                                plt.close()
                            else:   # if modality is physiological data
                                ...
                                # TODO: add elbow plots for physiological data

                elif step == "cpa":
                    # Create empty dataframe to store changepoints
                    changepoints_all = []

                    # Loop through subjects
                    for subject_index, subject in enumerate(subjects):
                        print("Processing subject ", subject, " of ", str(len(subjects)), "...")

                        # Load data
                        data = pd.read_csv(os.path.join(datapath_dataset, data_files[subject_index]))

                        # Create subject result folder
                        resultpath_subject = os.path.join(resultpath_dataset, f"sub_{subject}")
                        if f"sub_{subject}" not in os.listdir(resultpath_dataset):
                            os.mkdir(resultpath_subject)
                        else:
                            print("Caution: Result folder already exists. Files may be overwritten.")

                        # Group data by video
                        grouped_data = data.groupby("quadrant") if dataset == "AVR" else data.groupby("video_id")

                        # Create empty list to store changepoints
                        changepoint_data = []

                        # Loop over videos
                        for video, group_data in grouped_data:

                            if "annotations" in modality:
                                valence_data = group_data["cr_v"].values
                                arousal_data = group_data["cr_a"].values

                                # reshape data to fit the input format of the algorithm
                                valence_data = np.array(valence_data).reshape(-1, 1)
                                arousal_data = np.array(arousal_data).reshape(-1, 1)

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
                                cp_plot_valence = plot_changepoints(
                                    valence_changepoints,
                                    valence_data,
                                    sampling_rates[dataset[modality_index]],
                                    f"Changepoint Analysis of annotation data of {dataset} dataset for Valence (Subject: {subject}, Video: {video})",
                                    "Time (seconds)",
                                    "Valence",
                                )
                                # show plot
                                # plt.show()

                                # save plot to subject result folder
                                cp_plot_valence.savefig(
                                    os.path.join(
                                        resultpath_subject,
                                        f"sub_{subject}_changepoints_V{video}_valence.pdf",
                                    )
                                )
                                plt.close()

                                # visualize changepoints for arousal
                                cp_plot_arousal = plot_changepoints(
                                    arousal_changepoints,
                                    arousal_data,
                                    sampling_rates[dataset[modality_index]],
                                    f"Changepoint Analysis of annotation data of {dataset} dataset for Arousal (Subject: {subject}, Video: {video})",
                                    "Time (seconds)",
                                    "Arousal",
                                )
                                # show plot
                                # plt.show()

                                # save plot to subject result folder
                                cp_plot_arousal.savefig(
                                    os.path.join(
                                        resultpath_subject,
                                        f"sub_{subject}_changepoints_V{video}_arousal.pdf",
                                    )
                                )
                                plt.close()

                                # convert changepoints to seconds (rounded to two decimals)
                                valence_changepoints_seconds = [
                                    round((changepoint / sampling_rates[dataset]), 2) for changepoint in valence_changepoints
                                ]
                                arousal_changepoints_seconds = [
                                    round((changepoint / sampling_rates[dataset]), 2) for changepoint in arousal_changepoints
                                ]

                                # add changepoints to changepoint_data
                                changepoint_data.append(
                                    {
                                        "subject": subject,
                                        "video": video,
                                        "valence_changepoints": valence_changepoints_seconds,
                                        "number_valence_changepoints": len(valence_changepoints_seconds),
                                        "arousal_changepoints": arousal_changepoints_seconds,
                                        "number_arousal_changepoints": len(arousal_changepoints_seconds),
                                        "model": model,
                                        "jump_value": jump,
                                        "penalty_value": pen,
                                    }
                                )
                            
                            else:   # if modality is physiological data
                                ...
                                # TODO: add changepoint plots for physiological data

                        
                        # Create dataframe from changepoint_data
                        changepoint_df = pd.DataFrame(changepoint_data)

                        # Add changepoint_df to changepoints_all
                        changepoints_all.append(changepoint_df)

                        # Display the changepoint dataframe
                        # print(changepoint_df)

                        # Save changepoint dataframe to csv
                        # Change name to include the two parameters model & jump (so that when we test different values, we save different files)
                        changepoint_df.to_csv(
                            os.path.join(
                                resultpath_subject,
                                f"sub_{subject}_{modality}_changepoint_data_model={model}_jump={jump}.csv",
                            ),
                            index=False,
                        )
                    
                    # Create dataframe from changepoints_all
                    changepoints_all_df = pd.concat(changepoints_all)

                    # Create new folder for all changepoints
                    resultpath_set_all = os.path.join(resultpath_dataset, "all")
                    if "all" not in os.listdir(resultpath_dataset):
                        os.mkdir(resultpath_set_all)

                    # save changepoints_all_df to csv
                    changepoints_all_df.to_csv(
                        os.path.join(
                            resultpath_set_all,
                            f"{modality}_changepoint_data_model={model}_jump={jump}.csv",
                        ),
                        index=False,
                    )


                elif step == "summary statistics": 
                    ...

                elif step == "test":
                    ...

                else:
                    print("Error: Step not recognized. Please specify a valid step.")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
