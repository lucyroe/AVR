"""
Script to perform a Hidden Markov Model (HMM) analysis on Annotation, ECG and EEG Data.

Inputs: Preprocessed Annotation, ECG and EEG data.

Outputs:
- Hidden Markov Model (HMM) for Annotation, ECG and EEG data.
- Plot of the data with the hidden states marked in color vertically.

Functions:
- create_model: Create and train a Hidden Markov Model (HMM) on the given data.
- plot_hidden_states: Plot the data with the hidden states marked in color vertically.

Steps:
1. GET DATA
2. HIDDEN MARKOV MODEL (HMM)
    2a. Create and train the Hidden Markov Model for the physiological data.
    2b. Create and train the Hidden Markov Model for the annotations.
    2c. Save the Hidden Markov Model.
    2d. Plot the data with the hidden states marked in color vertically.

Required packages: hmmlearn

Author: Lucy Roellecke
Contact: lucy.roellecke@fu-berlin.de
Last update: May 28th, 2024
"""

# %% Import
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/NeVRo/preprocessed/"
resultpath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/NeVRo/hmm/"

# create the result folder if it does not exist
Path(resultpath).mkdir(parents=True, exist_ok=True)

modalities = ["physiological", "annotations"]
# which preprocessing mode to use for physiological data
preprocessing_mode = "Antonin"
# "Lucy",
# which physiological measures to analyze
physiological_measures = ["IBI", "HF_HRV", "posterior_alpha_power"]

modalities_scales = {
    "physiological": {"IBI": [0.4, 1.8], "HF_HRV": [0, 0.12], "posterior_alpha_power": [1e-11, 1e-8]},
    "annotations": [-1, 1],
}

# if debug is True, only one subject will be analyzed
debug = False

# Number of states (low and high HR) = hyperparameter that needs to be chosen
number_of_states = 2

# Number of iterations for training the HMM
iterations = 1000


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def create_model(data, num_states, iterations=1000):
    """
    Create and train a Hidden Markov Model (HMM) on the given data. Then predicts the hidden states.

    Args:
    ----
        data (np.array): The data to train the HMM on.
        num_states (int): The number of states of the HMM.
        iterations (int): The number of iterations to train the HMM (defaults to 1000).

    Returns:
    -------
        model (GaussianHMM): The trained HMM model.
        hidden_states (np.array): The predicted hidden states.
    """
    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=iterations)
    model.fit(data)
    hidden_states = model.predict(data)

    return model, hidden_states


def plot_hidden_states(data, hidden_states, num_states, title, ylabel, scale):
    """
    Plot the data with the hidden states marked in color vertically.

    Args:
    ----
        data (np.array): The data to plot.
        hidden_states (np.array): The hidden states of the data.
        num_states (int): The number of states.
        title (str): The title of the plot.
        ylabel (str): The label of the y-axis.
        scale (list): The scale of the data.
    """
    plt.plot(data)
    plt.title(title)
    for i in range(num_states):
        plt.fill_between(
            np.arange(len(data)), scale[0], scale[1], where=(hidden_states == i), color="C" + str(i), alpha=0.3
        )

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. GET DATA
if __name__ == "__main__":
    for modality in modalities:
        print(f"\nProcessing {modality} data...")

        # get right file path for the dataset
        filepath = Path(datapath) / modality
        if modality == "physiological":
            filepath = filepath / preprocessing_mode
            for physiological_measure in physiological_measures:
                print(f"\nProcessing {physiological_measure} data...")

                # list all files in the physiological folder corresponding to the physiological measure
                pattern_measure = f"*_{physiological_measure}_preprocessed.tsv"
                file_list = glob.fnmatch.filter(os.listdir(filepath), pattern_measure)

                # delete all files that are not subject files
                file_list = [file for file in file_list if file.startswith("sub")]

                # sort the list in ascending order
                file_list = sorted(file_list)

                # get participant list from the file list
                subject_list = [file.split("_")[1][0:3] for file in file_list]

                # only analyze one subject if debug is True
                if debug:
                    subject_list = [subject_list[0]]

                # Loop over all subjects
                for subject_index, subject in enumerate(subject_list):
                    print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

                    # Get the right file for the subject
                    pattern_subject = f"sub_{subject}_mov_Space_{physiological_measure}_preprocessed.tsv"
                    subject_file = glob.fnmatch.filter(file_list, pattern_subject)[0]

                    # Read in data
                    subject_data = pd.read_csv(filepath / subject_file, sep="\t")

                    # Get measure
                    data = subject_data[physiological_measure].values.reshape(-1, 1)

                    # STEP 2. HMM

                    # Create and train the Hidden Markov Model
                    print("\nCreating and training the Hidden Markov Model...")

                    model_physiological, hidden_states_physiological = create_model(data, number_of_states, iterations)

                    # Define the file path to save the model
                    model_path = Path(resultpath) / modality / subject

                    # Create a dataframe with the means and variances of the hidden states
                    means = model_physiological.means_
                    variances = model_physiological.covars_
                    hidden_states_df = pd.DataFrame(
                        {
                            "Hidden State": np.arange(1, number_of_states + 1),
                            "Mean": means.flatten(),
                            "Variance": variances.flatten(),
                        }
                    )

                    # Save the dataframe as tsv file
                    hidden_states_file = f"sub_{subject}_{physiological_measure}_hmm.tsv"
                    hidden_states_physiological_file = model_path / hidden_states_file
                    hidden_states_physiological_file.parent.mkdir(parents=True, exist_ok=True)
                    hidden_states_df.to_csv(hidden_states_physiological_file, sep="\t", index=False)

                    # Plot the physiological measure with the hidden states marked in color vertically
                    plot_hidden_states(
                        data,
                        hidden_states_physiological,
                        number_of_states,
                        f"{physiological_measure} with Hidden States",
                        physiological_measure,
                        modalities_scales[modality][physiological_measure],
                    )

                    # Save the plot
                    plot_file = f"sub_{subject}_{physiological_measure}_hmm.png"
                    plot_physiological_file = model_path / plot_file
                    plt.savefig(plot_physiological_file)

                    # Show the plot
                    # plt.show()  # noqa: ERA001

                    plt.close()

        else:
            # list all files in the annotations folder
            file_list = os.listdir(filepath)

            # deletes all hidden files from the list
            file_list = [file for file in file_list if not file.startswith(".")]

            # delete all files that are not subject files
            file_list = [file for file in file_list if file.startswith("sub")]

            # sort the list in ascending order
            file_list = sorted(file_list)

            # get participant list from the file list
            subject_list = [file.split("_")[1][0:3] for file in file_list]

            # only analyze one subject if debug is True
            if debug:
                subject_list = [subject_list[0]]

            # Loop over all subjects
            for subject_index, subject in enumerate(subject_list):
                print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject_list)) + "...")

                # Get the right file for the subject
                pattern_subject = f"sub_{subject}_mov_space_arousal_preprocessed.tsv"
                subject_file = glob.fnmatch.filter(file_list, pattern_subject)[0]

                # Read in data
                subject_data = pd.read_csv(filepath / subject_file, sep="\t")

                # Get the annotations
                arousal = subject_data["cr_a"].dropna().values.reshape(-1, 1)

                # STEP 2. HMM

                # Create and train the Hidden Markov Model
                print("\nCreating and training the Hidden Markov Model...")

                model_arousal, hidden_states_arousal = create_model(arousal, number_of_states, iterations)

                # Define the file path to save the model
                model_path = Path(resultpath) / modality / subject

                # Create a dataframe with the means and variances of the hidden states
                means = model_arousal.means_
                variances = model_arousal.covars_
                hidden_states_df = pd.DataFrame(
                    {
                        "Hidden State": np.arange(1, number_of_states + 1),
                        "Mean": means.flatten(),
                        "Variance": variances.flatten(),
                    }
                )

                # Save the dataframe as tsv file
                hidden_states_file = f"sub_{subject}_{modality}_hmm.tsv"
                hidden_states_annotation_file = model_path / hidden_states_file
                hidden_states_annotation_file.parent.mkdir(parents=True, exist_ok=True)
                hidden_states_df.to_csv(hidden_states_annotation_file, sep="\t", index=False)

                # Plot the annotations with the hidden states marked in color vertically
                plot_hidden_states(
                    arousal,
                    hidden_states_arousal,
                    number_of_states,
                    "Arousal with Hidden States",
                    "Arousal",
                    modalities_scales[modality],
                )

                # Save the plot
                plot_file = f"sub_{subject}_{modality}_hmm.png"
                plot_annotation_file = model_path / plot_file
                plt.savefig(plot_annotation_file)

                # Show the plot
                # plt.show()  # noqa: ERA001

                plt.close()


# %%
