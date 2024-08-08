"""
Script to perform a Hidden Markov Model (HMM) analysis on Affective VR cardiac and neural data.

Required packages: hmmlearn

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 22 May 2024
Last update: 8 August 2024
"""

# %%
data_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"
results_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/"
subjects = ["001", "002", "003"]
debug = True
show_plots = True
# TODO: wrap everything in a function
"""
Function to perform a Hidden Markov Model (HMM) analysis on Affective VR cardiac and neural data.

Inputs: Extracted ECG and EEG features.

Outputs:
- Three Hidden Markov Models (HMMs): Cardiac Model, Neural Model, Integrated Model.
- Plot of the data with the hidden states marked in color vertically.

Functions:
- create_model: Create and train a Hidden Markov Model (HMM) on the given data.
- plot_hidden_states: Plot the data with the hidden states marked in color vertically.

Steps:
1. GET DATA
2. HIDDEN MARKOV MODELs (HMMs) TODO: adapt if necessary
    2a. Create and train the Cardiac Hidden Markov Model.
    2b. Create and train the Neural Hidden Markov Model.
    2c. Create and train the Integrated Hidden Markov Model.
    2d. Save the HMMs.
    2e. Plot the data with the hidden states marked in color vertically.
"""
# %% Import
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "features"
resultpath = Path(results_dir) / "phase3" / "AVR"

# Which HMMs to create
models = ["cardiac","neural", "integrated"]
# Which features are used for which HMM
models_features = {
    "cardiac": ["ibi", "hf-hrv"],
    "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"],
    "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"],
}

# Define whether features should be z-scored to have mean 0 and standard deviation 1
z_score = True

# Number of states (four quadrants of the Affect Grid) = hyperparameter that needs to be chosen
number_of_states = 4

# Number of iterations for training the HMM
iterations = 1000

# Colors TODO

# Only analyze one subject if debug is True
if debug:
    subjects = [subjects[0]]

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def create_model(data, lengths, num_states, iterations=1000):
    """
    Create and train a Hidden Markov Model (HMM) on the given data. Then predicts the hidden states.

    Args:
    ----
        data (np.array): The data to train the HMM on, can contain multiple features.
        lengths (list): The lengths of the sequences in the data.
        num_states (int): The number of states of the HMM.
        iterations (int): The number of iterations to train the HMM (defaults to 1000).

    Returns:
    -------
        model (GMMHMM): The trained HMM model.
        hidden_states (np.array): The predicted hidden states.
    """
    # Set the random seed for reproducibility
    seed = 42
    model = GMMHMM(n_components=num_states, n_iter=iterations, random_state=seed, covariance_type="tied")
    model.fit(data, lengths)
    hidden_states = model.predict(data, lengths)

    return model, hidden_states


def plot_hidden_states(data, hidden_states, axis, num_states, title, ylabel):
    """
    Plot the data with the hidden states marked in color vertically.

    Args:
    ----
        data (np.array): The data to plot.
        hidden_states (np.array): The hidden states of the data.
        axis (matplotlib.axis): The axis to plot on.
        num_states (int): The number of states.
        title (str): The title of the plot.
        ylabel (str): The label of the y-axis.
    """
    axis.plot(data)
    axis.set_title(title)
    for i in range(num_states):
        axis.fill_between(
            np.arange(len(data)), min(data), max(data), where=(hidden_states == i), color="C" + str(i), alpha=0.3
        )

    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# %% STEP 1. GET DATA
# Loop over all models
for model in models:
    print("+++++++++++++++++++++++++++++++++")
    print(f"Initiating {model} model...")
    print("+++++++++++++++++++++++++++++++++\n")

    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject)) + "...")
        # Get the right datapath
        subject_datapath = datapath / f"sub-{subject}" / "eeg"

        # Create empty list to store the data of all features
        all_features = []
        # Loop over all features
        for feature in models_features[model]:
            print(f"Loading {feature} data...")

            # Load the data for the feature
            if model == "cardiac":
                feature_datafile = f"sub-{subject}_task-AVR_ecg_features.tsv"
                feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]
            elif model == "neural":
                if feature in ["posterior_alpha", "frontal_alpha", "frontal_theta"]:
                    feature_datafile = f"sub-{subject}_task-AVR_eeg_features_{feature.split('_')[0]}_power.tsv"
                    feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature.split("_")[1]]
                else:
                    feature_datafile = f"sub-{subject}_task-AVR_eeg_features_whole-brain_power.tsv"
                    feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]
            else:   # integrated model
                if feature in ["ibi", "hf-hrv"]:
                    feature_datafile = f"sub-{subject}_task-AVR_ecg_features.tsv"
                    feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]
                elif feature in ["posterior_alpha", "frontal_alpha", "frontal_theta"]:
                    feature_datafile = f"sub-{subject}_task-AVR_eeg_features_{feature.split('_')[0]}_power.tsv"
                    feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature.split("_")[1]]
                else:
                    feature_datafile = f"sub-{subject}_task-AVR_eeg_features_whole-brain_power.tsv"
                    feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]
            
            # Z-score the data if necessary
            if z_score:
                feature_data = (feature_data - feature_data.mean()) / feature_data.std()
            
            # Append the feature data to the list
            all_features.append(feature_data.values.reshape(-1, 1))

        # Check if the lengths of the features are the same
        if len(set([len(feature) for feature in all_features])) != 1:
            # Cut the features to the same length
            min_length = min([len(feature) for feature in all_features])
            all_features = [feature[:min_length] for feature in all_features]

        # Concatenate all features
        print("Concatenating all features...\n")
        data = np.concatenate(all_features)
        lengths = [len(feature) for feature in all_features]

        # %% STEP 2. HMMs

        # Create and train the Hidden Markov Model
        print("Creating and training the Hidden Markov Model...")

        hmm, hidden_states = create_model(data, lengths, number_of_states, iterations)

        # Define the file path to save the model
        model_path = resultpath / f"sub-{subject}" / "hmm"

        # Create the model directory if it does not exist yet
        model_path.mkdir(parents=True, exist_ok=True)

        # Create a dataframe with the state sequence corresponding to each timepoint of each feature
        hidden_states_df = pd.DataFrame()
        for feature_index, feature in enumerate(models_features[model]):
            hidden_states_sequence = hidden_states[feature_index*len(all_features[feature_index]): (feature_index+1)*len(all_features[feature_index])]
            hidden_states_df[feature] = hidden_states_sequence

        # Add a column with the time as first column
        hidden_states_df.insert(0, "timepoint", np.arange(len(hidden_states_df)))

        # Create a dataframe with the summary statistics of the hidden states
        means = hmm.means_
        variances = hmm.covars_
        hidden_states_df_stats = pd.DataFrame(
            {
                "subject": subject,
                "model": model,
                "hidden_state": np.arange(1, number_of_states + 1),
                "mean": means.flatten(),
                "variance": variances.flatten(),
            }
        )

        print("Saving results...")

        # Save both dataframes to a tsv file
        # TODO

        # Create a plot for the model with a subplot for each feature
        fig, axs = plt.subplots(len(all_features), 1, figsize=(10, 5 * len(all_features)))
        for feature_index, feature in enumerate(models_features[model]):
            plot_hidden_states(
                all_features[feature_index],
                hidden_states[feature_index*len(all_features[feature_index]): (feature_index+1)*len(all_features[feature_index])],
                axs[feature_index],
                number_of_states,
                f"{feature}",
                feature,
            )

        # Set the title of the plot
        fig.suptitle(f"{model.capitalize()} Hidden Markov Model for subject {subject}", fontsize=16)

        # Save the plot
        plot_file = f"sub_{subject}_{model}_hmm.png"
        plt.savefig(model_path / plot_file)

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()

        print(f"Finished {model} model.\n")
# %%