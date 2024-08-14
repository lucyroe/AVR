"""
Script to perform a Hidden Markov Model (HMM) analysis on Affective VR cardiac and neural data.

Required packages: hmmlearn

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 22 May 2024
Last update: 14 August 2024
"""


def hmm(  # noqa: C901, PLR0912, PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003"],  # noqa: B006
    debug=False,
    show_plots=False,
):
    """
    Perform a Hidden Markov Model (HMM) analysis on Affective VR cardiac and neural data.

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
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from hmmlearn.hmm import GMMHMM

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    datapath = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "features"
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to create
    models = ["cardiac", "neural", "integrated"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
    }

    # Define whether features should be z-scored to have mean 0 and standard deviation 1
    z_score = True

    # Number of states (four quadrants of the Affect Grid) = hyperparameter that needs to be chosen
    number_of_states = 4

    # Number of iterations for training the HMM
    iterations = 1000

    # Number of Gaussian mixtures in the Gaussian Mixture Model (GMM) for each hidden state
    number_of_mixtures = 1

    # Covariance type for the GMMHMM
    covariance_type = "diag"

    color_features = "black"
    # Colors of the hidden states
    colors_states = ["#009E73", "#CC79A7", "#0072B2", "#D55E00"]  # green, pink, dark blue, dark orange

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    def create_model(data, number_of_states, iterations=1000):
        """
        Create and train a Hidden Markov Model (HMM) on the given data. Then predict the hidden states.

        Args:
        ----
            data (np.array): The data to train the HMM on, can contain multiple features.
            number_of_states (int): The number of hidden states of the HMM.
            iterations (int): The number of iterations to train the HMM (defaults to 1000).

        Returns:
        -------
            model (GMMHMM): The trained HMM model.
            hidden_states (np.array): The predicted hidden states.
        """
        # Set the random seed for reproducibility
        seed = 42
        model = GMMHMM(
            n_components=number_of_states,
            n_mix=number_of_mixtures,
            n_iter=iterations,
            random_state=seed,
            covariance_type=covariance_type,
        )
        model.fit(data)
        hidden_states = model.predict(data)

        return model, hidden_states

    def plot_hidden_states(  # noqa: PLR0913
        data, hidden_states, axis, num_states, title, ylabel, legend
    ):
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
            legend (list): The legend of the plot.
        """
        axis.plot(data, color=color_features, linewidth=1)
        axis.set_title(title)

        # Calculate min and max for proper scaling
        data_min = min(data)
        data_max = max(data)

        # Loop over all states
        for i in range(num_states):
            # Fill the area between the min and max of the data where the hidden state is i
            axis.fill_between(
                np.arange(len(data)),
                data_min,
                data_max,
                where=hidden_states == i,
                color=colors_states[i],
                alpha=0.5,
                label=legend[i],
            )
        # TODO: problem of 1s windows between changes -> fill_between does not work properly

        # Transform the x-axis to min
        xticks = axis.get_xticks()
        axis.set_xticks(xticks)
        axis.set_xticklabels([f"{int(x/60)}" for x in xticks])
        axis.set_xlabel("Time (min)")
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. GET DATA
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Initiating {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        # Loop over all subjects
        for subject_index, subject in enumerate(subjects):
            print("---------------------------------")
            print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject)) + "...")
            print("---------------------------------\n")
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
                        feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[
                            feature.split("_")[1]
                        ]
                    else:
                        feature_datafile = f"sub-{subject}_task-AVR_eeg_features_whole-brain_power.tsv"
                        feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]
                elif model == "integrated":
                    if feature in ["ibi", "hf-hrv"]:
                        feature_datafile = f"sub-{subject}_task-AVR_ecg_features.tsv"
                        feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]
                    elif feature in ["posterior_alpha", "frontal_alpha", "frontal_theta"]:
                        feature_datafile = f"sub-{subject}_task-AVR_eeg_features_{feature.split('_')[0]}_power.tsv"
                        feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[
                            feature.split("_")[1]
                        ]
                    else:
                        feature_datafile = f"sub-{subject}_task-AVR_eeg_features_whole-brain_power.tsv"
                        feature_data = pd.read_csv(subject_datapath / feature_datafile, sep="\t")[feature]

                # Z-score the data if necessary
                if z_score:
                    feature_data = (feature_data - feature_data.mean()) / feature_data.std()

                # Append the feature data to the list
                all_features.append(feature_data.values.reshape(-1, 1))  # noqa: PD011

            # Check if the lengths of the features are the same
            if len({len(feature) for feature in all_features}) != 1:
                # Cut the features to the same length
                min_length = min([len(feature) for feature in all_features])
                all_features = [feature[:min_length] for feature in all_features]

            # Put all features in one array that has the shape (n_samples, n_features)
            print("Combining all features into one array...\n")
            data = np.column_stack(tuple(all_features))

            # %% STEP 2. HMMs

            # Create and train the Hidden Markov Model
            print("Creating and training the Hidden Markov Model...")

            hmm, hidden_states = create_model(data, number_of_states, iterations)

            print("Saving results...\n")

            # Define the file path to save the model
            hmm_path = resultpath / f"sub-{subject}" / "hmm" / model

            # Create the model directory if it does not exist yet
            hmm_path.mkdir(parents=True, exist_ok=True)

            features_string = "_".join(models_features[model])

            # Create a dataframe with the state sequence corresponding to each timepoint
            hidden_states_df = pd.DataFrame({"state": hidden_states})

            # Add a column with the time as first column
            hidden_states_df.insert(0, "timestamp", np.arange(len(hidden_states_df)))

            # Save the hidden states to a tsv file
            hidden_states_file = f"sub-{subject}_task-AVR_{model}_model_hidden_states_{features_string}.tsv"
            hidden_states_df.to_csv(hmm_path / hidden_states_file, sep="\t", index=False)

            # Create a dataframe with the data and the hidden states
            data_df = pd.DataFrame(data, columns=models_features[model])
            data_df["state"] = hidden_states
            # Add the time as first column
            data_df.insert(0, "timestamp", np.arange(len(data_df)))

            # Save the data with the hidden states to a tsv file
            data_file = f"sub-{subject}_task-AVR_{model}_model_data_{features_string}.tsv"
            data_df.to_csv(hmm_path / data_file, sep="\t", index=False)

            # Create a metadata json file with the model parameters
            hmm_metadata = {
                "subject": subject,
                "model": model,
                "number_of_states": number_of_states,
                "iterations": iterations,
                "number_of_mixtures": number_of_mixtures,
                "covariance_type": covariance_type,
                "features": models_features[model],
                "number_of_features": len(models_features[model]),
                "z_score": z_score,
            }

            # Save the metadata to a json file
            metadata_file = f"sub-{subject}_task-AVR_{model}_model_metadata_{features_string}.json"
            with Path(hmm_path / metadata_file).open("w") as f:
                json.dump(hmm_metadata, f)

            # Create a dictionary with the state parameters
            hmm_parameters = {}
            for state in range(number_of_states):
                # Get the percentage of time spent in the state
                percentage = len(hidden_states[hidden_states == state]) / len(hidden_states)
                hmm_state_parameters = {
                    "state": state,
                    "percentage": percentage,
                    "means": hmm.means_[state].tolist(),
                    "covars": hmm.covars_[state].tolist(),
                    "startprob": hmm.startprob_[state].tolist(),
                    "transmat": hmm.transmat_[state].tolist(),
                }
                # Add the state parameters to the dictionary
                hmm_parameters[f"state_{state}"] = hmm_state_parameters

            # Save the state parameters to a json file
            parameters_file = f"sub-{subject}_task-AVR_{model}_model_parameters_{features_string}.json"
            with Path(hmm_path / parameters_file).open("w") as f:
                json.dump(hmm_parameters, f)

            # Create a plot for the model with a subplot for each feature
            fig, axs = plt.subplots(len(all_features), 1, figsize=(10, 5 * len(all_features)))
            for feature_index, feature in enumerate(models_features[model]):
                plot_hidden_states(
                    all_features[feature_index],
                    hidden_states,
                    axs[feature_index] if len(all_features) > 1 else axs,
                    number_of_states,
                    f"{feature}",
                    feature,
                    legend=[f"State {i}" for i in range(number_of_states)],
                )

            # Set the title of the plot
            fig.suptitle(f"{model.capitalize()} Model for subject {subject}", fontsize=16)

            # Save the plot
            plot_file = f"sub_{subject}_{model}_hmm_{features_string}.png"
            plt.savefig(hmm_path / plot_file)

            # Show the plot
            if show_plots:
                plt.show()

            plt.close()

        print(f"Finished {model} model.\n")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    hmm()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
