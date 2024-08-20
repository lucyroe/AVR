"""
Script to perform a Hidden Markov Model (HMM) analysis on Affective VR cardiac and neural data.

Required packages: hmmlearn, pickle, seaborn

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 22 May 2024
Last update: 20 August 2024
"""

def hmm(  # noqa: C901, PLR0912, PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003","004", "005", "006", "007", "009",  # noqa: B006
                "012", "014", "015", "016", "018", "019",
                "020", "021", "022", "024", "025", "026", "027", "028",
                "030", "031", "032", "034", "037", "038",
                "040", "041", "042", "043", "045", "046"],
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
    1. Get data: Load the extracted ECG and EEG features.
    2. HMMs: Create and train the Hidden Markov Models.
    3. Save results: Save the models, hidden states, data, metadata, and parameters.
    """
    # %% Import
    import json
    import pickle
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from hmmlearn.hmm import GMMHMM
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    datapath = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "features"
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to create
    models = ["cardiac", "neural", "integrated"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
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

    # Colors of the hidden states
    colors_states = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]  #  dark orange, dark blue, green, pink

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    def create_model(data, number_of_states, lengths, iterations=1000):
        """
        Create and train a Hidden Markov Model (HMM) on the given data. Then predict the hidden states.

        Args:
        ----
            data (np.array): The data to train the HMM on, can contain multiple features.
            number_of_states (int): The number of hidden states of the HMM.
            lengths (np.array): The lengths of the sequences in the data.
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
        model.fit(data, lengths)
        hidden_states = model.predict(data, lengths)

        return model, hidden_states

    def plot_hidden_states(  # noqa: PLR0913
        data, hidden_states, axis, num_states, title, legend
    ):
        """
        Plot the data with the hidden states.

        Args:
        ----
            data (np.array): The data to plot.
            hidden_states (np.array): The hidden states of the data.
            axis (matplotlib.axis): The axis to plot on.
            num_states (int): The number of states.
            title (str): The title of the plot.
            legend (list): The legend of the plot.
        """
        # Define the intervals with the hidden states to be of different colors
        segments = []
        colors = []
        color_labels = []

        all_states = np.unique(hidden_states)
        all_states.sort()

        times = np.arange(len(data))

        for i in range(len(data) - 1):
            xi = times[i:i+2]
            yi = data[i:i+2]

            if hidden_states[xi[1]] == all_states[0]:
                colors.append(colors_states[0])
                color_labels.append(legend[0])
            elif hidden_states[xi[1]] == all_states[1]:
                colors.append(colors_states[1])
                color_labels.append(legend[1])
            elif hidden_states[xi[1]] == all_states[2]:
                colors.append(colors_states[2])
                color_labels.append(legend[2])
            elif hidden_states[xi[1]] == all_states[3]:
                colors.append(colors_states[3])
                color_labels.append(legend[3])

            segments.append(np.column_stack([xi, yi]))

        # Create a LineCollection from the segments and colors
        lc = LineCollection(segments, colors=colors, linewidth=2)

        # Add the LineCollection to the axis
        axis.add_collection(lc)

        # Set the title of the plot
        if title in ("ibi", "hf-hrv"):
            title = title.upper()
            ylabel = "Value (z-scored)"
        else:
            title = title.replace("_", " ").title()
            ylabel = "Power (z-scored)"

        axis.set_title(title, fontsize=14, fontweight="bold")

        # Adjust the limits of the plot
        axis.set_xlim(-1, len(data) + 100)
        axis.set_ylim(data.min()-1, data.max()+1)

        # Create custom legend handles
        legend_handles = [
            Line2D([0], [0], color=colors_states[i], lw=2, label=legend[i]) for i in range(num_states)
                    ]

        # Add the legend to the plot
        axis.legend(handles=legend_handles, loc="upper right")

        # Transform the x-axis to min
        xticks = axis.get_xticks()
        axis.set_xticks(xticks)
        axis.set_xticklabels([f"{int(x/60)}" for x in xticks])
        axis.set_xlabel("Time (min)")
        axis.set_ylabel(ylabel)

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. GET DATA
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Initiating {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        # Create empty dataframe to store the data of all subjects
        data_all_subjects = pd.DataFrame()

        # Loop over all subjects
        for subject_index, subject in enumerate(subjects):
            print("---------------------------------")
            print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
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

            # Add the data to the dataframe
            data_all_subjects = pd.concat(
                [data_all_subjects, pd.DataFrame(data, columns=models_features[model])], axis=0
            )

        # %% STEP 2. HMMs
        # Create and train the Hidden Markov Model
        print("Creating and training the Hidden Markov Model...")
        hmm_all_subjects, hidden_states_all_subjects = create_model(
            data_all_subjects, number_of_states, len(data_all_subjects), iterations
        )

        # %% STEP 3. SAVE RESULTS
        print("Saving results...\n")

        # Define the file path to save the model
        hmm_path = resultpath / "avg" / "hmm" / model

        # Create the model directory if it does not exist yet
        hmm_path.mkdir(parents=True, exist_ok=True)

        features_string = "_".join(models_features[model])

        # Create a dataframe with the state sequence corresponding to each timepoint
        hidden_states_all_subjects_df = pd.DataFrame({"state": hidden_states_all_subjects})

        # Add a column with the time as first column
        hidden_states_all_subjects_df.insert(
            0, "timestamp", np.tile(np.arange(len(hidden_states_all_subjects_df) / len(subjects)), len(subjects))
        )

        # Add the subject ID to the hidden states
        hidden_states_all_subjects_df.insert(
            0, "subject", np.repeat(subjects, len(hidden_states_all_subjects_df) // len(subjects))
        )

        # Save the hidden states to a tsv file
        hidden_states_all_subjects_file = f"all_subjects_task-AVR_{model}_model_hidden_states_{features_string}.tsv"
        hidden_states_all_subjects_df.to_csv(hmm_path / hidden_states_all_subjects_file, sep="\t", index=False)

        # Create a dataframe with the data and the hidden states
        data_all_subjects_df = pd.DataFrame(data_all_subjects, columns=models_features[model])
        data_all_subjects_df["state"] = hidden_states_all_subjects
        # Add the time as first column
        data_all_subjects_df.insert(
            0, "timestamp", np.tile(np.arange(len(hidden_states_all_subjects_df) / len(subjects)), len(subjects))
        )
        # Add the subject ID to the data
        data_all_subjects_df.insert(
            0, "subject", np.repeat(subjects, len(hidden_states_all_subjects_df) // len(subjects))
        )

        # Save the data with the hidden states to a tsv file
        data_file_all_subjects = f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv"
        data_all_subjects_df.to_csv(hmm_path / data_file_all_subjects, sep="\t", index=False)

        # Create a metadata json file with the model parameters
        hmm_metadata_all_subjects = {
            "subjects": subjects,
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
        metadata_file_all_subjects = f"all_subjects_task-AVR_{model}_model_metadata_{features_string}.json"
        with Path(hmm_path / metadata_file_all_subjects).open("w") as f:
            json.dump(hmm_metadata_all_subjects, f)

        # Create a dictionary with the state parameters
        hmm_parameters_all_subjects = {}
        for state in range(number_of_states):
            # Get the percentage of time spent in the state
            percentage = len(hidden_states_all_subjects[hidden_states_all_subjects == state]) / len(
                hidden_states_all_subjects
            )
            hmm_state_parameters_all_subjects = {
                "state": state,
                "percentage": percentage,
                "means": hmm_all_subjects.means_[state].tolist(),
                "covars": hmm_all_subjects.covars_[state].tolist(),
                "startprob": hmm_all_subjects.startprob_[state].tolist(),
                "transmat": hmm_all_subjects.transmat_[state].tolist(),
            }
            # Add the state parameters to the dictionary
            hmm_parameters_all_subjects[f"state_{state}"] = hmm_state_parameters_all_subjects

        # Save the state parameters to a json file
        parameters_file_all_subjects = f"all_subjects_task-AVR_{model}_model_parameters_{features_string}.json"
        with Path(hmm_path / parameters_file_all_subjects).open("w") as f:
            json.dump(hmm_parameters_all_subjects, f)

        # Save the model for future use
        hmm_model_file = f"all_subjects_task-AVR_{model}_model_{features_string}.pkl"
        with Path(hmm_path / hmm_model_file).open("wb") as f:
            pickle.dump(hmm_all_subjects, f)

        # Create a plot for the model for each participant with a subplot for each feature
        for subject in subjects:
            hmm_path_subject = resultpath / f"sub-{subject}" / "hmm" / model
            # Create the subject directory if it does not exist yet
            hmm_path_subject.mkdir(parents=True, exist_ok=True)

            # Plot the hidden states for each feature
            sns.set(style="ticks")
            fig, axs = plt.subplots(len(all_features), 1, figsize=(10, 5 * len(all_features)))
            for feature_index, feature in enumerate(models_features[model]):
                plot_hidden_states(
                    data_all_subjects_df[data_all_subjects_df["subject"] == subject][feature],
                    (hidden_states_all_subjects_df[hidden_states_all_subjects_df["subject"] == subject]
                        ["state"]).reset_index(drop=True),
                    axs[feature_index] if len(all_features) > 1 else axs,
                    number_of_states,
                    f"{feature}",
                    legend=[f"State {i}" for i in range(number_of_states)],
                )

            # Set the title of the plot
            fig.suptitle(f"{model.capitalize()} Model for Subject {subject}", fontsize=16)

            # Remove the x-labels from all but the last subplot
            for ax in axs[:-1]:
                ax.set_xlabel("")

            # Remove the space between the subplots and the title
            plt.subplots_adjust(top=0.9)

            # Save the plot
            plot_file = f"sub-{subject}_{model}_hmm_{features_string}.png"
            plt.savefig(hmm_path_subject / plot_file)

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
