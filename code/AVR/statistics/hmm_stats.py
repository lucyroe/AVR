"""
Script to calculate summary stats and test for differences between the hidden states of the different HMMs.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 14 August 2024
Last update: 16 August 2024
"""

def hmm_stats(  # noqa: C901, PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003"],  # noqa: B006
    debug=False):
    """
    Calculate summary statistics for the HMMs.

    Inputs: Features with hidden states from the HMMs

    Outputs: Summary statistics for the HMMs

    Steps:
    1. Get data
    2. Calculate stats
    3. Calculate averaged stats
    """
    # %% Import
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    annotation_path = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "features" / "avg" / "beh"
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to analyze
    models = ["cardiac", "neural", "integrated"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
    }

    # Number of states (four quadrants of the Affect Grid)
    number_of_states = 4

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. GET DATA
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Calculating stats for {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        # Load the features with hidden states
        hmm_path = resultpath / "avg" / "hmm" / model
        features_string = "_".join(models_features[model])
        hidden_states_file = f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv"
        hidden_states_data = pd.read_csv(hmm_path / hidden_states_file, sep="\t")

        # Load the annotations
        annotation_file = f"all_subjects_task-AVR_beh_features.tsv"
        annotations = pd.read_csv(annotation_path / annotation_file, sep="\t")

        # Initialize the summary stats for all subjects
        summary_stats_all_subjects = pd.DataFrame()
        global_stats_all_subjects = pd.DataFrame()
        # Loop over all subjects
        for subject_index, subject in enumerate(subjects):
            print("---------------------------------")
            print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subject)) + "...")
            print("---------------------------------\n")

            # Get the data for the subject
            hidden_states_subject = hidden_states_data[hidden_states_data["subject"] == int(subject)].reset_index(drop=True)
            annotations_subject = annotations[annotations["subject"] == int(subject)].reset_index(drop=True)

            # Delete subject and video column
            annotations_subject = annotations_subject.drop(columns=["subject", "video"])
            hidden_states_subject = hidden_states_subject.drop(columns=["subject"])

            # Check if the timestamps have the same length
            if len(hidden_states_subject) != len(annotations_subject):
                # Cut the longer one to the length of the shorter one
                min_length = min(len(hidden_states_subject), len(annotations_subject))
                hidden_states_subject = hidden_states_subject[:min_length]
                annotations_subject = annotations_subject[:min_length]

            # Merge the hidden states with the annotations
            data = pd.merge(hidden_states_subject, annotations_subject, on="timestamp")

            # Get a list of all features
            features = [col for col in data.columns if col not in ["timestamp", "state"]]

            # STEP 2. CALCULATE STATS
            # Loop over all features
            summary_stats_all = pd.DataFrame()
            for feature in features:
                print(f"Calculating stats for feature {feature}...")

                summary_stats = pd.DataFrame()
                # Calculate the mean and standard deviation for each hidden state
                summary_stats["mean"] = data.groupby("state")[feature].mean()
                summary_stats["std"] = data.groupby("state")[feature].std()
                # Calculate the minimum and maximum for each hidden state
                summary_stats["min"] = data.groupby("state")[feature].min()
                summary_stats["max"] = data.groupby("state")[feature].max()

                summary_stats.insert(0, "feature", feature)

                # Append the summary stats for the feature to the summary stats for all features
                summary_stats_all = pd.concat([summary_stats_all, summary_stats], axis=0)

            # Add the subject ID to the summary stats as first column
            summary_stats_all.insert(0, "subject", subject)

            # Append the summary stats for the subject to the summary stats for all subjects
            summary_stats_all_subjects = pd.concat([summary_stats_all_subjects, summary_stats_all], axis=0)

            # Calculate the global statistics for each state
            print("Calculating global stats...")
            global_stats = pd.DataFrame()
            # Loop over all hidden states
            for state in range(number_of_states):
                # Get the data for the state
                data_state = hidden_states_subject[hidden_states_subject["state"] == state]
                data_state.reset_index(drop=True, inplace=True)

                # Fractional Occupancy
                fractional_occupancy = len(data_state) / len(hidden_states_subject)
                global_stats.loc[state, "fractional_occupancy"] = fractional_occupancy

                # Lifetime & Intervaltime
                lifetime = 0
                lifetimes = []
                intervaltimes = []
                # Loop over all timepoints
                for index, timepoint in enumerate(data_state["timestamp"]):
                    # Check if the timepoint is the first one
                    if index == 0:
                        continue
                    elif timepoint == data_state["timestamp"][index-1] + 1:
                        lifetime += 1
                    else:
                        lifetimes.append(lifetime)
                        lifetime = 0
                        intervaltime = timepoint - data_state["timestamp"][index-1]
                        intervaltimes.append(intervaltime)

                # Calculate the mean lifetime
                mean_lifetime = np.mean(lifetimes)
                global_stats.loc[state, "mean_lifetime"] = mean_lifetime
                # Calculate the mean intervaltime
                mean_intervaltime = np.mean(intervaltimes)
                global_stats.loc[state, "mean_intervaltime"] = mean_intervaltime

            # Add the subject ID to the global stats as first column
            global_stats.insert(0, "subject", subject)

            # Append the global stats for the subject to the global stats for all subjects
            global_stats_all_subjects = pd.concat([global_stats_all_subjects, global_stats], axis=0)

        # Add the state to the summary stats as second column
        summary_stats_all_subjects.insert(1, "state", np.tile(np.arange(number_of_states), len(subjects)*len(features)))

        # Reset the index of the summary stats for all subjects
        summary_stats_all_subjects = summary_stats_all_subjects.reset_index(drop=True)

        # Save the summary stats for all subjects to a file
        resultpath_model = resultpath / "avg" / "hmm" / model
        # Create the directory if it does not exist
        resultpath_model.mkdir(parents=True, exist_ok=True)

        summary_stats_all_subjects.to_csv(resultpath_model / f"all_subjects_task-AVR_{model}_model_states_stats.tsv", sep="\t", index=False)

        # Add the state to the global stats as second column
        global_stats_all_subjects.insert(1, "state", np.tile(np.arange(number_of_states), len(subjects)))

        # Reset the index of the global stats for all subjects
        global_stats_all_subjects = global_stats_all_subjects.reset_index(drop=True)

        # Save the global stats for all subjects to a file
        global_stats_all_subjects.to_csv(resultpath_model / f"all_subjects_task-AVR_{model}_model_states_global_stats.tsv", sep="\t", index=False)

        # %% STEP 3. CALCULATE AVERAGED STATS
        # Calculate the statistics for each hidden state averaged across all subjects
        summary_stats_averaged = pd.DataFrame()
        global_stats_averaged = pd.DataFrame()
        for state in range(number_of_states):
            print(f"Calculating averaged stats for state {state}...")

            summary_stats = pd.DataFrame()
            # Calculate the mean and standard deviation for each feature
            summary_stats["mean"] = summary_stats_all_subjects[summary_stats_all_subjects["state"]
                    == state].groupby("feature")["mean"].mean()
            summary_stats["std"] = summary_stats_all_subjects[summary_stats_all_subjects["state"]
                    == state].groupby("feature")["std"].mean()
            # Calculate the minimum and maximum for each feature
            summary_stats["min"] = summary_stats_all_subjects[summary_stats_all_subjects["state"]
                    == state].groupby("feature")["min"].mean()
            summary_stats["max"] = summary_stats_all_subjects[summary_stats_all_subjects["state"]
                    == state].groupby("feature")["max"].mean()
            summary_stats.insert(0, "state", state)

            # Append the summary stats for the state to the summary stats for all states
            summary_stats_averaged = pd.concat([summary_stats_averaged, summary_stats], axis=0)

            # Calculate the global statistics for each state
            global_stats = pd.DataFrame()
            # Calculate the mean fractional occupancy
            global_stats.loc[state, "fractional_occupancy"] = global_stats_all_subjects[global_stats_all_subjects["state"] == state]["fractional_occupancy"].mean()
            # Calculate the mean lifetime
            global_stats.loc[state, "mean_lifetime"] = global_stats_all_subjects[global_stats_all_subjects["state"] == state]["mean_lifetime"].mean()
            # Calculate the mean intervaltime
            global_stats.loc[state, "mean_intervaltime"] = global_stats_all_subjects[global_stats_all_subjects["state"] == state]["mean_intervaltime"].mean()
            global_stats.insert(0, "state", state)

            # Append the global stats for the state to the global stats for all states
            global_stats_averaged = pd.concat([global_stats_averaged, global_stats], axis=0)

        # Add a column with the feature names
        summary_stats_averaged.insert(1, "feature", np.tile(features, number_of_states))

        # Reset the index of the summary stats for all states
        summary_stats_averaged = summary_stats_averaged.reset_index(drop=True)

        # Save the summary stats for all states to a file
        summary_stats_averaged.to_csv(resultpath_model / f"avg_task-AVR_{model}_model_states_stats.tsv", sep="\t", index=False)

        # Reset the index of the global stats for all states
        global_stats_averaged = global_stats_averaged.reset_index(drop=True)

        # Save the global stats for all states to a file
        global_stats_averaged.to_csv(resultpath_model / f"avg_task-AVR_{model}_model_states_global_stats.tsv", sep="\t", index=False)

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    hmm_stats()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END