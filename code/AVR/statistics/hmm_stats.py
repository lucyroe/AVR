"""
Script to calculate summary stats of the hidden states of the different HMMs.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 14 August 2024
Last update: 22 August 2024
"""


def hmm_stats(  # noqa: C901, PLR0915, PLR0912
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003","004", "005", "006", "007", "009",  # noqa: B006
        "012", "014", "015", "016", "018", "019",
        "020", "021", "022", "024", "025", "026", "027", "028", "029",
        "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
        "040", "041", "042", "043", "045", "046"],
    debug=False,
):
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
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to analyze
    models = ["cardiac", "neural", "integrated", "subjective"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "subjective": ["valence", "arousal"],
    }

    # Number of states (four quadrants of the Affect Grid)
    number_of_states = 4

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. GET DATA
    # First load all the data from all participants
    data_path = resultpath / "avg" / "hmm" / "integrated"
    annotation_path = resultpath / "avg" / "hmm" / "subjective"
    features_string_integrated = "_".join(models_features["integrated"])
    data_all_subjects = pd.read_csv(data_path /
            f"all_subjects_task-AVR_integrated_model_data_{features_string_integrated}.tsv",
        sep="\t")
    data_all_subjects = data_all_subjects.drop(columns=["state"])

    # Load the annotations
    annotation_features_string = "_".join(models_features["subjective"])
    annotation_file = f"all_subjects_task-AVR_subjective_model_data_{annotation_features_string}.tsv"
    annotations = pd.read_csv(annotation_path / annotation_file, sep="\t")
    annotations = annotations.drop(columns=["state"])

    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Calculating stats for {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        # Load the features with hidden states
        hmm_path = resultpath / "avg" / "hmm" / model
        features_string = "_".join(models_features[model])
        hidden_states_filename = f"all_subjects_task-AVR_{model}_model_hidden_states_{features_string}.tsv"
        hidden_states_file = pd.read_csv(hmm_path / hidden_states_filename, sep="\t")

        # Initialize the summary stats for all subjects
        summary_stats_all_subjects = pd.DataFrame()
        global_stats_all_subjects = pd.DataFrame()
        # Loop over all subjects
        for subject_index, subject in enumerate(subjects):
            print("---------------------------------")
            print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
            print("---------------------------------\n")

            # Get the data for the subject
            data_subject = data_all_subjects[data_all_subjects["subject"] == int(subject)].reset_index(drop=True)
            # Get the hidden states for the subject
            hidden_states_subject = hidden_states_file[hidden_states_file["subject"] == int(subject)].reset_index(
                drop=True
            )
            # Get the annotations for the subject
            annotations_subject = annotations[annotations["subject"] == int(subject)].reset_index(drop=True)

            # Delete subject and duplicate timestamp columns
            data_subject = data_subject.drop(columns=["subject"])
            annotations_subject = annotations_subject.drop(columns=["subject", "timestamp"])
            hidden_states_subject = hidden_states_subject.drop(columns=["subject", "timestamp"])

            # Check if the timestamps have the same length
            if len(data_subject) != len(annotations_subject) != len(hidden_states_subject):
                # Cut the longer one to the length of the shorter one
                min_length = min(len(data_subject), len(annotations_subject), len(hidden_states_subject))
                data_subject = data_subject[:min_length]
                annotations_subject = annotations_subject[:min_length]
                hidden_states_subject = hidden_states_subject[:min_length]

            # Merge the hidden states with the data and the annotations
            data = pd.concat([data_subject, hidden_states_subject, annotations_subject], axis=1)

            # Get a list of all features
            features = [col for col in data.columns if col not in ["timestamp", "state"]]

            # Sort the features alphabetically
            features = sorted(features)

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

                # Sort the summary stats by the hidden states
                summary_stats = summary_stats.sort_index()

                # Check if the summary stats are calculated for all states
                if len(summary_stats) != number_of_states:
                    # Add a row with NaNs for the missing states
                    for state in range(number_of_states):
                        if state not in summary_stats.index:
                            summary_stats.loc[state, :] = [np.nan] * len(summary_stats.columns)

                        # Sort the summary stats by the hidden states
                        summary_stats = summary_stats.sort_index()

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
                data_state = data[data["state"] == state]
                data_state = data_state.reset_index(drop=True)

                # Check if the state is empty (state is not visited)
                if len(data_state) == 0:
                    # Add NaNs to the respective columns
                    global_stats.loc[state, "fractional_occupancy"] = np.nan
                    global_stats.loc[state, "mean_lifetime"] = np.nan
                    global_stats.loc[state, "mean_intervaltime"] = np.nan
                    continue

                # Fractional Occupancy
                fractional_occupancy = len(data_state) / len(data)
                global_stats.loc[state, "fractional_occupancy"] = fractional_occupancy

                # Lifetime & Intervaltime
                lifetime = 0
                lifetimes = []
                intervaltimes = []
                # Loop over all timepoints
                for index, timepoint in enumerate(data_state["timestamp"]):
                    # Check if the timepoint is the first one
                    if index == 0:
                        lifetime = 0
                    elif timepoint == data_state["timestamp"][index - 1] + 1:
                        lifetime += 1
                        if index == len(data_state) - 1:
                            lifetimes.append(lifetime)
                            lifetime = 0
                        elif timepoint != data_state["timestamp"][index + 1] - 1:
                            lifetimes.append(lifetime)
                            lifetime = 0
                    else:
                        lifetimes.append(lifetime)
                        lifetime = 0
                        intervaltime = timepoint - data_state["timestamp"][index - 1]
                        intervaltimes.append(intervaltime)

                # Calculate the mean lifetime
                mean_lifetime = 0 if len(lifetimes) == 0 else np.mean(lifetimes)
                global_stats.loc[state, "mean_lifetime"] = mean_lifetime
                # Calculate the mean intervaltime
                mean_intervaltime = 0 if len(intervaltimes) == 0 else np.mean(intervaltimes)
                global_stats.loc[state, "mean_intervaltime"] = mean_intervaltime

            # Add the subject ID to the global stats as first column
            global_stats.insert(0, "subject", subject)

            # Append the global stats for the subject to the global stats for all subjects
            global_stats_all_subjects = pd.concat([global_stats_all_subjects, global_stats], axis=0)

        # Add the state to the summary stats as second column
        summary_stats_all_subjects.insert(
            1, "state", np.tile(np.arange(number_of_states), len(subjects) * len(features))
        )

        # Reset the index of the summary stats for all subjects
        summary_stats_all_subjects = summary_stats_all_subjects.reset_index(drop=True)

        # Save the summary stats for all subjects to a file
        resultpath_model = resultpath / "avg" / "hmm" / model
        # Create the directory if it does not exist
        resultpath_model.mkdir(parents=True, exist_ok=True)

        summary_stats_all_subjects.to_csv(
            resultpath_model / f"all_subjects_task-AVR_{model}_model_states_stats.tsv", sep="\t", index=False
        )

        # Add the state to the global stats as second column
        global_stats_all_subjects.insert(1, "state", np.tile(np.arange(number_of_states), len(subjects)))

        # Reset the index of the global stats for all subjects
        global_stats_all_subjects = global_stats_all_subjects.reset_index(drop=True)

        # Save the global stats for all subjects to a file
        global_stats_all_subjects.to_csv(
            resultpath_model / f"all_subjects_task-AVR_{model}_model_states_global_stats.tsv", sep="\t", index=False
        )

        # %% STEP 3. CALCULATE AVERAGED STATS
        # Calculate the statistics for each hidden state averaged across all subjects
        summary_stats_averaged = pd.DataFrame()
        global_stats_averaged = pd.DataFrame()
        for state in range(number_of_states):
            print(f"Calculating averaged stats for state {state}...")

            summary_stats = pd.DataFrame()
            # Calculate the mean and standard deviation for each feature
            summary_stats["mean"] = (
                summary_stats_all_subjects[summary_stats_all_subjects["state"] == state]
                .groupby("feature")["mean"]
                .mean()
            )
            summary_stats["std"] = (
                summary_stats_all_subjects[summary_stats_all_subjects["state"] == state]
                .groupby("feature")["std"]
                .mean()
            )
            # Calculate the minimum and maximum for each feature
            summary_stats["min"] = (
                summary_stats_all_subjects[summary_stats_all_subjects["state"] == state]
                .groupby("feature")["min"]
                .mean()
            )
            summary_stats["max"] = (
                summary_stats_all_subjects[summary_stats_all_subjects["state"] == state]
                .groupby("feature")["max"]
                .mean()
            )
            summary_stats.insert(0, "state", state)

            # Sort the summary stats by the features
            summary_stats = summary_stats.sort_index()

            # Append the summary stats for the state to the summary stats for all states
            summary_stats_averaged = pd.concat([summary_stats_averaged, summary_stats], axis=0)

            # Calculate the global statistics for each state
            global_stats = pd.DataFrame()
            # Calculate the mean fractional occupancy
            global_stats.loc[state, "mean_fractional_occupancy"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["fractional_occupancy"].mean()

            # Calculate the std of the fractional occupancy
            global_stats.loc[state, "std_fractional_occupancy"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["fractional_occupancy"].std()

            # Calculate the min of the fractional occupancy
            global_stats.loc[state, "min_fractional_occupancy"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["fractional_occupancy"].min()

            # Calculate the max of the fractional occupancy
            global_stats.loc[state, "max_fractional_occupancy"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["fractional_occupancy"].max()

            # Calculate the mean lifetime
            global_stats.loc[state, "mean_lifetime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_lifetime"].mean()

            # Calculate the std of the lifetime
            global_stats.loc[state, "std_lifetime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_lifetime"].std()

            # Calculate the min of the lifetime
            global_stats.loc[state, "min_lifetime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_lifetime"].min()

            # Calculate the max of the lifetime
            global_stats.loc[state, "max_lifetime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_lifetime"].max()

            # Calculate the mean intervaltime
            global_stats.loc[state, "mean_intervaltime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_intervaltime"].mean()

            # Calculate the std of the intervaltime
            global_stats.loc[state, "std_intervaltime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_intervaltime"].std()

            # Calculate the min of the intervaltime
            global_stats.loc[state, "min_intervaltime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_intervaltime"].min()

            # Calculate the max of the intervaltime
            global_stats.loc[state, "max_intervaltime"] = global_stats_all_subjects[
                global_stats_all_subjects["state"] == state
            ]["mean_intervaltime"].max()

            global_stats.insert(0, "state", state)

            # Append the global stats for the state to the global stats for all states
            global_stats_averaged = pd.concat([global_stats_averaged, global_stats], axis=0)

        # Add a column with the feature names
        summary_stats_averaged.insert(1, "feature", np.tile(features, number_of_states))

        # Reset the index of the summary stats for all states
        summary_stats_averaged = summary_stats_averaged.reset_index(drop=True)

        # Save the summary stats for all states to a file
        summary_stats_averaged.to_csv(
            resultpath_model / f"avg_task-AVR_{model}_model_states_stats.tsv", sep="\t", index=False
        )

        # Reset the index of the global stats for all states
        global_stats_averaged = global_stats_averaged.reset_index(drop=True)

        # Save the global stats for all states to a file
        global_stats_averaged.to_csv(
            resultpath_model / f"avg_task-AVR_{model}_model_states_global_stats.tsv", sep="\t", index=False
        )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    hmm_stats()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
