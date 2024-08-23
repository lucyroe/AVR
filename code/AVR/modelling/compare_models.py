"""
Script to compare the different Hidden Markov Model (HMM) models.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 19 August 2024
Last update: 23 August 2024
"""


# %%
def compare_models(  # noqa: PLR0915
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003","004", "005", "006", "007", "009",  # noqa: B006
        "012", "014", "015", "016", "018", "019",
        "020", "021", "022", "024", "025", "026", "027", "028", "029",
        "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
        "040", "041", "042", "043", "045", "046"],
    debug=False,
):
    """
    Compare the different Hidden Markov Model (HMM) models.

    Inputs: TODO

    Outputs:
    - TODO

    Functions:
    - TODO

    Steps:
    1. TODO
    """
    # %% Import
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to compare
    models = ["cardiac", "neural", "integrated", "subjective"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "subjective": ["valence", "arousal"],
    }

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    pass

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. GET MODELS AND DATA
    # Initialize dataframes to store the results
    df_model_quality = pd.DataFrame(columns=["log-likelihood", "AIC", "BIC"])
    df_accuracy = pd.DataFrame(columns=["correlation", "accuracy"])
    df_distance = pd.DataFrame(columns=["distance_valence_state_0", "distance_arousal_state_0",
        "distance_valence_state_1", "distance_arousal_state_1", "distance_valence_state_2", "distance_arousal_state_2",
        "distance_valence_state_3", "distance_arousal_state_3", "distance_valence_mean", "distance_arousal_mean",
        "distance_mean"])

    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Loading {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        modelpath = resultpath / "avg" / "hmm" / model
        features_string = "_".join(models_features[model])
        with (Path(modelpath) / f"all_subjects_task-AVR_{model}_model_{features_string}.pkl").open("rb") as f:
            hmm = pickle.load(f)

        # Get the data
        data = pd.read_csv(modelpath / f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv", sep="\t")

        # Drop unnecessary columns
        data = data.drop(columns=["subject", "timestamp", "state"])

        # %% STEP 2. ASSESS RELATIVE MODEL QUALITY
        print("Assessing relative model quality...\n")

        # 2a. Log-likelihood
        log_likelihood = hmm.score(data, lengths=[len(data)])

        # 2b. AIC
        number_of_parameters = hmm.n_components**2 + 2 * hmm.n_components * data.shape[1] - 1
        aic = 2 * number_of_parameters - 2 * log_likelihood

        # 2c. BIC
        bic = np.log(len(data)) * number_of_parameters - 2 * log_likelihood

        # Store the results
        df_model_quality.loc[model] = [log_likelihood, aic, bic]

        # %% STEP 3. ASSESS ACCURACY OF THE MODELS
        print("Assessing accuracy of the models...\n")

        # 3a. Get the hidden states
        hidden_states_file = f"all_subjects_task-AVR_{model}_model_hidden_states_{features_string}.tsv"
        hidden_states_model = pd.read_csv(modelpath / hidden_states_file, sep="\t")

        # 3b. Get the hidden states identified by the subjective model
        hidden_states_subjective_file = "all_subjects_task-AVR_subjective_model_hidden_states_valence_arousal.tsv"
        hidden_states_subjective = pd.read_csv(
            resultpath / "avg" / "hmm" / "subjective" / hidden_states_subjective_file, sep="\t"
        )

        hidden_states_model_formatted = pd.DataFrame(columns=["subject", "timestamp", "state"])
        hidden_states_subjective_formatted = pd.DataFrame(columns=["subject", "timestamp", "state"])
        # Check if the lengths of the hidden states are the same
        for subject in hidden_states_model["subject"].unique():
            # Get the subject's hidden states
            hidden_states_model_subject = hidden_states_model[hidden_states_model["subject"] == subject]
            hidden_states_subjective_subject = hidden_states_subjective[hidden_states_subjective["subject"] == subject]
            if len(hidden_states_model_subject) != len(hidden_states_subjective_subject):
                # Cut the longer one
                min_length = min(len(hidden_states_model_subject), len(hidden_states_subjective_subject))
                hidden_states_model_subject = hidden_states_model_subject[:min_length]
                hidden_states_subjective_subject = hidden_states_subjective_subject[:min_length]
            hidden_states_model_formatted = pd.concat([hidden_states_model_formatted, hidden_states_model_subject])
            hidden_states_subjective_formatted = pd.concat(
                [hidden_states_subjective_formatted, hidden_states_subjective_subject]
            )

        hidden_states_model = hidden_states_model_formatted.reset_index(drop=True)
        hidden_states_subjective = hidden_states_subjective_formatted.reset_index(drop=True)

        # 3c. Compare the hidden states
        correlation = hidden_states_model["state"].corr(hidden_states_subjective["state"])
        accuracy = np.mean(hidden_states_model["state"] == hidden_states_subjective["state"])

        # Store the results
        df_accuracy.loc[model] = [correlation, accuracy]

        # %% STEP 4. CALCULATE DISTANCE BETWEEN RATINGS
        print("Calculating distance between ratings...\n")

        # 4a. Get the ratings
        ratings_file = "all_subjects_task-AVR_subjective_model_data_valence_arousal.tsv"
        ratings = pd.read_csv(resultpath / "avg" / "hmm" / "subjective" / ratings_file, sep="\t")

        # Drop unnecessary columns
        ratings = ratings.drop(columns=["subject", "timestamp", "state"])

        # Combine the ratings with the hidden states
        ratings_model = pd.concat([hidden_states_model, ratings], axis=1)
        ratings_subjective = pd.concat([hidden_states_subjective, ratings], axis=1)

        # 4b. Calculate the distance between the ratings
        for state in range(4):
            # Get the ratings for the current state
            ratings_model_state = ratings_model[ratings_model["state"] == state]
            ratings_subjective_state = ratings_subjective[ratings_subjective["state"] == state]

            # Get the mean values
            mean_model = ratings_model_state.mean()[["valence", "arousal"]]
            mean_subjective = ratings_subjective_state.mean()[["valence", "arousal"]]

            # Calculate the distance
            distance_valence = mean_model["valence"] - mean_subjective["valence"]
            distance_arousal = mean_model["arousal"] - mean_subjective["arousal"]

            # Store the results
            df_distance.loc[model, f"distance_valence_state_{state}"] = distance_valence
            df_distance.loc[model, f"distance_arousal_state_{state}"] = distance_arousal

        # 4c. Calculate the distance between the ratings for all states
        # Calculate the mean
        mean_distance_valence = df_distance.loc[model, ["distance_valence_state_0", "distance_valence_state_1",
            "distance_valence_state_2", "distance_valence_state_3"]].mean()
        mean_distance_arousal = df_distance.loc[model, ["distance_arousal_state_0", "distance_arousal_state_1",
            "distance_arousal_state_2", "distance_arousal_state_3"]].mean()
        mean_distance = (mean_distance_valence + mean_distance_arousal) / 2
        # Store the results
        df_distance.loc[model, "distance_valence_mean"] = mean_distance_valence
        df_distance.loc[model, "distance_arousal_mean"] = mean_distance_arousal
        df_distance.loc[model, "distance_mean"] = mean_distance


    # %% STEP 4. SAVE THE RESULTS
    # Save the results
    df_model_quality.to_csv(resultpath / "avg" / "hmm" / "model_quality.tsv", sep="\t")
    df_accuracy.to_csv(resultpath / "avg" / "hmm" / "model_accuracy.tsv", sep="\t")
    df_distance.to_csv(resultpath / "avg" / "hmm" / "model_distance.tsv", sep="\t")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    compare_models()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
