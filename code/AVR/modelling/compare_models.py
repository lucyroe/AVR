"""
Script to compare the different Hidden Markov Model (HMM) models.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 19 August 2024
Last update: 19 August 2024
"""


def compare_models(
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003"],
    debug=False,
    show_plots=False,
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
    import json
    from pathlib import Path
    import pickle

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from hmmlearn.hmm import GMMHMM

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to compare
    models = ["cardiac", "neural", "integrated"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
    }

    # Number of states (four quadrants of the Affect Grid) = hyperparameter that needs to be chosen
    number_of_states = 4

    color_features = "black"
    # Colors of the hidden states
    colors_states = ["#009E73", "#CC79A7", "#0072B2", "#D55E00"]  # green, pink, dark blue, dark orange

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    pass

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1. GET MODELS AND DATA
    # Initialize dataframes to store the results
    df_model_quality = pd.DataFrame(columns=["log-likelihood", "AIC", "BIC"], rows=models)
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Loading {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        modelpath = resultpath / "phase3" / "AVR" / "avg" / "hmm" / model
        features_string = "_".join(models_features[model])
        with modelpath / f"all_subjects_task-AVR_{model}_model_{features_string}.pkl" as f:
            hmm = pickle.load(f)

        # Get the data
        data = pd.read_csv(modelpath / f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv", sep="\t")

        # Drop unnecessary columns
        data.drop(columns=["subject", "timestamp", "state"])

        # %% STEP 2. ASSESS RELATIVE MODEL QUALITY
        print("Assessing relative model quality...\n")

        # 2a. Log-likelihood
        log_likelihood = hmm.score(data, lengths=[len(data)])

        # 2b. AIC
        number_of_parameters = hmm.n_components ** 2 + 2 * hmm.n_components * data.shape[1] - 1
        aic = 2 * number_of_parameters - 2 * log_likelihood

        # 2c. BIC
        bic = np.log(len(data)) * number_of_parameters - 2 * log_likelihood

        # Store the results
        df_model_quality.loc[model] = [log_likelihood, aic, bic]

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    compare_models()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
