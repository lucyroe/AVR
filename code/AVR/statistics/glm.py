"""
Script to calculate a general linear model (GLM) to compare features across hidden affective states.

Required packages: statsmodels, scipy

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 15 August 2024
Last update: 16 August 2024
"""

def glm(  # noqa: PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003"],  # noqa: B006
    debug=False):
    """
    Fit all features from the HMMs to a general linear model (GLM) to compare features across hidden affective states.

    Inputs: Features with hidden states from the HMMs

    Outputs: GLM results (first level and second level)

    Steps:
    1. Load data
    2. Test assumptions of MANOVA
        2.1 Normal distribution of data within groups (Shapiro-Wilk test)
        2.2 Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
        2.3 Linearity
    3. Perform first level GLM (MANOVA)
    4. Perform second level GLM (one-sample t-test)
    """
    # %% Import
    from pathlib import Path

    import pandas as pd
    import scipy
    from statsmodels.multivariate.manova import MANOVA

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    annotation_path = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "features" / "avg" / "beh"
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to analyze
    models = ["cardiac"]
    #"neural", "integrated"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
    }

    # Set the significance level
    alpha = 0.05

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% STEP 1: LOAD DATA
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Fitting GLM for {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        # Load the features with hidden states
        hmm_path = resultpath / "avg" / "hmm" / model
        features_string = "_".join(models_features[model])
        hidden_states_file = f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv"
        hidden_states_data = pd.read_csv(hmm_path / hidden_states_file, sep="\t")

        # Load the annotations
        annotation_file = f"all_subjects_task-AVR_beh_features.tsv"
        annotations = pd.read_csv(annotation_path / annotation_file, sep="\t")

        # %% STEP 2: TEST ASSUMPTIONS OF MANOVA TODO
        # 1. Normal distribution of data within groups (Shapiro-Wilk test)
        # Loop over groups
        # Create a matrix of plots for each group and variable
        # Determine the number of rows and columns for the subplot grid
        num_rows = len(data_all["group"].unique())
        num_cols = len(variable_names[phases[1]]) * len(test_statistics)

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Initialize a counter for the current axis
        ax_counter = 0

        # Create a table with the results of the Shapiro-Wilk test
        table_normality = pd.DataFrame()
        for group in data_all["group"].unique():
            data_group = data_all[data_all["group"] == group]
            for variable in variable_names[phases[1]]:
                for test_statistic in test_statistics:
                    # Perform Shapiro-Wilk test
                    shapiro_test = stats.shapiro(data_group[test_statistic + "_" + variable])

                    # Plot histogram on the current axis
                    data_group[test_statistic + "_" + variable].plot.hist(ax=axes[ax_counter])

                    # Add labels to the plot
                    if ax_counter % num_cols == 0:
                        axes[ax_counter].set_ylabel(f"{group}")
                    elif ax_counter % num_cols == 3:
                        axes[ax_counter].set_ylabel("")
                        # Add number of participants to the side of the plot
                        axes[ax_counter].text(
                            1.2,
                            0.5,
                            f"n = {len(subjects) if group == phases[1] else len(subjects_phase1)}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[ax_counter].transAxes,
                        )
                    else:  # no title
                        axes[ax_counter].set_ylabel("")
                    first_subplot_last_row = num_rows * num_cols - num_cols
                    if ax_counter >= first_subplot_last_row:
                        axes[ax_counter].set_xlabel(f"{variable} {test_statistic}")

                    # Make p-value bold and put it onto red ground if it is below significance level
                    if shapiro_test[1] < alpha:
                        axes[ax_counter].text(
                            0.5,
                            0.5,
                            f"p = {shapiro_test[1]:.3f}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[ax_counter].transAxes,
                            weight="bold",
                            backgroundcolor="red",
                        )
                    else:
                        axes[ax_counter].text(
                            0.5,
                            0.5,
                            f"p = {shapiro_test[1]:.3f}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[ax_counter].transAxes,
                            backgroundcolor="white",
                        )

                    # Increment the axis counter
                    ax_counter += 1

                    # Append results to table_normality
                    table_normality = table_normality._append(
                        {
                            "Group": group,
                            "Variable": f"{test_statistic} {variable}",
                            "Statistic": shapiro_test[0],
                            "Samples": len(data_group),
                            "p-value": shapiro_test[1],
                            "Significance": shapiro_test[1] < alpha,
                        },
                        ignore_index=True,
                    )

        # Set the title of the figure
        fig.suptitle("Histograms of variables for each group and p-value of Shapiro-Wilk-Test to check Normality of data")

        # Switch from scientific notation to fixed notation
        pd.options.display.float_format = "{:.5f}".format

        # Round p-values to three decimal places
        table_normality["p-value"] = table_normality["p-value"].round(3)
        # Round all other values except the p-values to two decimal places
        table_normality[["Statistic"]] = table_normality[["Statistic"]].round(2)

        # Save the table as a tsv file
        table_normality.to_csv(Path(results_dir_comparison) / "normality_test.tsv", sep="\t")

        # Save the plot
        fig.savefig(Path(results_dir_comparison) / "histograms_normality.png")

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()

        # 2. Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
        # Create a table with the results of Barlett's and Levene's tests
        table_homogeinity = pd.DataFrame()
        for variable in variable_names[phases[1]]:
            for test_statistic in test_statistics:
                # subset data
                data_phase1_hp = data_all[(data_all["group"] == "phase1 HP")][test_statistic + "_" + variable]
                data_phase1_hn = data_all[(data_all["group"] == "phase1 HN")][test_statistic + "_" + variable]
                data_phase1_lp = data_all[(data_all["group"] == "phase1 LP")][test_statistic + "_" + variable]
                data_phase1_ln = data_all[(data_all["group"] == "phase1 LN")][test_statistic + "_" + variable]
                data_other_phase = data_all[(data_all["group"] == phases[1])][test_statistic + "_" + variable]

                # Perform Barlett's test
                bartlett_stats, bartlett_p = stats.bartlett(
                    data_phase1_hp, data_phase1_hn, data_phase1_lp, data_phase1_ln, data_other_phase
                )

                # Perform Levene's test
                levene_stats, levene_p = stats.levene(
                    data_phase1_hp, data_phase1_hn, data_phase1_lp, data_phase1_ln, data_other_phase
                )

                # Append results to table_homogeinity
                table_homogeinity = table_homogeinity._append(
                    {
                        "Variable": f"{test_statistic} {variable}",
                        "Test": "Barlett",
                        "Statistic": bartlett_stats,
                        "p-value": bartlett_p,
                        "Samples": [len(data_phase1_hp), len(data_phase1_hn), len(data_phase1_lp), len(data_phase1_ln),
                        len(data_other_phase)]
                    },
                    ignore_index=True,
                )
                table_homogeinity = table_homogeinity._append(
                    {
                        "Variable": f"{test_statistic} {variable}",
                        "Test": "Levene",
                        "Statistic": levene_stats,
                        "p-value": levene_p,
                        "Samples": [len(data_phase1_hp), len(data_phase1_hn), len(data_phase1_lp), len(data_phase1_ln),
                        len(data_other_phase)]
                    },
                    ignore_index=True,
                )

        # Switch from scientific notation to fixed notation
        pd.options.display.float_format = "{:.5f}".format

        # Round p-values to three decimal places
        table_homogeinity["p-value"] = table_homogeinity["p-value"].round(3)
        # Round all other values except the p-values to two decimal places
        table_homogeinity[["Statistic"]] = table_homogeinity[["Statistic"]].round(2)

        # Mark significant results
        table_homogeinity["Significance"] = table_homogeinity["p-value"] < alpha

        # Save the table as a tsv file
        table_homogeinity.to_csv(Path(results_dir_comparison) / "homogeneity_test.tsv", sep="\t")

        # 3. Linearity
        # Create scatterplot matrix for each group with all variables
        for group in data_all["group"].unique():
            data_group = data_all[data_all["group"] == group]
            # Drop columns
            data_scatter_group = data_group.drop(columns=["group", "subject", "n_samples"])
            figure, axis = plt.subplots(figsize=(10, 10))
            pd.plotting.scatter_matrix(data_scatter_group, figsize=(10, 10), ax=axis, diagonal="kde")
            figure.suptitle(f"Scatterplot matrix for {group} (n = "
            f"{len(subjects) if group == phases[1] else len(subjects_phase1)})")

            # Save the plot
            figure.savefig(Path(results_dir_comparison) / f"scatterplot_matrix_{group}.png")

            # Show the plot
            if show_plots:
                plt.show()

            plt.close()

        # %% STEP 3: FIRST LEVEL GLM
        print("Performing first level GLM...")

        # Initialize the results dataframe
        results = pd.DataFrame(columns=["subject", "test", "value", "num_df", "den_df", "F", "p-value", "significance"])

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

            # Create the formula for the GLM
            list_features = models_features[model] + ["valence", "arousal"]
            # Rename features to avoid problems with the formula
            list_features = [f.replace("-", "_") for f in list_features]
            data = data.rename(columns={f: f.replace("-", "_") for f in data.columns})
            list_variables = "+".join(list_features)

            manova_model = MANOVA.from_formula(f"{list_variables} ~ state", data=data)

            # Add the results to the results dataframe
            # Add the results to the dataframe
            for index_test_statistic, test_statistic in enumerate(["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]):
                index_row = len(["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"])*subject_index+index_test_statistic
                results.loc[index_row, "subject"] = subject
                results.loc[index_row, "test"] = test_statistic
                results.loc[index_row, "value"] = manova_model.mv_test().results["state"]["stat"]["Value"][test_statistic]
                results.loc[index_row, "num_df"] = manova_model.mv_test().results["state"]["stat"]["Num DF"][test_statistic]
                results.loc[index_row, "den_df"] = manova_model.mv_test().results["state"]["stat"]["Den DF"][test_statistic]
                results.loc[index_row, "F"] = manova_model.mv_test().results["state"]["stat"]["F Value"][test_statistic]
                results.loc[index_row, "p-value"] = manova_model.mv_test().results["state"]["stat"]["Pr > F"][test_statistic]
                results.loc[index_row, "significance"] = results.loc[index_row, "p-value"] < alpha

        # Save the results
        glm_results_path = resultpath / "avg" / "glm"
        # Create the directory if it does not exist
        glm_results_path.mkdir(parents=True, exist_ok=True)
        results.to_csv(glm_results_path / f"all_subjects_task-AVR_{model}_model_glm_results.tsv", sep="\t", index=False)

        # %% STEP 4: SECOND LEVEL GLM
        # Test the effect of the hidden states on the features across all subjects
        print("Performing second level GLM...")

        # Perform a one-sample t-test to test if the features are significantly different in the hidden states
        # Compare the MANOVA results to a t-test against 0 (no effect)
        results_ttest = pd.DataFrame()
        for index_test_statistic, test_statistic in enumerate(["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]):
            # Extract the values and convert to numeric, coercing errors to NaN
            values = pd.to_numeric(results[results["test"] == test_statistic]["value"], errors='coerce').dropna()

            # Perform a one-sample t-test
            tstats = scipy.stats.ttest_1samp(values, 0, alternative="two-sided")
            results_ttest.loc[index_test_statistic, "test"] = test_statistic
            results_ttest.loc[index_test_statistic, "t-value"] = tstats.statistic
            results_ttest.loc[index_test_statistic, "df"] = tstats.df
            results_ttest.loc[index_test_statistic, "p-value"] = tstats.pvalue
            results_ttest.loc[index_test_statistic, "significance"] = results_ttest.loc[index_test_statistic, "p-value"] < alpha

        # Save the results
        results_ttest.to_csv(glm_results_path / f"avg_task-AVR_{model}_model_glm_results_ttest.tsv", sep="\t", index=False)

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    glm()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END