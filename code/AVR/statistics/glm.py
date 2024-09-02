"""
Script to calculate a general linear model (GLM) to compare features across hidden affective states.

Required packages: statsmodels, scipy, sklearn, scikit-bio

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 15 August 2024
Last update: 2 September 2024
"""
# %%
def glm(  # noqa: PLR0915, C901
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    subjects=["001", "002", "003","004", "005", "006", "007", "009",  # noqa: B006
        "012", "014", "015", "016", "018", "019",
        "020", "021", "022", "024", "025", "026", "027", "028", "029",
        "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
        "040", "041", "042", "043", "045", "046"],
    debug=False,
    show_plots=False,
):
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
        3.1 MANOVA for each subject
        3.2 Perform a non-parametric MANOVA (PERMANOVA) for the case of non-normality
    4. Perform second level GLM (one-sample t-test)
    5. Perform post-hoc tests to see between which states the features differ
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    from numpy.linalg import LinAlgError
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from skbio.stats.distance import DistanceMatrix, permanova
    from statsmodels.multivariate.manova import MANOVA

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    resultpath = Path(results_dir) / "phase3" / "AVR"

    # Which HMMs to analyze
    models = ["cardiac", "neural", "integrated", "subjective", "multimodal"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "subjective": ["valence", "arousal"],
        "multimodal": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma",
                    "valence", "arousal"],
    }

    # Set the significance level
    alpha = 0.05

    # Only analyze one subject if debug is True
    if debug:
        subjects = [subjects[0]]

    # %% Functions  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    def test_assumptions(data, list_variables, group_name):  # noqa: C901, PLR0915, PLR0912
        """
        Test the assumptions of the MANOVA.

        The assumptions are:
        1. Normal distribution of data within groups (Shapiro-Wilk test)
        2. Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
        3. Linearity

        Arguments:
        ---------
        data : pd.DataFrame
            The data to be tested
        list_variables : list
            The list of variables to be tested
        group_name : str
            The name of the column containing the group information

        Returns:
        -------
        table_normality : pd.DataFrame
            Table with the results of the Shapiro-Wilk test
        fig_normality : plt.Figure
            Figure with histograms of the variables for each group and p-value of the Shapiro-Wilk test
        table_homogeinity : pd.DataFrame
            Table with the results of Barlett's and Levene's tests
        """
        # 1. Normal distribution of data within groups (Shapiro-Wilk test)
        # Loop over groups
        # Get the list of groups
        list_groups = data[group_name].unique()
        # Sort the groups
        list_groups.sort()
        # Create a matrix of plots for each group and variable
        # Determine the number of rows and columns for the subplot grid
        num_cols = len(list_variables)
        num_rows = len(list_groups)

        # Create a figure and a grid of subplots
        fig_normality, axes = plt.subplots(num_rows, num_cols, figsize=(num_rows * 5, num_cols * 3))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Initialize a counter for the current axis
        ax_counter = 0

        # Create a table with the results of the Shapiro-Wilk test
        table_normality = pd.DataFrame()
        for group in list_groups:
            data_group = data[data[group_name] == group]
            for variable in list_variables:
                # Perform Shapiro-Wilk test
                shapiro_test = stats.shapiro(data_group[variable])

                # Plot histogram on the current axis
                data_group[variable].plot.hist(ax=axes[ax_counter])

                # Add labels to the plot
                if ax_counter % num_cols == 0:
                    axes[ax_counter].set_ylabel(f"{group}")
                elif ax_counter % num_cols == len(list_groups) - 1:
                    axes[ax_counter].set_ylabel("")
                    # Add number of participants to the side of the plot
                    axes[ax_counter].text(
                        1.2,
                        0.5,
                        f"n = {len(data_group)}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axes[ax_counter].transAxes,
                    )
                else:  # no title
                    axes[ax_counter].set_ylabel("")

                first_subplot_last_row = num_rows * num_cols - num_cols
                if ax_counter >= first_subplot_last_row:
                    axes[ax_counter].set_xlabel(f"{variable}")

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
                        "State": group,
                        "Variable": f"{variable}",
                        "Statistic": shapiro_test[0],
                        "Samples": len(data_group),
                        "p-value": shapiro_test[1],
                        "Significance": shapiro_test[1] < alpha,
                    },
                    ignore_index=True,
                )

        # Set the title of the figure
        fig_normality.suptitle(
            "Histograms of variables for each group and p-value of Shapiro-Wilk-Test to check Normality of data"
        )

        # Round p-values to three decimal places
        table_normality["p-value"] = table_normality["p-value"].round(3)
        # Round all other values except the p-values to two decimal places
        table_normality[["Statistic"]] = table_normality[["Statistic"]].round(2)

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()

        # 2. Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
        # Create a table with the results of Barlett's and Levene's tests
        table_homogeinity = pd.DataFrame()
        for variable in list_variables:
            # Initialize empty array to store data for each group
            data_groups = []

            for group in list_groups:
                # subset data
                data_group = data[(data[f"{group_name}"] == group)][variable]
                data_group.reset_index(drop=True)
                # Transform data to 1D array
                data_group = np.array(data_group)

                # Append data to data_groups
                data_groups.append(data_group)

            # Perform Barlett's test
            bartlett_stats, bartlett_p = stats.bartlett(*data_groups)

            # Perform Levene's test
            levene_stats, levene_p = stats.levene(*data_groups)

            # Append results to table_homogeinity
            table_homogeinity = table_homogeinity._append(
                    {
                        "Variable": f"{variable}",
                        "Test": "Barlett",
                        "Statistic": bartlett_stats,
                        "p-value": bartlett_p,
                        "Samples": [len(data_group) for data_group in data_groups],
                    },
                    ignore_index=True,
                )
            table_homogeinity = table_homogeinity._append(
                    {
                        "Variable": f"{variable}",
                        "Test": "Levene",
                        "Statistic": levene_stats,
                        "p-value": levene_p,
                        "Samples": [len(data_state) for data_state in data_groups],
                    },
                    ignore_index=True,
                )

        # Round p-values to three decimal places
        table_homogeinity["p-value"] = table_homogeinity["p-value"].round(3)
        # Round all other values except the p-values to two decimal places
        table_homogeinity[["Statistic"]] = table_homogeinity[["Statistic"]].round(2)

        # Mark significant results
        table_homogeinity["Significance"] = table_homogeinity["p-value"] < alpha

        # 3. Linearity
        # Create list to save figures
        figs_linearity = []
        # Create scatterplot matrix for each group with all variables
        for group in list_groups:
            data_group = data[data[f"{group_name}"] == group]
            # Drop columns
            data_scatter_group = data_group.drop(columns=[group_name, "timestamp"])

            try:
                fig_linearity, axis = plt.subplots(figsize=(10, 10))
                pd.plotting.scatter_matrix(data_scatter_group, ax=axis, diagonal="kde")
                fig_linearity.suptitle(f"Scatterplot matrix for state {group} (n = {len(data_group)})")

                # Show the plot
                if show_plots:
                    plt.show()

                plt.close()

                figs_linearity.append(fig_linearity)

            except LinAlgError:
                print(f"LinAlgError: Cannot create scatterplot matrix for participant {subject}")
                return table_normality, fig_normality, table_homogeinity, None

        return table_normality, fig_normality, table_homogeinity, figs_linearity

    def post_hoc_tests(data, variables, groups, variable_name, alpha=0.05) -> pd.DataFrame:
        """
        Perform post-hoc tests to compare variables between the videos.

        Arguments:
        ---------
        data: dataframe
        variables: list of variables to be compared
        groups: list of groups to be compared
        variable_name: name of the variable
        alpha: significance level

        Returns:
        -------
        results_table_posthoc_tests: dataframe
        """
        # Perform pairwise comparisons between groups
        # Bonferroni correction for multiple comparisons
        # Perform t-tests for each variable and statistic
        # Create table with group-comparisons and t-test results
        results_table_posthoc_tests = pd.DataFrame()
        for variable in variables:
            for group1 in groups:
                for group2 in groups:
                    if group1 != group2:
                        data_group1 = data[data[f"{variable_name}"] == group1][variable]
                        data_group2 = data[data[f"{variable_name}"] == group2][variable]
                        # Perform t-test
                        ttest_stats, ttest_p = stats.ttest_rel(data_group1, data_group2)
                        # Append results to results_table
                        results_table_posthoc_tests = results_table_posthoc_tests._append(
                            {
                                "Variable": f"{variable}",
                                "Group1": group1,
                                "Sample1": len(data_group1),
                                "Group2": group2,
                                "Sample2": len(data_group2),
                                "t-statistic": ttest_stats,
                                "p-value": ttest_p,
                            },
                            ignore_index=True,
                        )

        # Bonferroni correction for multiple comparisons
        # Number of comparisons
        n_comparisons = len(groups) * (len(groups) - 1) * len(variables)

        # Corrected alpha level
        alpha_corrected = alpha / n_comparisons

        # Mark significant results
        results_table_posthoc_tests["Significance"] = results_table_posthoc_tests["p-value"] < alpha_corrected

        # Round p-values to three decimal places
        results_table_posthoc_tests["p-value"] = results_table_posthoc_tests["p-value"].round(3)
        # Round statistics to two decimal places
        results_table_posthoc_tests[["t-statistic"]] = results_table_posthoc_tests[["t-statistic"]].round(2)

        return results_table_posthoc_tests


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
        features_string_integrated = "_".join(models_features["integrated"])
        hidden_states_file = f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv"
        hidden_states_data = pd.read_csv(hmm_path / hidden_states_file, sep="\t")

        all_data_file = f"all_subjects_task-AVR_integrated_model_data_{features_string_integrated}.tsv"
        all_data = pd.read_csv(resultpath / "avg" / "hmm" / "integrated" / all_data_file, sep="\t")

        features_annotations = "_".join(models_features["subjective"])
        annotation_file = f"all_subjects_task-AVR_subjective_model_data_{features_annotations}.tsv"
        annotations = pd.read_csv(resultpath / "avg" / "hmm" / "subjective" / annotation_file, sep="\t")

        # Check if the length per subject is the same
        if len(annotations) != len(all_data):
            # Shorten the annotations dataframe to the length of the all_data dataframe
            new_annotations = []
            for subject in annotations["subject"].unique():
                # Get the length of the data for the subject
                length_data = len(all_data[all_data["subject"] == subject])
                # Get the length of the annotations for the subject
                length_annotations = len(annotations[annotations["subject"] == subject])
                # Check if the lengths are different
                if length_data != length_annotations:
                    # Shorten the annotations dataframe to the length of the all_data dataframe
                    annotations_subject = annotations[annotations["subject"] == subject][:length_data]
                    new_annotations.append(annotations_subject)

            annotations_short = pd.concat(new_annotations)

        # Add valence and arousal to the data
        all_data["valence"] = annotations_short["valence"]
        all_data["arousal"] = annotations_short["arousal"]

        # Drop the state column
        all_data = all_data.drop(columns=["state"])

        # Add the state column of the respective model
        all_data["state"] = hidden_states_data["state"]

        # Initialize the results dataframe
        results_manova = pd.DataFrame(
            columns=["subject", "test", "value", "num_df", "den_df", "F", "p-value", "significance"]
        )

        results_permanova = pd.DataFrame(
            columns=["subject", "test", "num_groups", "n_samples", "n_permutations", "F", "p-value", "significance"]
        )

        # Loop over all subjects
        for subject_index, subject in enumerate(subjects):
            print("---------------------------------")
            print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")
            print("---------------------------------\n")

            # Get the data for the subject
            hidden_states_subject = hidden_states_data[hidden_states_data["subject"] == int(subject)].reset_index(
                drop=True
            )

            # Delete subject column
            data = hidden_states_subject.drop(columns=["subject"])

            # Get list of states
            list_states = data["state"].unique()
            list_states.sort()

            # Get the list of features
            list_features = models_features[model].copy()
            list_features.sort()

            # STEP 2: TEST ASSUMPTIONS OF MANOVA
            print("Testing assumptions of MANOVA...")
            table_normality, figure_normality, table_homogeinity, figures_linearity = test_assumptions(
                data, list_features, "state"
            )

            # Save the results
            glm_subject_results_path = resultpath / f"sub-{subject}" / "glm" / model

            # Create the directory if it does not exist
            glm_subject_results_path.mkdir(parents=True, exist_ok=True)

            table_normality.to_csv(
                glm_subject_results_path / f"sub-{subject}_task-AVR_{model}_model_glm_normality_results.tsv",
                sep="\t",
                index=False,
            )
            table_homogeinity.to_csv(
                glm_subject_results_path / f"sub-{subject}_task-AVR_{model}_model_glm_homogeneity_results.tsv",
                sep="\t",
                index=False,
            )

            if figures_linearity is not None:
                # Save the figures
                figure_normality.savefig(
                    glm_subject_results_path / f"sub-{subject}_task-AVR_{model}_model_glm_normality_results.png"
                )
                for index, figure in enumerate(figures_linearity):
                    figure.savefig(
                        glm_subject_results_path
                        / f"sub-{subject}_task-AVR_{model}_model_glm_linearity_results_state_{index}.png"
                    )

            # STEP 3: FIRST LEVEL GLM
            print("Performing first level GLM...")

            # Create the formula for the GLM
            # Rename features to avoid problems with the formula
            list_features = [f.replace("-", "_") for f in list_features]
            data = data.rename(columns={f: f.replace("-", "_") for f in data.columns})
            list_variables = "+".join(list_features)

            # 3.1 Perform MANOVA for each subject
            manova_model = MANOVA.from_formula(f"{list_variables} ~ state", data=data)

            # Add the results to the results dataframe
            # Add the results to the dataframe
            for index_test_statistic, test_statistic in enumerate(
                ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]
            ):
                index_row = (
                    len(["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"])
                    * subject_index
                    + index_test_statistic
                )
                results_manova.loc[index_row, "subject"] = subject
                results_manova.loc[index_row, "test"] = test_statistic
                results_manova.loc[index_row, "value"] = manova_model.mv_test().results["state"]["stat"]["Value"][
                    test_statistic
                ]
                results_manova.loc[index_row, "num_df"] = manova_model.mv_test().results["state"]["stat"]["Num DF"][
                    test_statistic
                ]
                results_manova.loc[index_row, "den_df"] = manova_model.mv_test().results["state"]["stat"]["Den DF"][
                    test_statistic
                ]
                results_manova.loc[index_row, "F"] = manova_model.mv_test().results["state"]["stat"]["F Value"][
                    test_statistic
                ]
                results_manova.loc[index_row, "p-value"] = manova_model.mv_test().results["state"]["stat"]["Pr > F"][
                    test_statistic
                ]
                results_manova.loc[index_row, "significance"] = results_manova.loc[index_row, "p-value"] < alpha

            # 3.2 Perform a non-parametric MANOVA (PERMANOVA) for the case of non-normality
            # Create a distance matrix (Euclidean distance) for the features
            # Copy of the data with features only
            data_features = data[list_features]
            distance_matrix = DistanceMatrix(squareform(pdist(data_features, metric="euclidean")), ids=data.index)

            # Perform PERMANOVA
            permanova_model = permanova(distance_matrix, data["state"].reset_index(drop=True), permutations=1000)

            # Add the results to the results dataframe
            results_permanova.loc[subject_index, "subject"] = subject
            results_permanova.loc[subject_index, "test"] = permanova_model["method name"]
            results_permanova.loc[subject_index, "num_groups"] = permanova_model["number of groups"]
            results_permanova.loc[subject_index, "n_samples"] = permanova_model["sample size"]
            results_permanova.loc[subject_index, "n_permutations"] = permanova_model["number of permutations"]
            results_permanova.loc[subject_index, "F"] = permanova_model["test statistic"]
            results_permanova.loc[subject_index, "p-value"] = permanova_model["p-value"]
            results_permanova.loc[subject_index, "significance"] = results_permanova.loc[subject_index,
                                                                    "p-value"] < alpha

        # Save the results
        glm_results_path = resultpath / "avg" / "glm" / model
        # Create the directory if it does not exist
        glm_results_path.mkdir(parents=True, exist_ok=True)
        results_manova.to_csv(
            glm_results_path / f"all_subjects_task-AVR_{model}_model_glm_results_manova.tsv", sep="\t", index=False
        )

        results_permanova.to_csv(
            glm_results_path / f"all_subjects_task-AVR_{model}_model_glm_results_permanova.tsv", sep="\t", index=False
        )

        # %% STEP 4: SECOND LEVEL GLM
        # Test the effect of the hidden states on the features across all subjects (with significant results)
        print("Performing second level GLM...")

        # Perform a one-sample t-test to test if the features are significantly different in the hidden states
        # Compare the MANOVA results to a t-test against 0 (no effect)
        results_ttest_manova = pd.DataFrame()
        results_ttest_permanova = pd.DataFrame()

        # Calculate the t-test for the MANOVA results
        for index_test_statistic, test_statistic in enumerate(
            ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]
        ):
            # Extract the values and convert to numeric, coercing errors to NaN
            values_manova = pd.to_numeric(results_manova[results_manova["test"] ==
                                        test_statistic]["value"], errors="coerce").dropna()

            # Log-transform the values
            log_values_manova = np.log(values_manova)

            # Perform a one-sample t-test
            tstats = scipy.stats.ttest_1samp(log_values_manova, 0, alternative="two-sided")
            results_ttest_manova.loc[index_test_statistic, "test"] = test_statistic
            results_ttest_manova.loc[index_test_statistic, "t-value"] = tstats.statistic
            results_ttest_manova.loc[index_test_statistic, "df"] = tstats.df
            results_ttest_manova.loc[index_test_statistic, "p-value"] = tstats.pvalue
            results_ttest_manova.loc[index_test_statistic, "significance"] = (
                results_ttest_manova.loc[index_test_statistic, "p-value"] < alpha
            )

        # Calculate the t-test for the PERMANOVA results
        # Extract the values and convert to numeric, coercing errors to NaN
        values_permanova = pd.to_numeric(results_permanova["F"], errors="coerce").dropna()

        # Log-transform the values
        log_values_permanova = np.log(values_permanova)

        # Perform a one-sample t-test
        tstats = scipy.stats.ttest_1samp(log_values_permanova, 0, alternative="two-sided")
        results_ttest_permanova.loc[0, "test"] = "PERMANOVA"
        results_ttest_permanova.loc[0, "t-value"] = tstats.statistic
        results_ttest_permanova.loc[0, "df"] = tstats.df
        results_ttest_permanova.loc[0, "p-value"] = tstats.pvalue
        results_ttest_permanova.loc[0, "significance"] = results_ttest_permanova.loc[0, "p-value"] < alpha

        # Save the results
        results_ttest_manova.to_csv(
            glm_results_path / f"avg_task-AVR_{model}_model_glm_results_manova_ttest.tsv", sep="\t", index=False
        )

        results_ttest_permanova.to_csv(
            glm_results_path / f"avg_task-AVR_{model}_model_glm_results_permanova_ttest.tsv", sep="\t", index=False
        )

        # %% STEP 5: POST-HOC TESTS
        # Perform post-hoc tests to see between which states the features differ
        print("Performing post-hoc tests...")

        # Get the list of features to perform the post-hoc tests
        list_features = all_data.columns.tolist()
        list_features.remove("state")
        list_features.remove("subject")
        list_features.remove("timestamp")

        # Calculate the mean of the features for each state for each subject
        feature_data_mean = all_data.groupby(["subject", "state"]).mean()

        # Drop the timestamp column
        feature_data_mean = feature_data_mean.drop(columns=["timestamp"])
        # Add a column with states from the index
        feature_data_mean["state"] = feature_data_mean.index.get_level_values("state")
        # Add a column with subjects from the index
        feature_data_mean["subject"] = feature_data_mean.index.get_level_values("subject")
        feature_data_mean = feature_data_mean.reset_index(drop=True)

        # Get only the significant subjects
        significant_subjects_manova = results_manova[results_manova["significance"]]["subject"].unique()
        significant_subjects_permanova = results_permanova[results_permanova["significance"]]["subject"].unique()
        # Transform the subjects to integers
        significant_subjects_manova = [int(subject) for subject in significant_subjects_manova]
        significant_subjects_permanova = [int(subject) for subject in significant_subjects_permanova]

        # Get the corresponding data
        feature_data_manova = feature_data_mean[feature_data_mean["subject"].isin(significant_subjects_manova)]
        feature_data_permanova = feature_data_mean[feature_data_mean["subject"].isin(significant_subjects_permanova)]

        # Perform post-hoc tests to see between which states the features differ
        results_table_posthoc_tests_manova = post_hoc_tests(feature_data_manova, list_features, list_states,
                                            "state", alpha)
        results_table_posthoc_tests_permanova = post_hoc_tests(feature_data_permanova, list_features, list_states,
                                            "state", alpha)

        # Save the results
        results_table_posthoc_tests_manova.to_csv(
            glm_results_path / f"avg_task-AVR_{model}_model_glm_results_manova_posthoc_tests.tsv", sep="\t",
            index=False
        )
        results_table_posthoc_tests_permanova.to_csv(
            glm_results_path / f"avg_task-AVR_{model}_model_glm_results_permanova_posthoc_tests.tsv", sep="\t",
            index=False
        )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    glm()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
