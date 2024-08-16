"""
Script to read in and calculate univariate statistics of the participants of AVR phase 3.

Required packages: statsmodels, scipy, pingouin

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 9 August 2024
Last update: 16 August 2024
"""


def univariate_statistics(  # noqa: C901, PLR0912, PLR0915
    subjects=["001", "002", "003","004", "005", "006", "007", "009",  # noqa: B006
                "011", "012", "014", "015", "016", "017", "018", "019",
                "020", "021", "022", "024", "025", "026", "027", "028", "029",
                "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
                "040", "041", "042", "043", "044", "045", "046", "047"],
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    debug=False,
    show_plots=False,
):
    """
    Calculate univariate statistics.

    Inputs:     Pre-survey questionnaire data
                Preprocessed annotation data
                Preprocessed physiological data (EEG, ECG)

    Outputs:    Demographics statistics
                Annotation data statistics
                Physiological data statistics

    Steps:
    1. LOAD & FORMAT DATA
        1a. Load & Format Questionnaire Data
        1b. Load & Format Annotation Data
        1c. Load & Format Event Markers
        1d. Load & Format Physiological Data
    2. TEST ASSUMPTIONS
        2a. rm ANOVA for Annotation Data: Normality (Shapiro-Wilk), Sphericity (Mauchly's Test)
        2b. rm ANOVA for Physiological Data: Normality (Shapiro-Wilk), Sphericity (Mauchly's Test)
    3. CALCULATE DESCRIPTIVE STATISTICS
        3a. Demographics
        3b. Annotation Data
        3c. Physiological Data

    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import pingouin as pg
    from scipy import stats
    from statsmodels.stats.anova import AnovaRM

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # Only analyze one subject when debug mode is on
    if debug:
        subjects = [subjects[0]]

    # Specify the data path info (in BIDS format)
    # Change with the directory of data storage
    data_dir = Path(data_dir) / "phase3"
    exp_name = "AVR"
    derivative_name = "derivatives"  # derivates folder
    preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
    features_name = "features"  # features folder (inside preprocessed
    averaged_name = "avg"  # averaged data folder (inside preprocessed)
    results_dir = Path(results_dir) / "phase3"

    # List of different videos
    videos = ["spaceship", "invasion", "asteroids", "underwood"]

    # List of EEG features to be analyzed
    features = ["posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"]

    # Significance level
    alpha = 0.05

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
    def test_assumptions(data, group_name, variable_names, alpha=0.05) -> pd.DataFrame:
        """
        Test the assumptions for a repeated measures ANOVA.

        1. Normal distribution of data within groups (Shapiro-Wilk test)
        2. Sphericity of data (Mauchly's Test)

        Arguments:
        ---------
        data: dataframe
        group_name: name of the group variable
        variable_names: list of variables to be compared
        alpha: significance level

        Returns:
        -------
        table_normality: dataframe
        fig: figure
        table_sphericity: dataframe
        """
        # 1. Normal distribution of data within groups (Shapiro-Wilk test)
        # Loop over groups
        # Create a matrix of plots for each group and variable
        # Determine the number of rows and columns for the subplot grid
        num_rows = len(videos)
        num_cols = len(variable_names)

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 2))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Initialize a counter for the current axis
        ax_counter = 0

        # Create a table with the results of the Shapiro-Wilk test
        table_normality = pd.DataFrame()
        for group in videos:
            data_group = data[data[group_name] == group]
            for variable in variable_names:
                # Perform Shapiro-Wilk test
                shapiro_test = stats.shapiro(data_group[variable])

                # Plot histogram on the current axis
                data_group[variable].plot.hist(ax=axes[ax_counter])

                # Add labels to the plot
                if ax_counter % num_cols == 0:
                    axes[ax_counter].set_ylabel(f"{group}")
                elif ax_counter % num_cols == len(videos) - 2:
                    axes[ax_counter].set_ylabel("")
                    # Add number of participants to the side of the plot
                    axes[ax_counter].text(
                        1.2,
                        0.5,
                        f"n = {len(subjects)}",
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
                        "Group": group,
                        "Variable": f"{variable}",
                        "Statistic": shapiro_test[0],
                        "Samples": len(data_group),
                        "p-value": shapiro_test[1],
                        "Significance": shapiro_test[1] < alpha,
                    },
                    ignore_index=True,
                )

        # Set the title of the figure
        fig.suptitle(
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

        # 2. Sphericity of data (Mauchly's Test)
        # Perform Mauchly's Test for each variable
        table_sphericity = pd.DataFrame()
        for variable in variable_names:
            spher, w, chi2, dof, pval = pg.sphericity(data, dv=variable, subject="subject", within=group_name)

            # Append results to table_sphericity
            table_sphericity = table_sphericity._append(
                {
                    "Variable": f"{variable}",
                    "Sphericity": spher,
                    "W": w,
                    "Chi2": chi2,
                    "df": dof,
                    "p-value": pval,
                    "Significance": pval < alpha,
                },
                ignore_index=True,
            )

        return table_normality, fig, table_sphericity

    def descriptives(data, variables, groups, group_name) -> pd.DataFrame:
        """
        Calculate descriptive statistics for the data.

        Arguments:
        ---------
        data: dataframe
        variables: list of variables to be compared
        groups: list of groups to be compared
        group_name: name of the group variable

        Returns:
        -------
        results_descriptives: dataframe
        """
        results_descriptives = {}
        for variable in variables:
            stats_group = pd.DataFrame()
            for group in groups:
                stats_group_variable = data[data[f"{group_name}"] == group][variable].describe()
                stats_group.loc[f"mean_{variable}", group] = stats_group_variable["mean"]
                stats_group.loc[f"std_{variable}", group] = stats_group_variable["std"]
                stats_group.loc[f"min_{variable}", group] = stats_group_variable["min"]
                stats_group.loc[f"max_{variable}", group] = stats_group_variable["max"]

            stats_overall = data[variable].describe()
            stats_group.loc[f"mean_{variable}", "overall"] = stats_overall["mean"]
            stats_group.loc[f"std_{variable}", "overall"] = stats_overall["std"]
            stats_group.loc[f"min_{variable}", "overall"] = stats_overall["min"]
            stats_group.loc[f"max_{variable}", "overall"] = stats_overall["max"]

            # Add the results to the dataframe
            results_descriptives[variable] = stats_group

        # Concatenate the results of the descriptives
        return pd.concat(results_descriptives, axis=0)

    def perform_anova(data, variables, group_name, alpha=0.05) -> pd.DataFrame:
        """
        Perform repeated measures ANOVA to test for significant differences in variables between the groups.

        Arguments:
        ---------
        data: dataframe
        variables: list of variables to be compared
        group_name: name of the group variable
        alpha: significance level

        Returns:
        -------
        results_annotation_anova: dataframe
        """
        # Create a dataframe to store the results
        results_annotation_anova = pd.DataFrame()

        # Perform repeated measures ANOVA to test for significant differences in variables between the groups
        for variable in variables:
            anova = AnovaRM(data, variable, "subject", within=[group_name], aggregate_func="mean")
            results = anova.fit()

            # Add the results to the dataframe
            results_annotation_anova.loc[f"rmANOVA {variable}", "F"] = results.anova_table["F Value"][0]
            results_annotation_anova.loc[f"rmANOVA {variable}", "Num DF"] = results.anova_table["Num DF"][0]
            results_annotation_anova.loc[f"rmANOVA {variable}", "Den DF"] = results.anova_table["Den DF"][0]
            results_annotation_anova.loc[f"rmANOVA {variable}", "p"] = results.anova_table["Pr > F"][0]

        # Add column with significance
        results_annotation_anova["Significance"] = results_annotation_anova["p"] < alpha

        return results_annotation_anova

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
                        # Interpolate the data if the lengths of the two datasets do not match
                        if len(data_group1) != len(data_group2):
                            required_length = max(len(data_group1), len(data_group2))
                            data_group1 = data_group1.reset_index(drop=True)
                            data_group2 = data_group2.reset_index(drop=True)
                            data_group1 = data_group1.reindex(range(required_length)).interpolate()
                            data_group2 = data_group2.reindex(range(required_length)).interpolate()
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

    # %% STEP 1. LOAD & FORMAT DATA

    # 1a. Load Questionnaire Data
    questionnaire_filepath = data_dir / exp_name / "data_avr_phase3_pre_survey_2024-08-08_13-49.csv"
    data_questionnaire = pd.read_csv(questionnaire_filepath, sep="\t", header=1, encoding="utf-16")

    participant_ids = data_questionnaire["ID Number (free text)"]

    # Get age and gender
    age = data_questionnaire["Age.1"]
    gender = data_questionnaire["Gender"]  # 1: female; 2: male; 3: non-binary

    # Add all variablees to a dataframe
    demographics = pd.DataFrame()
    demographics["subject"] = participant_ids
    demographics["age"] = age
    demographics["gender"] = gender

    # Save dataframe as tsv file
    demographics.to_csv(data_dir / exp_name / "demographics.tsv", sep="\t", index=False)

    # 1b. Load Annotation Data
    # Initialize dataframe to store annotation data
    annotation_data = pd.DataFrame()

    annotation_filepath = (
        data_dir
        / exp_name
        / derivative_name
        / preprocessed_name
        / averaged_name
        / "beh"
        / "all_subjects_task-AVR_beh_preprocessed.tsv"
    )
    annotation_data_file = pd.read_csv(annotation_filepath, sep="\t")

    subjects_list = annotation_data_file["subject"]
    timestamps = annotation_data_file["timestamp"]

    # Get valence and arousal ratings
    valence = annotation_data_file["valence"]
    arousal = annotation_data_file["arousal"]

    # Add all variables to a dataframe
    annotation_data["subject"] = subjects_list
    annotation_data["timestamp"] = timestamps
    annotation_data["valence"] = valence
    annotation_data["arousal"] = arousal

    # 1c. Load Event Markers
    event_filepath = data_dir / exp_name / derivative_name / preprocessed_name / "events_experiment.tsv"
    event_data = pd.read_csv(event_filepath, sep="\t")

    timestamps_events = event_data["onset"]

    # Get event markers
    events = event_data["event_name"]

    # Assign the corresponding video name to each period of the data
    list_subjects = []
    for subject in annotation_data["subject"].unique():
        subject_data = annotation_data[annotation_data["subject"] == subject]
        for video in videos:
            counter = 0
            for row in subject_data.iterrows():
                # Get the timestamps of the start and end of the video
                timestamp_start_video = timestamps_events[events == f"start_{video}"].reset_index()["onset"][counter]
                timestamp_stop_video = timestamps_events[events == f"end_{video}"].reset_index()["onset"][counter]
                if row[1]["timestamp"] >= timestamp_start_video and row[1]["timestamp"] <= timestamp_stop_video:
                    subject_data.loc[row[0], "video"] = video
                if video == "spaceship" and row[1]["timestamp"] >= timestamp_stop_video:
                    counter += 1
        list_subjects.append(subject_data)

    # Concatenate all subjects
    annotation_data = pd.concat(list_subjects)

    # Get any rows with nan values in the "video" variable (periods of fade-ins/fade-outs in between the videos)
    nan_rows_annotation = annotation_data[annotation_data.isna().any(axis=1)]
    # Count the number of nan rows
    print(f"Number of nan rows in annotation data (fade-ins/fade-outs of the videos): {len(nan_rows_annotation)}")
    # These rows will later be ignored in calculating the descriptive statistics

    annotation_features_dir = data_dir / exp_name / derivative_name / features_name / averaged_name / "beh"
    # Create directory if it does not exist
    annotation_features_dir.mkdir(parents=True, exist_ok=True)
    # Save annotation data to features folder
    annotation_data.to_csv(annotation_features_dir / "all_subjects_task-AVR_beh_features.tsv", sep="\t", index=False)

    # 1d. Load Physiological Data
    # Initialize dataframes to store physiological data
    list_ecg_data = []
    list_eeg_data = []
    # Loop over all subjects
    for subject in subjects:
        subject_datapath = data_dir / exp_name / derivative_name / features_name / f"sub-{subject}" / "eeg"
        ecg_data_subject = pd.read_csv(subject_datapath / f"sub-{subject}_task-AVR_ecg_features.tsv", sep="\t")
        # Add subject ID to the dataframe as first column
        ecg_data_subject.insert(0, "subject", subject)

        # Rename columns so they don't have a - in the name
        ecg_data_subject = ecg_data_subject.rename(columns={"lf-hrv": "lf_hrv", "hf-hrv": "hf_hrv"})

        # Initialize dataframe to store feature data
        list_features_subject = pd.DataFrame()
        # Loop over physiological features
        for feature in features:
            if len(feature.split("_")) > 1:
                region = feature.split("_")[0]
                band = feature.split("_")[1]
                # Get the data
                filename = f"sub-{subject}_task-AVR_eeg_features_{region}_power.tsv"
                data = pd.read_csv(subject_datapath / filename, sep="\t")
                feature_data = data[band]
                list_features_subject[feature] = feature_data
            else:
                # Get the data
                filename = f"sub-{subject}_task-AVR_eeg_features_whole-brain_power.tsv"
                data = pd.read_csv(subject_datapath / filename, sep="\t")
                feature_data = data[feature]
                list_features_subject[feature] = feature_data

        # Add timestamp and subject ID to the dataframe
        list_features_subject.insert(0, "subject", subject)
        list_features_subject.insert(1, "timestamp", data["timestamp"])

        # Compare the lengths of the timestamp columns of both dataframes
        if len(ecg_data_subject["timestamp"]) != len(list_features_subject["timestamp"]):
            print(
                f"The length of the timestamp columns of the ECG and EEG dataframes of subject {subject} do not match."
            )
            print("The longer dataframe will be truncated to the length of the shorter dataframe.")
            # Truncate the longer dataframe to the length of the shorter dataframe
            min_length = min(len(ecg_data_subject["timestamp"]), len(list_features_subject["timestamp"]))
            ecg_data_subject = ecg_data_subject[:min_length]
            list_features_subject = list_features_subject[:min_length]

        list_ecg_data.append(ecg_data_subject)
        list_eeg_data.append(list_features_subject)

    # Concatenate all subjects
    ecg_data = pd.concat(list_ecg_data)
    eeg_data = pd.concat(list_eeg_data)

    # Concatenate both dataframes horizontally
    physiological_data = pd.concat([ecg_data, eeg_data], axis=1)
    # Remove duplicate columns
    physiological_data = physiological_data.loc[:, ~physiological_data.columns.duplicated()]

    # Assign the corresponding video name to each period of the data
    list_subjects = []
    for subject in physiological_data["subject"].unique():
        subject_data = physiological_data[physiological_data["subject"] == subject]
        for video in videos:
            counter = 0
            for row in subject_data.iterrows():
                # Get the timestamps of the start and end of the video
                timestamp_start_video = timestamps_events[events == f"start_{video}"].reset_index()["onset"][counter]
                timestamp_stop_video = timestamps_events[events == f"end_{video}"].reset_index()["onset"][counter]
                if row[1]["timestamp"] >= timestamp_start_video and row[1]["timestamp"] <= timestamp_stop_video:
                    subject_data.loc[row[0], "video"] = video
                if video == "spaceship" and row[1]["timestamp"] >= timestamp_stop_video:
                    counter += 1
        list_subjects.append(subject_data)

    # Concatenate all subjects
    physiological_data = pd.concat(list_subjects)

    # Get any rows with nan values in the "video" variable (periods of fade-ins/fade-outs in between the videos)
    nan_rows_physiological = physiological_data[physiological_data.isna().any(axis=1)]
    # Count the number of nan rows
    print(
        f"Number of nan rows in physiological data (fade-ins/fade-outs of the videos): {len(nan_rows_physiological)}"
    )
    # These rows will later be ignored in calculating the descriptive statistics

    physiological_features_dir = data_dir / exp_name / derivative_name / features_name / averaged_name / "eeg"
    # Create directory if it does not exist
    physiological_features_dir.mkdir(parents=True, exist_ok=True)
    # Save physiological data to features folder
    physiological_data.to_csv(
        physiological_features_dir / "all_subjects_task-AVR_physio_features.tsv", sep="\t", index=False
    )
    # %% STEP 2. TEST ASSUMPTIONS
    # 2a. For Annotation Data
    # Test the assumptions for a repeated measures ANOVA
    table_normality_annotation, fig_normality_annotation, table_sphericity_annotation = test_assumptions(
        annotation_data, "video", ["valence", "arousal"], alpha
    )

    # Save the results of the Shapiro-Wilk test as a tsv file
    table_normality_annotation.to_csv(
        results_dir / exp_name / averaged_name / "stats" / "annotation_normality.tsv", sep="\t", index=False
    )

    # Save the results of the Mauchly's Test as a tsv file
    table_sphericity_annotation.to_csv(
        results_dir / exp_name / averaged_name / "stats" / "annotation_sphericity.tsv", sep="\t", index=False
    )

    # Save the figure
    fig_normality_annotation.savefig(results_dir / exp_name / averaged_name / "stats" / "annotation_normality.png")

    # 2b. For Physiological Data
    # Test the assumptions for a repeated measures ANOVA
    table_normality_physiological, fig_normality_physiological, table_sphericity_physiological = test_assumptions(
        physiological_data, "video", physiological_data.columns.unique()[2:-1], alpha
    )

    # Save the results of the Shapiro-Wilk test as a tsv file
    table_normality_physiological.to_csv(
        results_dir / exp_name / averaged_name / "stats" / "physiological_normality.tsv", sep="\t", index=False
    )

    # Save the results of the Mauchly's Test as a tsv file
    table_sphericity_physiological.to_csv(
        results_dir / exp_name / averaged_name / "stats" / "physiological_sphericity.tsv", sep="\t", index=False
    )

    # Save the figure
    fig_normality_physiological.savefig(
        results_dir / exp_name / averaged_name / "stats" / "physiological_normality.png"
    )

    # %% STEP 3. CALCULATE DESCRIPTIVE STATISTICS

    # Define results directory
    results_filepath_stats = results_dir / exp_name / averaged_name / "stats"

    # Create directory if it does not exist
    results_filepath_stats.mkdir(parents=True, exist_ok=True)

    # 2a. Demographics
    # Calculate descriptive statistics of age
    age_stats = demographics["age"].describe()

    # Calculate descriptive statistics of gender
    gender_stats = demographics["gender"].value_counts()

    # Add all stats to a dataframe
    demographics_stats = pd.DataFrame()
    demographics_stats.loc[0, "number_of_participants"] = age_stats["count"]
    demographics_stats.loc[0, "mean_age"] = age_stats["mean"]
    demographics_stats.loc[0, "std_age"] = age_stats["std"]
    demographics_stats.loc[0, "min_age"] = age_stats["min"]
    demographics_stats.loc[0, "max_age"] = age_stats["max"]
    demographics_stats.loc[0, "number_female"] = gender_stats[1]
    demographics_stats.loc[0, "number_male"] = gender_stats[2]
    demographics_stats.loc[0, "number_non_binary"] = gender_stats[3]

    # Save dataframe as tsv file
    demographics_stats.to_csv(results_filepath_stats / "demographics_stats.tsv", sep="\t", index=False)

    # 2b. Annotation Data
    # Calculate descriptive statistics of valence and arousal ratings
    video_stats_annotation = descriptives(annotation_data, ["valence", "arousal"], videos, "video")

    # Save dataframe as tsv file
    video_stats_annotation.to_csv(results_filepath_stats / "annotation_stats.tsv", sep="\t", index=True)

    # Perform repeated measures ANOVA to test for differences in valence & arousal ratings between the videos
    results_annotation_anova = perform_anova(annotation_data, ["valence", "arousal"], "video", alpha)

    # Save results to a tsv file
    results_annotation_anova.to_csv(results_filepath_stats / "annotation_anova.tsv", sep="\t", index=True)

    # Post hoc tests
    results_table_posthoc_tests_annotation = post_hoc_tests(
        annotation_data, ["valence", "arousal"], videos, "video", alpha
    )

    # Save the results of the post-hoc tests as a tsv file
    results_table_posthoc_tests_annotation.to_csv(results_filepath_stats / "annotation_posthoc_results.tsv", sep="\t")

    # 2c. Physiological Data
    # Calculate descriptive statistics of physiological features
    video_stats_physiological = descriptives(
        physiological_data, physiological_data.columns.unique()[2:-1], videos, "video"
    )

    # Save dataframe as tsv file
    video_stats_physiological.to_csv(results_filepath_stats / "physiological_stats.tsv", sep="\t", index=True)

    # Perform repeated measures ANOVA to test for significant differences in physiological features between the videos
    results_physiological_anova = perform_anova(
        physiological_data, physiological_data.columns.unique()[2:-1], "video", alpha
    )

    # Save results to a tsv file
    results_physiological_anova.to_csv(results_filepath_stats / "physiological_anova.tsv", sep="\t", index=True)

    # Post hoc tests
    results_table_posthoc_tests_physiological = post_hoc_tests(
        physiological_data, physiological_data.columns.unique()[2:-1], videos, "video", alpha
    )

    # Save the results of the post-hoc tests as a tsv file
    results_table_posthoc_tests_physiological.to_csv(
        results_filepath_stats / "physiological_posthoc_results.tsv", sep="\t"
    )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    univariate_statistics()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
