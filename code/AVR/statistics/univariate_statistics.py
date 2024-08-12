"""
Script to read in and calculate univariate statistics of the participants of AVR phase 3.

Required packages: statsmodels, scipy

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 9 August 2024
Last update: 12 August 2024
"""

def univariate_statistics(  # noqa: C901, PLR0912, PLR0915
    subjects=["001", "002", "003"],  # noqa: B006
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    debug=False,
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
    2. CALCULATE UNIVARIATE STATISTICS
        2a. Demographics
        2b. Annotation Data
        2c. Physiological Data

    """
    # %% Import
    from pathlib import Path

    import pandas as pd
    from scipy import stats
    from statsmodels.multivariate.manova import MANOVA
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

    def perform_manova(data, variables, group_name, alpha=0.05) -> pd.DataFrame:
        """
        Perform multi-factor ANOVA (MANOVA) to test for significant differences in variables between the groups.

        Arguments:
        ---------
        data: dataframe
        variables: list of variables to be compared
        group_name: name of the group variable
        alpha: significance level

        Returns:
        -------
        results_annotation_manova: dataframe
        """
        # Create a dataframe to store the results
        results_annotation_manova = pd.DataFrame()

        # Perform multi-factor ANOVA (MANOVA) for all variables together
        list_variables = "+".join(variables)
        manova = MANOVA.from_formula(f"{list_variables} ~ {group_name}", data=data, aggregate_func="mean")
        results = manova.mv_test()

        # Add the results to the dataframe
        results_annotation_manova.loc["MANOVA Wilks' lambda", "Value"] = results.results[group_name]["stat"]["Value"][
            "Wilks' lambda"
        ]
        results_annotation_manova.loc["MANOVA Wilks' lambda", "Num DF"] = results.results[group_name]["stat"][
            "Num DF"
        ]["Wilks' lambda"]
        results_annotation_manova.loc["MANOVA Wilks' lambda", "Den DF"] = results.results[group_name]["stat"][
            "Den DF"
        ]["Wilks' lambda"]
        results_annotation_manova.loc["MANOVA Wilks' lambda", "F"] = results.results[group_name]["stat"]["F Value"][
            "Wilks' lambda"
        ]
        results_annotation_manova.loc["MANOVA Wilks' lambda", "p"] = results.results[group_name]["stat"]["Pr > F"][
            "Wilks' lambda"
        ]
        results_annotation_manova.loc["MANOVA Pillai's trace", "Value"] = results.results[group_name]["stat"]["Value"][
            "Pillai's trace"
        ]
        results_annotation_manova.loc["MANOVA Pillai's trace", "Num DF"] = results.results[group_name]["stat"][
            "Num DF"
        ]["Pillai's trace"]
        results_annotation_manova.loc["MANOVA Pillai's trace", "Den DF"] = results.results[group_name]["stat"][
            "Den DF"
        ]["Pillai's trace"]
        results_annotation_manova.loc["MANOVA Pillai's trace", "F"] = results.results[group_name]["stat"]["F Value"][
            "Pillai's trace"
        ]
        results_annotation_manova.loc["MANOVA Pillai's trace", "p"] = results.results[group_name]["stat"]["Pr > F"][
            "Pillai's trace"
        ]
        results_annotation_manova.loc["MANOVA Hotelling-Lawley trace", "Value"] = results.results[group_name]["stat"][
            "Value"
        ]["Hotelling-Lawley trace"]
        results_annotation_manova.loc["MANOVA Hotelling-Lawley trace", "Num DF"] = results.results[group_name]["stat"][
            "Num DF"
        ]["Hotelling-Lawley trace"]
        results_annotation_manova.loc["MANOVA Hotelling-Lawley trace", "Den DF"] = results.results[group_name]["stat"][
            "Den DF"
        ]["Hotelling-Lawley trace"]
        results_annotation_manova.loc["MANOVA Hotelling-Lawley trace", "F"] = results.results[group_name]["stat"][
            "F Value"
        ]["Hotelling-Lawley trace"]
        results_annotation_manova.loc["MANOVA Hotelling-Lawley trace", "p"] = results.results[group_name]["stat"][
            "Pr > F"
        ]["Hotelling-Lawley trace"]
        results_annotation_manova.loc["MANOVA Roy's greatest root", "Value"] = results.results[group_name]["stat"][
            "Value"
        ]["Roy's greatest root"]
        results_annotation_manova.loc["MANOVA Roy's greatest root", "Num DF"] = results.results[group_name]["stat"][
            "Num DF"
        ]["Roy's greatest root"]
        results_annotation_manova.loc["MANOVA Roy's greatest root", "Den DF"] = results.results[group_name]["stat"][
            "Den DF"
        ]["Roy's greatest root"]
        results_annotation_manova.loc["MANOVA Roy's greatest root", "F"] = results.results[group_name]["stat"][
            "F Value"
        ]["Roy's greatest root"]
        results_annotation_manova.loc["MANOVA Roy's greatest root", "p"] = results.results[group_name]["stat"][
            "Pr > F"
        ]["Roy's greatest root"]

        # Add column with significance
        results_annotation_manova["Significance"] = results_annotation_manova["p"] < alpha

        return results_annotation_manova

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

        # Change formatting of all group variables
        for group in results_table_posthoc_tests["Group1"].unique():
            results_table_posthoc_tests["Group1"] = results_table_posthoc_tests["Group1"].replace(
                group, group.replace("phase1", "Phase 1")
            )
            results_table_posthoc_tests["Group1"] = results_table_posthoc_tests["Group1"].replace(
                group, group.replace("phase3", "Phase 3")
            )
        for group in results_table_posthoc_tests["Group2"].unique():
            results_table_posthoc_tests["Group2"] = results_table_posthoc_tests["Group2"].replace(
                group, group.replace("phase1", "Phase 1")
            )
            results_table_posthoc_tests["Group2"] = results_table_posthoc_tests["Group2"].replace(
                group, group.replace("phase3", "Phase 3")
            )

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

    # %% STEP 2. CALCULATE DESCRIPTIVE STATISTICS

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

    # Perform Multi-factor ANOVA (MANOVA) for both valence and arousal
    results_annotation_manova = perform_manova(annotation_data, ["valence", "arousal"], "video", alpha)

    # Save results to a tsv file
    results_annotation_anova.to_csv(results_filepath_stats / "annotation_anova.tsv", sep="\t", index=True)
    results_annotation_manova.to_csv(results_filepath_stats / "annotation_manova.tsv", sep="\t", index=True)

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

    # Perform Multi-factor ANOVA (MANOVA) for all physiological features
    results_physiological_manova = perform_manova(
        physiological_data, physiological_data.columns.unique()[2:-1], "video", alpha
    )

    # Save results to a tsv file
    results_physiological_anova.to_csv(results_filepath_stats / "physiological_anova.tsv", sep="\t", index=True)
    results_physiological_manova.to_csv(results_filepath_stats / "physiological_manova.tsv", sep="\t", index=True)

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
