"""
Performing statistical tests (MANOVA) to compare the variability between AVR phase 1 and phase 3.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 1 August 2024
Last updated: 7 August 2024
"""

def compare_variability_phase1_phase3(  # noqa: C901, PLR0912, PLR0915
    subjects=["001", "002", "003"],  # noqa: B006
    subjects_phase1=["06", "08", "10", "12", "14", "16", "18", "19", "20",  # noqa: B006
                        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                        "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                        "41", "42", "43", "44", "45", "46", "47", "48", "49", "51",
                        "53", "55", "57", "59", "61", "63", "65", "67", "69", "71", "73", "75"],
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    show_plots=False,
):
    """
    Perform statistical tests (MANOVA) to compare the variability between AVR phase 1 and phase 3.

    The following steps are performed:
    1. Calculate summary statistics
    2. Test assumptions of MANOVA
        2.1 Normal distribution of data within groups (Shapiro-Wilk test)
        2.2 Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
        2.3 Linearity
    3. Perform statistical tests
        3.1 Perform MANOVA
        3.2 Post-hoc tests
            3.2.1 Perform pairwise comparisons between groups
            3.2.2 Bonferroni correction for multiple comparisons
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats
    from statsmodels.multivariate.manova import MANOVA

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    phases = ["phase1", "phase3"]  # Phases for which the raincloud plot should be plotted

    # Path where results should be saved
    results_dir_comparison = Path(results_dir) / f"comparison_{phases[0]}_{phases[1]}"
    # Create directory if it does not exist
    results_dir_comparison.mkdir(parents=True, exist_ok=True)

    quadrants = ["HP", "HN", "LP", "LN"]  # Quadrants/videos in phase 1 for which the raincloud plot should be plotted
    # In phase 2, there was only one video

    variable_names = {phases[0]: ["cr_v", "cr_a"], phases[1]: ["valence", "arousal"]}
    # In phase 1, different rating methods were used
    # We only compare Flubber from phase 1 to Flubber from phase 2 or 3
    test_statistics = ["mean", "std_dev"]  # statistics for which the raincloud plot should be plotted

    # Significance level
    alpha = 0.05

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% LOAD DATA, CALCULATE STATISTICS, AND SAVE AS NEW FILE
    # Create empty dataframe to store data
    data_all = pd.DataFrame()

    for phase in phases:
        # Get name of directory
        if phase == "phase1":
            directory = Path(data_dir) / phase / "preprocessed" / "annotations"
            subject_list = subjects_phase1
        else:
            directory = Path(data_dir) / phase / "AVR" / "derivatives" / "preproc"
            subject_list = subjects

        # Create empty dataframe to store statistics
        statistics = pd.DataFrame()

        # Loop over all subjects
        for subject in subject_list:
            # Load data
            if phase == "phase1":
                data = pd.read_csv(directory / f"sub_{subject}_cr_preprocessed.csv")
            else:
                file = directory / f"sub-{subject}" / "beh" / f"sub-{subject}_task-AVR_beh_preprocessed.tsv"
                data = pd.read_csv(file, sep="\t")

            # Loop over all variables
            for variable_index, variable in enumerate(variable_names[phase]):
                if phase == "phase1":
                    # Delete all rows from the dataframe where the rating method is not Flubber
                    flubber_data = data[data["rating_method"] == "Flubber"]
                    # Loop over all quadrants
                    for quadrant in quadrants:
                        # Get only the data for the current quadrant
                        quadrant_data = flubber_data[flubber_data["quadrant"] == quadrant]

                        # Calculate mean
                        mean_subject = quadrant_data[variable].mean()
                        # Calculate standard deviation
                        stats_subject = quadrant_data[variable].std()
                        # Count number of samples
                        n_samples_subject = quadrant_data[variable].count()

                        # Create new row
                        new_row = {
                            "subject": subject,
                            "phase": phase,
                            "quadrant": quadrant,
                            f"mean_{variable_names[phases[1]][variable_index]}": mean_subject,
                            f"std_dev_{variable_names[phases[1]][variable_index]}": stats_subject,
                            "n_samples": n_samples_subject,
                        }

                        # Append new row to dataframe
                        statistics = statistics._append(new_row, ignore_index=True)

                else:  # for phase 3
                    # Calculate mean
                    mean_subject = data[variable].mean()
                    # Calculate standard deviation
                    stats_subject = data[variable].std()
                    # Count number of samples
                    n_samples_subject = data[variable].count()

                    # Create new row
                    new_row = {
                        "subject": subject,
                        "phase": phase,
                        f"mean_{variable_names[phases[1]][variable_index]}": mean_subject,
                        f"std_dev_{variable_names[phases[1]][variable_index]}": stats_subject,
                        "n_samples": n_samples_subject,
                    }

                    # Append new row to dataframe
                    statistics = statistics._append(new_row, ignore_index=True)

        # Formatting
        if phase == "phase1":
            stats_formatted = pd.DataFrame()
            for subject in subject_list:
                stats_subject = statistics[statistics["subject"] == subject]
                # Merge all variables into one row
                stats_subject_new = stats_subject[: len(quadrants)]
                stats_subject_new["mean_arousal"] = stats_subject["mean_arousal"][len(quadrants):].to_numpy()
                stats_subject_new["std_dev_arousal"] = stats_subject["std_dev_arousal"][len(quadrants):].to_numpy()
                stats_subject_new["mean_valence"] = stats_subject["mean_valence"][:len(quadrants)].to_numpy()
                stats_subject_new["std_dev_valence"] = stats_subject["std_dev_valence"][:len(quadrants)].to_numpy()

                # Append new row to dataframes
                stats_formatted = stats_formatted._append(stats_subject_new, ignore_index=True)

        else:
            stats_formatted = pd.DataFrame()
            for subject in subjects:
                stats_subject = statistics[statistics["subject"] == subject]
                # Merge all variables into one row
                stats_subject_new = stats_subject[:1]
                stats_subject_new["mean_arousal"] = stats_subject["mean_arousal"][1:].to_numpy()
                stats_subject_new["std_dev_arousal"] = stats_subject["std_dev_arousal"][1:].to_numpy()
                stats_subject_new["mean_valence"] = stats_subject["mean_valence"][:1].to_numpy()
                stats_subject_new["std_dev_valence"] = stats_subject["std_dev_valence"][:1].to_numpy()

                # Append new row to dataframes
                stats_formatted = stats_formatted._append(stats_subject_new, ignore_index=True)

        # Sort the dataframe by subject
        stats_sorted = stats_formatted.sort_values(by="subject")

        # Combine phase and quadrant column to create a new column with the group variable (for phase 1)
        if phase == "phase1":
            stats_sorted["group"] = stats_sorted["phase"] + " " + stats_sorted["quadrant"]
            # Drop quadrant and phase column
            stats_sorted = stats_sorted.drop(columns=["quadrant", "phase"])
            # Drop index
            stats_sorted = stats_sorted.reset_index(drop=True)
        else:
            stats_sorted["group"] = stats_sorted["phase"]
            # Drop phase column
            stats_sorted = stats_sorted.drop(columns="phase")

        # Add the data to the data_all dataframe
        data_all = data_all._append(stats_sorted)

        # Save the dataframe as a tsv file
        stats_sorted.to_csv(directory / f"stats_{phase}.tsv", sep="\t", index=False)

    # %% TEST ASSUMPTIONS
    # Test assumptions of MANOVA

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

    # %% PERFORM STATISTICAL TESTS
    # Perform MANOVA
    # Group variables: phase 1 HP, phase 1 HN, phase 1 LP, phase 1 LN, phase 2 or 3
    # Variables: mean_valence, std_dev_valence, mean_arousal, std_dev_arousal

    maov = MANOVA.from_formula(
        "mean_valence + std_dev_valence + mean_arousal + std_dev_arousal ~ group", data=data_all
    )

    # Transform the results of the MANOVA into a table
    results_manova = pd.DataFrame(
        {
            "Value": maov.mv_test().results["group"]["stat"]["Value"],
            "Num_DF": maov.mv_test().results["group"]["stat"]["Num DF"],
            "Den_DF": maov.mv_test().results["group"]["stat"]["Den DF"],
            "F-value": maov.mv_test().results["group"]["stat"]["F Value"],
            "p-value": maov.mv_test().results["group"]["stat"]["Pr > F"],
            "Significance": maov.mv_test().results["group"]["stat"]["Pr > F"] < alpha,
        }
    )

    # Switch from scientific notation to fixed notation
    pd.options.display.float_format = "{:.5f}".format

    for test_statistic in results_manova.index:
        # Round p-values to three decimal places
        results_manova.loc[test_statistic, "p-value"] = round(results_manova.loc[test_statistic, "p-value"], 3)
        # Round statistics to two decimal places
        results_manova.loc[test_statistic, "Value"] = round(results_manova.loc[test_statistic, "Value"], 2)
        results_manova.loc[test_statistic, "F-value"] = round(results_manova.loc[test_statistic, "F-value"], 2)
        # Add number of samples to the table
        results_manova.loc[test_statistic, "Samples"] = [len(subjects_phase1) * 4 + len(subjects)]
    # Round degrees of freedom to zero decimal places
    results_manova[["Num_DF", "Den_DF"]] = results_manova[["Num_DF", "Den_DF"]].astype(int)

    # Save the results of the MANOVA as a tsv file
    results_manova.to_csv(Path(results_dir_comparison) / "manova_results.tsv", sep="\t")

    # Post-hoc tests
    # Perform pairwise comparisons between groups
    # Bonferroni correction for multiple comparisons
    # Perform t-tests for each variable and statistic
    # Create table with group-comparisons and t-test results
    results_table = pd.DataFrame()
    for variable in variable_names[phases[1]]:
        for test_statistic in test_statistics:
            for group1 in data_all["group"].unique():
                for group2 in data_all["group"].unique():
                    if group1 != group2:
                        data_group1 = data_all[data_all["group"] == group1][test_statistic + "_" + variable]
                        data_group2 = data_all[data_all["group"] == group2][test_statistic + "_" + variable]
                        ttest_stats, ttest_p = stats.ttest_ind(data_group1, data_group2)
                        # Append results to results_table
                        results_table = results_table._append(
                            {
                                "Variable": f"{variable} {test_statistic}",
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
    n_comparisons = (
        len(data_all["group"].unique())
        * (len(data_all["group"].unique()) - 1)
        * len(variable_names[phases[1]])
        * len(test_statistics)
    )

    # Corrected alpha level
    alpha_corrected = alpha / n_comparisons

    # Mark significant results
    results_table["Significance"] = results_table["p-value"] < alpha_corrected

    # Switch from scientific notation to fixed notation
    pd.options.display.float_format = "{:.5f}".format

    # Round p-values to three decimal places
    results_table["p-value"] = results_table["p-value"].round(3)
    # Round statistics to two decimal places
    results_table[["t-statistic"]] = results_table[["t-statistic"]].round(2)

    # Change formatting of all group variables
    for group in results_table["Group1"].unique():
        results_table["Group1"] = results_table["Group1"].replace(group, group.replace("phase1", "Phase 1"))
        results_table["Group1"] = results_table["Group1"].replace(group, group.replace("phase3", "Phase 3"))
    for group in results_table["Group2"].unique():
        results_table["Group2"] = results_table["Group2"].replace(group, group.replace("phase1", "Phase 1"))
        results_table["Group2"] = results_table["Group2"].replace(group, group.replace("phase3", "Phase 3"))

    # Save the results of the post-hoc tests as a tsv file
    results_table.to_csv(Path(results_dir_comparison) / "posthoc_results.tsv", sep="\t")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    compare_variability_phase1_phase3()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
