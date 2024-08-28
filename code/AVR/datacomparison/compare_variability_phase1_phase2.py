"""
Performing statistical tests (MANOVA) to compare the variability between AVR phase 1 and phase 2.

The following steps are performed:
1. Load data from summary statistics calculated in raincloud_plot.py (run raincloud_plot.py before running this script)
2. Test assumptions of MANOVA
    2.1 Normal distribution of data within groups (Shapiro-Wilk test)
    2.2 Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
    2.3 Linearity
3. Perform statistical tests
    3.1 Perform MANOVA
    3.2 Post-hoc tests
        3.2.1 Perform pairwise comparisons between groups
        3.2.2 Bonferroni correction for multiple comparisons

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 10 June 2024
Last updated: 10 June 2024
"""

# %% Import
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# path where data is saved
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"
# path where results should be saved
resultpath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/comparison_phase1_phase2/"
phases = ["phase1", "phase2"]  # phases for which the differences should be compared
quadrants = ["HP", "HN", "LP", "LN"]  # quadrants/videos in phase 1 for which the differences should be compared
# in phase 2, there was only one video

rating_method = ["Flubber"]  # rating method for which the differences should be compared
variables = ["cr_v", "cr_a"]  # variables for which the differences should be compared
#  "cr_dist", "cr_angle"
# in phase 1, different rating methods were used -> we only compare Flubber from phase 1 to Flubber from phase 2
variable_names = {"cr_v": "Valence", "cr_a": "Arousal"}

statistics = ["mean", "std_dev"]  # statistics for which the differences should be compared
statistic_names = {"mean": "Mean", "std_dev": "Standard Deviation"}

# Significance level
alpha = 0.05

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Create empty dataframe to store data
    data_all = pd.DataFrame()
    # %% LOAD DATA
    # Loop over all phases
    for phase in phases:
        # Get name of file with statistics calculated in raincloud_plot.py to load
        filename = f"stats_{phase}.csv"

        # Read in data
        data_phase = pd.read_csv(Path(datapath) / phase / filename)

        # Drop column with index
        data_phase = data_phase.drop(columns="Unnamed: 0")

        # Combine phase and quadrant column to create a new column with the group variable (for phase 1)
        if phase == "phase1":
            data_phase["group"] = data_phase["phase"] + " " + data_phase["quadrant"]
            # Drop quadrant and phase column
            data_phase = data_phase.drop(columns=["quadrant", "phase"])
        else:
            data_phase["group"] = data_phase["phase"]
            # Drop phase column
            data_phase = data_phase.drop(columns="phase")

        # Add data to data_all
        data_all = data_all._append(data_phase)

    # %% TEST ASSUMPTIONS
    # Test assumptions of MANOVA

    # 1. Normal distribution of data within groups (Shapiro-Wilk test)
    # Loop over groups
    # Create a matrix of plots for each group and variable
    # Determine the number of rows and columns for the subplot grid
    num_rows = len(data_all["group"].unique())
    num_cols = len(variables) * len(statistics)

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
        for variable in variables:
            for statistic in statistics:
                # Perform Shapiro-Wilk test
                shapiro_test = stats.shapiro(data_group[statistic + "_" + variable])

                # Plot histogram on the current axis
                data_group[statistic + "_" + variable].plot.hist(ax=axes[ax_counter])

                # Add labels to the plot
                if ax_counter % num_cols == 0:
                    axes[ax_counter].set_ylabel(f"{group}")
                else:  # no title
                    axes[ax_counter].set_ylabel("")
                first_subplot_last_row = num_rows * num_cols - num_cols
                if ax_counter >= first_subplot_last_row:
                    axes[ax_counter].set_xlabel(f"{variable_names[variable]} {statistic_names[statistic]}")

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
                        "Variable": f"{statistic_names[statistic]} {variable_names[variable]}",
                        "Statistic": shapiro_test[0],
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
    table_normality.to_csv(Path(resultpath) / "normality_test.tsv", sep="\t")

    # Show the plot
    #plt.show()  # noqa: ERA001

    # Save the plot
    fig.savefig(Path(resultpath) / "histograms_normality.png")

    # 2. Homogeneity of variance-covariance matrices (Barlett's and Levene's tests)
    # Create a table with the results of Barlett's and Levene's tests
    table_homogeinity = pd.DataFrame()
    for variable in variables:
        for statistic in statistics:
            # subset data
            data_phase1_hp = data_all[(data_all["group"] == "phase1 HP")][statistic + "_" + variable]
            data_phase1_hn = data_all[(data_all["group"] == "phase1 HN")][statistic + "_" + variable]
            data_phase1_lp = data_all[(data_all["group"] == "phase1 LP")][statistic + "_" + variable]
            data_phase1_ln = data_all[(data_all["group"] == "phase1 LN")][statistic + "_" + variable]
            data_phase2 = data_all[(data_all["group"] == "phase2")][statistic + "_" + variable]

            # Perform Barlett's test
            bartlett_stats, bartlett_p = stats.bartlett(
                data_phase1_hp, data_phase1_hn, data_phase1_lp, data_phase1_ln, data_phase2
            )

            # Perform Levene's test
            levene_stats, levene_p = stats.levene(
                data_phase1_hp, data_phase1_hn, data_phase1_lp, data_phase1_ln, data_phase2
            )

            # Append results to table_homogeinity
            table_homogeinity = table_homogeinity._append(
                {
                    "Variable": f"{statistic_names[statistic]} {variable_names[variable]}",
                    "Test": "Barlett",
                    "Statistic": bartlett_stats,
                    "p-value": bartlett_p,
                },
                ignore_index=True,
            )
            table_homogeinity = table_homogeinity._append(
                {
                    "Variable": f"{statistic_names[statistic]} {variable_names[variable]}",
                    "Test": "Levene",
                    "Statistic": levene_stats,
                    "p-value": levene_p,
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
    table_homogeinity.to_csv(Path(resultpath) / "homogeneity_test.tsv", sep="\t")

    # 3. Linearity
    # Create scatterplot matrix for each group with all variables
    for group in data_all["group"].unique():
        data_group = data_all[data_all["group"] == group]
        # Drop columns
        data_scatter_group = data_group.drop(columns=["group", "subject", "n_samples"])
        figure, axis = plt.subplots(figsize=(10, 10))
        pd.plotting.scatter_matrix(data_scatter_group, figsize=(10, 10), ax=axis, diagonal="kde")
        figure.suptitle(f"Scatterplot matrix for {group}")

        # Show the plot
        # plt.show()  # noqa: ERA001

        # Save the plot
        figure.savefig(Path(resultpath) / f"scatterplot_matrix_{group}.png")

    # %% PERFORM STATISTICAL TESTS
    # Perform MANOVA
    # Group variables: phase 1 HP, phase 1 HN, phase 1 LP, phase 1 LN, phase 2
    # Variables: mean_cr_v, std_dev_cr_v, mean_cr_a, std_dev_cr_a

    maov = MANOVA.from_formula("mean_cr_v + std_dev_cr_v + mean_cr_a + std_dev_cr_a ~ group", data=data_all)

    # Transform the results of the MANOVA into a table
    results_manova = pd.DataFrame(
        {
            "Value": maov.mv_test().results["group"]["stat"]["Value"],
            "Num_DF": maov.mv_test().results["group"]["stat"]["Num DF"],
            "Den_DF": maov.mv_test().results["group"]["stat"]["Den DF"],
            "F-value": maov.mv_test().results["group"]["stat"]["F Value"],
            "p-value": maov.mv_test().results["group"]["stat"]["Pr > F"],
            "Significance": maov.mv_test().results["group"]["stat"]["Pr > F"] < alpha
        }
    )

    # Switch from scientific notation to fixed notation
    pd.options.display.float_format = "{:.5f}".format

    for statistic in results_manova.index:
        # Round p-values to three decimal places
        results_manova.loc[statistic, "p-value"] = round(results_manova.loc[statistic, "p-value"], 3)
        # Round statistics to two decimal places
        results_manova.loc[statistic, "Value"] = round(results_manova.loc[statistic, "Value"], 2)
        results_manova.loc[statistic, "F-value"] = round(results_manova.loc[statistic, "F-value"], 2)
    # Round degrees of freedom to zero decimal places
    results_manova[["Num_DF", "Den_DF"]] = results_manova[["Num_DF", "Den_DF"]].astype(int)

    # Save the results of the MANOVA as a tsv file
    results_manova.to_csv(Path(resultpath) / "manova_results.tsv", sep="\t")

    # Post-hoc tests
    # Perform pairwise comparisons between groups
    # Bonferroni correction for multiple comparisons
    # Perform t-tests for each variable and statistic
    # Create table with group-comparisons and t-test results
    results_table = pd.DataFrame()
    for variable in variables:
        for statistic in statistics:
            for group1 in data_all["group"].unique():
                for group2 in data_all["group"].unique():
                    if group1 != group2:
                        data_group1 = data_all[data_all["group"] == group1][statistic + "_" + variable]
                        data_group2 = data_all[data_all["group"] == group2][statistic + "_" + variable]
                        ttest_stats, ttest_p = stats.ttest_ind(data_group1, data_group2)
                        # Append results to results_table
                        results_table = results_table._append(
                            {
                                "Variable": f"{variable_names[variable]} {statistic_names[statistic]}",
                                "Group1": group1,
                                "Group2": group2,
                                "t-statistic": ttest_stats,
                                "p-value": ttest_p,
                            },
                            ignore_index=True,
                        )

    # Bonferroni correction for multiple comparisons
    # Number of comparisons
    n_comparisons = (
        len(data_all["group"].unique()) * (len(data_all["group"].unique()) - 1) * len(variables) * len(statistics)
    )

    # Corrected alpha level
    alpha_corrected = alpha / n_comparisons

    # Mark significant results
    results_table["Significance"] = results_table["p-value"] < alpha_corrected

    # Count number of significant results
    n_significant = results_table["Significance"].sum()

    # Switch from scientific notation to fixed notation
    pd.options.display.float_format = "{:.5f}".format

    # Round p-values to three decimal places
    results_table["p-value"] = results_table["p-value"].round(3)
    # Round statistics to two decimal places
    results_table[["t-statistic"]] = results_table[["t-statistic"]].round(2)

    # Change formatting of all group variables
    for group in results_table["Group1"].unique():
        results_table["Group1"] = results_table["Group1"].replace(group, group.replace("phase1", "Phase 1"))
        results_table["Group1"] = results_table["Group1"].replace(group, group.replace("phase2", "Phase 2"))
    for group in results_table["Group2"].unique():
        results_table["Group2"] = results_table["Group2"].replace(group, group.replace("phase1", "Phase 1"))
        results_table["Group2"] = results_table["Group2"].replace(group, group.replace("phase2", "Phase 2"))

    # Save the results of the post-hoc tests as a tsv file
    results_table.to_csv(Path(resultpath) / "posthoc_results.tsv", sep="\t")

# %%
