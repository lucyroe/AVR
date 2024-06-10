"""
Plotting raincloud plot to compare the variability between AVR phase 1 and phase 2.

The following steps are performed:
1. Load data from the preprocessed annotations for each subject
2. Calculate the mean and standard deviation for each subject and each quadrant
3. Save the calculated statistics as a new file
4. Combine the data for all subjects and all phases (1 and 2) into one dataframe
5. Plot a raincloud plot for each variable (valence and arousal) and each statistic (mean and standard deviation)
    To mark the significant differences between the groups, the output statistics from compare_variability_phase1+2.py
    are used.
    This script needs to be run before significant differences can be marked.
6. Save the resulting four raincloud plots as png files

Author: Lucy Roellecke
Created on: 21 May 2024
Last updated: 10 June 2024
"""

# %% Import
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# path where data is saved
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"
# path where results should be saved
resultpath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/comparison_phase1_phase2/"
phases = ["phase1", "phase2"]  # phases for which the raincloud plot should be plotted
quadrants = ["HP", "HN", "LP", "LN"]  # quadrants/videos in phase 1 for which the raincloud plot should be plotted
# in phase 2, there was only one video

rating_method = ["Flubber"]  # rating method for which the raincloud plot should be plotted
variables = ["cr_v", "cr_a"]  # variables for which the raincloud plot should be plotted
#  "cr_dist", "cr_angle"
# in phase 1, different rating methods were used -> we only compare Flubber from phase 1 to Flubber from phase 2
variable_names = {"cr_v": "Valence", "cr_a": "Arousal"}

statistics = ["mean", "std_dev"]  # statistics for which the raincloud plot should be plotted
statistic_names = {"mean": "Mean", "std_dev": "Standard Deviation"}

# Colors
# Create a list of colors for the boxplots
boxplots_colors = ["#5fb0b4", "#5f85b4", "#645fb4", "#3f5f87", "#ff9900"]
# Create a list of colors for the violin plots
violin_colors = ["#5fb0b4", "#5f85b4", "#645fb4", "#3f5f87", "#ff9900"]
# Create a list of colors for the scatter plots
scatter_colors = ["#5fb0b4", "#5f85b4", "#645fb4", "#3f5f87", "#ff9900"]

mark_significant_differences = True  # if True, significant differences will be marked in the raincloud plot

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def plot_raincloud(data: pd.DataFrame, variable: str, statistic: str, significant_differences: list) -> plt.figure():
    """
    Plot a raincloud plot for the given data.

    Raincloud plots are made up of three different parts:
    - Boxplots.
    - Violin Plots.
    - Scatter Plots.

    Args:
    ----
    data: dataframe with the standard deviation for each participant.
    variable: variable for which the raincloud plot should be plotted.
    statistic: statistic for which the raincloud plot should be plotted (mean or std_dev).
    significant_differences: list of significant differences between the groups.

    Returns:
    -------
    None.
    """
    # Divide data for each part of the plot
    data_phase1_hn = data[(data["phase"] == "phase1") & (data["quadrant"] == "HN")][f"{statistic}_{variable}"]
    data_phase1_hp = data[(data["phase"] == "phase1") & (data["quadrant"] == "HP")][f"{statistic}_{variable}"]
    data_phase1_ln = data[(data["phase"] == "phase1") & (data["quadrant"] == "LN")][f"{statistic}_{variable}"]
    data_phase1_lp = data[(data["phase"] == "phase1") & (data["quadrant"] == "LP")][f"{statistic}_{variable}"]
    data_phase2 = data[data["phase"] == "phase2"][f"{statistic}_{variable}"]

    # Combine data for all phases in a list
    data_list = [data_phase1_hn, data_phase1_hp, data_phase1_ln, data_phase1_lp, data_phase2]

    figure, axes = plt.subplots(figsize=(10, 5))

    # Boxplots
    # Boxplot data
    boxplot_data = axes.boxplot(
        data_list, widths=0.15, patch_artist=True, showfliers=False, medianprops=dict(color="black")
    )

    # Change to the desired color and add transparency
    for patch, color in zip(boxplot_data["boxes"], boxplots_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Violin Plots
    # Violin plot data
    violin_data = axes.violinplot(
        data_list, points=max(data["subject"]), showmeans=False, showmedians=False, showextrema=False
    )

    for idx, body in enumerate(violin_data["bodies"]):
        # Get the center of the plot
        center = np.mean(body.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], center, np.inf)
        # Change to the desired color
        body.set_color(violin_colors[idx])

    # Scatter Plots
    for idx, features in enumerate(data_list):
        # Add jitter effect so the features do not overlap on the y-axis
        x = np.full(len(features), idx + 0.8)
        idxs = np.arange(len(x))
        out = x.astype(float)
        # Create a default_rng instance
        rng = np.random.default_rng()
        out.flat[idxs] += rng.uniform(low=-0.05, high=0.05, size=len(idxs))
        x = out
        plt.scatter(x, features, s=3, c=scatter_colors[idx])

    # Set labels
    plt.xticks([1, 2, 3, 4, 5], ["Phase 1 HN", "Phase 1 HP", "Phase 1 LN", "Phase 1 LP", "Phase 2"])
    plt.ylabel(f"{statistic_names[statistic]} of {variable_names[variable]}")
    plt.title(f"Comparison of the {statistic_names[statistic]} of {variable_names[variable]} for Phase 1 and Phase 2")

    if mark_significant_differences:
        # Mark significant differences with an asterisk and a line above the two groups
        counter = 0
        # Get distance between lines depending on statistic (mean or standard deviation)
        distance = 0.3 if statistic == "mean" else 0.1

        for difference in significant_differences:
            first_group = difference[0]
            second_group = difference[1]
            # Get x-tick labels and positions
            x_tick_labels = [tick.get_text() for tick in axes.get_xticklabels()]
            xtick_positions = axes.get_xticks()

            # Get position of the label for the first group
            label_index_first = x_tick_labels.index(first_group)
            specific_xtick_position_first_group = xtick_positions[label_index_first]
            # Get position of the label for the second group
            label_index_second = x_tick_labels.index(second_group)
            specific_xtick_position_second_group = xtick_positions[label_index_second]

            # Get maximum value of the two groups
            max_value = max(data_list[label_index_first].max(), data_list[label_index_second].max())

            # The color of line and asterisk should be black if the difference is between phase 1 and phase 2
            # else it should be grey
            color = "black" if first_group == "Phase 2" or second_group == "Phase 2" else "grey"

            # Plot a line between the two groups
            plt.plot(
                [specific_xtick_position_first_group, specific_xtick_position_second_group],
                [max_value + distance + counter, max_value + distance + counter],
                color=color,
            )

            # Add an asterisk in the middle of the line
            plt.text(
                (specific_xtick_position_first_group + specific_xtick_position_second_group) / 2,
                max_value + distance + counter,
                "*",
                fontsize=12,
                color=color,
            )

            counter += 0.1

    # Show plot
    plt.show()

    return figure


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    for phase in phases:
        # Get name of directory
        directory = Path(datapath) / phase / "preprocessed" / "annotations/"

        # %% LOAD DATA, CALCULATE STATISTICS, AND SAVE AS NEW FILE

        # List all files in directory
        files = os.listdir(directory)
        # Delete all files that are not csv files
        files = [file for file in files if file.endswith(".csv")]
        # Create subject list
        subjects = [file.split("_")[1] for file in files if file.endswith(".csv")]

        # Create empty dataframe to store statistics
        stats = pd.DataFrame()

        # Loop over all subjects
        for index, subject in enumerate(subjects):
            # Load data
            data = pd.read_csv(Path(directory) / files[index])

            # Loop over all variables
            for variable in variables:
                if phase == "phase1":
                    # Delete all rows from the dataframe where the rating method is not Flubber
                    Flubber_data = data[data["rating_method"] == "Flubber"]
                    # Loop over all quadrants
                    for quadrant in quadrants:
                        # Get only the data for the current quadrant
                        quadrant_data = Flubber_data[Flubber_data["quadrant"] == quadrant]

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
                            f"mean_{variable}": mean_subject,
                            f"std_dev_{variable}": stats_subject,
                            "n_samples": n_samples_subject,
                        }

                        # Append new row to dataframe
                        stats = stats._append(new_row, ignore_index=True)

                else:  # for phase 2
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
                        f"mean_{variable}": mean_subject,
                        f"std_dev_{variable}": stats_subject,
                        "n_samples": n_samples_subject,
                    }

                    # Append new row to dataframe
                    stats = stats._append(new_row, ignore_index=True)

        # Formatting
        if phase == "phase1":
            stats_formatted = pd.DataFrame()
            for subject in subjects:
                stats_subject = stats[stats["subject"] == subject]
                # Merge all variables into one row
                stats_subject_new = stats_subject[: len(quadrants)]
                stats_subject_new["mean_cr_a"] = stats_subject["mean_cr_a"][len(quadrants) :].to_numpy()
                stats_subject_new["std_dev_cr_a"] = stats_subject["std_dev_cr_a"][len(quadrants) :].to_numpy()

                # Append new row to dataframes
                stats_formatted = stats_formatted._append(stats_subject_new, ignore_index=True)

        else:
            stats_formatted = pd.DataFrame()
            for subject in subjects:
                stats_subject = stats[stats["subject"] == subject]
                # Merge all variables into one row
                stats_subject_new = stats_subject[:1]
                stats_subject_new["mean_cr_a"] = stats_subject["mean_cr_a"][1:].to_numpy()
                stats_subject_new["std_dev_cr_a"] = stats_subject["std_dev_cr_a"][1:].to_numpy()

                # Append new row to dataframes
                stats_formatted = stats_formatted._append(stats_subject_new, ignore_index=True)

        # Append a 0 for all subject numbers below 10
        stats_formatted["subject"] = stats_formatted["subject"].apply(lambda x: f"0{x}" if len(x) == 1 else x)
        # Sort the dataframe by subject
        stats_sorted = stats_formatted.sort_values(by="subject")

        # Save the dataframe as a new file
        stats_sorted.to_csv(Path(datapath) / phase / f"stats_{phase}.csv")

    # %% PLOT RAINCLOUD PLOTS AND SAVE
    # Create empty dataframe to combine all data for all phases
    combined_data = pd.DataFrame()
    # Loop over phases and read in dataframes
    for phase in phases:
        data = pd.read_csv(Path(datapath) / phase / f"stats_{phase}.csv")
        # Append data to combined_data
        combined_data = combined_data._append(data)

    # Read in the significant differences
    data_significant_differences = pd.read_csv(Path(resultpath) / "posthoc_results.tsv", sep="\t")

    # Filter for the significant differences
    significant_differences_dataframe = data_significant_differences[
        data_significant_differences["Significance"] == True  # noqa: E712
    ]

    # Create raincloud plot for each variable and each statistic
    for statistic in statistics:
        for variable in variables:
            if mark_significant_differences:
                # Get significant differences for the current variable and statistic
                significant_differences_current = significant_differences_dataframe[
                    significant_differences_dataframe["Variable"]
                    == f"{variable_names[variable]} {statistic_names[statistic]}"
                ]
                # Put the significant differences in a list (pairs of groups)
                significant_differences = [
                    [group1, group2]
                    for group1, group2 in zip(
                        significant_differences_current["Group1"], significant_differences_current["Group2"],
                        strict=True
                    )
                ]
                # Delete duplicates (no matter in which order the groups are)
                significant_differences = [sorted(difference) for difference in significant_differences]
                significant_differences = list({tuple(difference) for difference in significant_differences})
                # Sort alphabetically
                significant_differences = sorted(significant_differences)
            else:
                significant_differences = []

            # Plot raincloud plot
            figure = plot_raincloud(combined_data, variable, statistic, significant_differences)
            # Save figure
            figure.savefig(
                Path(resultpath) / f"raincloud_phase1+2_{statistic_names[statistic]}_{variable_names[variable]}.png"
            )

# %%
