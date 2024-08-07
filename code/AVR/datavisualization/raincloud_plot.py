"""
Plotting raincloud plot to compare the variability between AVR phase 1 and phase 2 or 3.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 21 May 2024
Last updated: 7 August 2024
"""


def raincloud_plot(  # noqa: C901, PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/phase3/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/phase3/",
    show_plots=False,
):
    """
    Plot raincloud plots to compare the variability between AVR phase 1 and phase 2 or 3.

    The following steps are performed:
        1. Load statistics for phase 1 and phase 2 or 3 (from compare_variability_phase1+3.py)
        2. Combine the data for all subjects and all phases (1 and 2 or 3) into one dataframe
        3. Plot a raincloud plot for each variable (valence and arousal) and each statistic
            (mean and standard deviation)
        4. Save the resulting four raincloud plots as png files
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    phases = ["phase1", "phase3"]  # Phases for which the raincloud plot should be plotted

    # Path where results should be saved
    results_dir_comparison = Path(results_dir) / f"comparison_{phases[0]}_{phases[1]}"

    variable_names = ["valence", "arousal"]
    statistics = ["mean", "std_dev"]  # statistics for which the raincloud plot should be plotted

    # Colors
    colors = {
        "valence": ["#56B4E9", "#0072B2", "#6C6C6C", "#CC79A7", "#009E73"],  # light blue, dark blue, grey, pink, green
        "arousal": [
            "#F0E442",
            "#E69F00",
            "#D55E00",
            "#CC79A7",
            "#009E73",
        ],  # yellow, light orange, dark orange, pink, green
    }

    mark_significant_differences = True  # if True, significant differences will be marked in the raincloud plot

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    def plot_raincloud(
        data: pd.DataFrame, variable: str, statistic: str, significant_differences: list
    ) -> plt.figure():
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
        data_phase1_hn = data[data["group"] == "phase1 HN"][f"{statistic}_{variable}"]
        data_phase1_hp = data[data["group"] == "phase1 HP"][f"{statistic}_{variable}"]
        data_phase1_ln = data[data["group"] == "phase1 LN"][f"{statistic}_{variable}"]
        data_phase1_lp = data[data["group"] == "phase1 LP"][f"{statistic}_{variable}"]
        data_other_phase = data[data["group"] == phases[1]][f"{statistic}_{variable}"]

        # Combine data for all phases in a list
        data_list = [data_phase1_hn, data_phase1_hp, data_phase1_ln, data_phase1_lp, data_other_phase]

        figure, axes = plt.subplots(figsize=(10, 5))

        # Boxplots
        # Boxplot data
        boxplot_data = axes.boxplot(
            data_list, widths=0.15, patch_artist=True, showfliers=False, medianprops=dict(color="black")
        )

        # Change to the desired color and add transparency
        for patch, color in zip(boxplot_data["boxes"], colors[variable], strict=False):
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
            body.set_color(colors[variable][idx])

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
            plt.scatter(x, features, s=3, c=colors[variable][idx])

        # Set labels
        plt.xticks([1, 2, 3, 4, 5], ["Phase 1 HN", "Phase 1 HP", "Phase 1 LN", "Phase 1 LP", "Phase 3"])
        plt.ylabel(f"{statistic} of {variable}")
        plt.title(f"Comparison of the {statistic} of {variable} for Phase 1 and Phase 3")

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

                # The color of line and asterisk should be black if the difference is between phase 1 and phase 3
                # else it should be grey
                color = "black" if first_group == "Phase 3" or second_group == "Phase 3" else "grey"

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
        if show_plots:
            plt.show()

        return figure

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% LOAD STATISTICS FOR PHASE 1 AND PHASE 2 OR 3
    # Create empty dataframe to combine all data for all phases
    combined_data = pd.DataFrame()
    # Loop over phases and read in dataframes
    for phase in phases:
        file_directory = (Path(data_dir) / phase / "preprocessed" / "annotations" if phase == "phase1"
            else Path(data_dir) / phase / "AVR" / "derivatives" / "preproc")
        data = pd.read_csv(file_directory / f"stats_{phase}.tsv", sep="\t")
        # Append data to combined_data
        combined_data = combined_data._append(data)

    # Read in the significant differences
    data_significant_differences = pd.read_csv(Path(results_dir_comparison) / "posthoc_results.tsv", sep="\t")

    # Filter for the significant differences
    significant_differences_dataframe = data_significant_differences[
        data_significant_differences["Significance"] == True  # noqa: E712
    ]

    # %% PLOT RAINCLOUD PLOTS AND SAVE
    # Create raincloud plot for each variable and each statistic
    for statistic in statistics:
        for variable in variable_names:
            if mark_significant_differences:
                # Get significant differences for the current variable and statistic
                significant_differences_current = significant_differences_dataframe[
                    significant_differences_dataframe["Variable"] == f"{variable} {statistic}"
                ]
                # Put the significant differences in a list (pairs of groups)
                significant_differences = [
                    [group1, group2]
                    for group1, group2 in zip(
                        significant_differences_current["Group1"],
                        significant_differences_current["Group2"],
                        strict=True,
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
                Path(results_dir_comparison)
                / f"raincloud_{phases[0]}_{phases[1]}_{statistic}_{variable}.png"
            )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    raincloud_plot()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
