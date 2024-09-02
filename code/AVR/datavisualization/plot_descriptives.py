"""
Plotting descriptive statistics for the AVR data.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 12 August 2024
Last updated: 2 September 2024
"""

def plot_descriptives(  # noqa: C901, PLR0915, PLR0912
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    show_plots=True,
):
    """
    Plot descriptive statistics for the AVR data.

    The following steps are performed:
        1. Load data.
        2. Plot average timeseries.
        3. Plot raincloud plots.
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # Path where the data is stored
    data_dir = Path(data_dir) / "phase3" / "AVR" / "derivatives"
    annotation_dir = Path(data_dir) / "features" / "avg" / "beh"
    physiological_dir = Path(data_dir) / "features" / "avg" / "eeg"
    events_dir = Path(data_dir) / "preproc"

    # Path where results should be saved
    results_dir_descriptives = Path(results_dir) / "phase3" / "AVR" / "avg"

    # List of datastreams
    datastreams = {
        "annotation": ["valence", "arousal"],
        "physiological": [
            "ibi",
            "hrv",
            "lf_hrv",
            "hf_hrv",
            "posterior_alpha",
            "frontal_alpha",
            "frontal_theta",
            "gamma",
            "beta",
        ],
    }

    # List of videos
    videos = ["spaceship", "invasion", "asteroids", "underwood"]

    # Colors
    colors = {
        "annotation": {"valence": "#0072B2", "arousal": "#0096c6"},  # shades of dark blue
        "physiological": {
            "ibi": "#CC79A7",
            "hrv": "#ff9789",
            "lf_hrv": "#f3849b",
            "hf_hrv": "#ffb375",
            # shades of pink-orange
            "posterior_alpha": "#009E73",
            "frontal_alpha": "#66b974",
            "frontal_theta": "#a5d279",
            "gamma": "#00c787",
            "beta": "#85b082",
        },  # shades of green
    }

    colors_videos = {
        "spaceship": "#6C6C6C",  # grey
        "invasion": "#56B4E9",  # shades of light blue
        "asteroids": "#649ee1",
        "underwood": "#7a86d1",
    }

    mark_significant_differences = True  # if True, significant differences will be marked in the boxplots

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    def plot_average_timeseries(data, variable_name, colors, colors_videos, ax):
        fig = sns.lineplot(
            data=data, x="timestamp", y=variable_name, hue="variable", palette=colors, ax=ax, legend="brief", lw=1.5
        )

        # Get the maximum y-value of the plot for the positioning of the video labels
        if len(data["variable"].unique()) == 1:
            mean = data.groupby("timestamp")[variable_name].mean()
            sem = data.groupby("timestamp")[variable_name].sem()
            ci_range = sem * stats.t.ppf((1 + 0.95) / 2.0, len(data.groupby("timestamp")[variable_name]) - 1)
            upper_ci = mean + ci_range
            max_y = upper_ci.max()
        else:
            max_all_variables = []
            for variable in data["variable"].unique():
                mean = data[data["variable"] == variable].groupby("timestamp")[variable_name].mean()
                sem = data[data["variable"] == variable].groupby("timestamp")[variable_name].sem()
                ci_range = sem * stats.t.ppf(
                    (1 + 0.95) / 2.0, len(data[data["variable"] == variable].groupby("timestamp")[variable_name]) - 1
                )
                upper_ci = mean + ci_range
                max_variable = upper_ci.max()
                max_all_variables.append(max_variable)
            max_y = max(max_all_variables)

        # Add shading for the different videos
        for video in videos:
            if video == "spaceship":
                timestamps_start_video = timestamps_events[event_names == f"start_{video}"]
                timestamps_stop_video = timestamps_events[event_names == f"end_{video}"]
                for timestamp_start_video, timestamp_stop_video in zip(
                    timestamps_start_video, timestamps_stop_video, strict=True
                ):
                    ax.axvspan(timestamp_start_video, timestamp_stop_video, color=colors_videos[video], alpha=0.1)
            else:
                # Get the timestamps of the start and end of the video
                timestamp_start_video = float(timestamps_events[event_names == f"start_{video}"])
                timestamp_stop_video = float(timestamps_events[event_names == f"end_{video}"])
                # Add a text label for the video
                ax.text(
                    timestamp_start_video + (timestamp_stop_video - timestamp_start_video) / 3,
                    max_y,
                    video.capitalize(),
                    color="white",
                    fontsize=14,
                    fontweight="bold",
                    bbox=dict(facecolor=colors_videos[video], alpha=0.8),
                )

        fig.set_ylabel(variable_name.capitalize())

        # Transform the x-axis to minutes and set the label
        fig.set_xlabel("Time (min)")
        xticks = fig.get_xticks()
        fig.set_xticklabels([f"{int(x/60)}" for x in xticks])

        # Set the legend
        fig.legend(title="", loc="upper right")
        # Capitalize the legend labels
        for t in fig.legend_.texts:
            if variable_name == "rating":
                t.set_text(t.get_text().capitalize())
            elif variable_name == "value":
                t.set_text(t.get_text().upper())
            else:
                t.set_text(t.get_text().replace("_", " ").title())

        # Get all variables
        variables = data["variable"].unique()
        variables_text = " & ".join(
            (" ".join(variable.title().split("_")) if variable_name != "value" else variable.upper())
            for variable in variables
        )
        # Set the title
        fig.set_title(f"{variables_text}", fontsize=16, fontweight="bold", pad=20)

    def plot_raincloudplot(  # noqa: PLR0913
        data: pd.DataFrame,
        variable: str,
        videos: list[str],
        y_label: str,
        significant_differences: list,
        axes: plt.Axes,
    ) -> plt.figure():
        """
        Plot a raincloud plot for the given data.

        Raincloud plots are made up of three different parts:
        - Boxplots.
        - Violin Plots.
        - Scatter Plots.

        Args:
        ----
        data: dataframe with the data to be plotted.
        variable: variable for which the raincloud plot should be plotted.
        videos: list of videos for which the data should be plotted.
        y_label: label for the y-axis.
        significant_differences: list of significant differences between the groups.
        axes: axes on which the plot should be plotted.

        Returns:
        -------
        None.
        """
        # Divide data for each part of the plot
        data_list = []
        for video in videos:
            data_video = data[data["video"] == video][f"{variable}"].groupby(data["subject"]).mean()
            data_list.append(data_video)

        # Boxplots
        # Boxplot data
        boxplot_data = axes.boxplot(
            data_list, widths=0.15, patch_artist=True, showfliers=False, medianprops=dict(color="black")
        )

        # Change to the desired color and add transparency
        for patch, color in zip(boxplot_data["boxes"], colors_videos.values(), strict=False):
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
            body.set_color(colors_videos[videos[idx]])

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
            axes.scatter(x, features, s=3, c=colors_videos[videos[idx]])

        # Set labels
        axes.set_xticklabels([f"{video.capitalize()}" for video in videos])
        axes.set_ylabel(y_label)

        # Set title
        title = (
            variable.upper()
            if variable in ("ibi", "hrv", "lf_hrv", "hf_hrv")
            else variable.replace("_", " ").title()
        )

        axes.set_title(f"{title}", fontsize=14, fontweight="bold", pad=20)

        if mark_significant_differences:
            # Mark significant differences with an asterisk and a line above the two groups
            counter = 0
            # Get distance between lines
            distance = 0.5 * max([data_list[0].max(), data_list[1].max(), data_list[2].max(), data_list[3].max()])

            for difference in significant_differences:
                first_group = difference[0].capitalize()
                second_group = difference[1].capitalize()
                # Get x-tick labels and positions
                x_tick_labels = [tick.get_text() for tick in axes.get_xticklabels()]
                xtick_positions = axes.get_xticks()

                # Get position of the label for the first group
                label_index_first = x_tick_labels.index(first_group)
                specific_xtick_position_first_group = xtick_positions[label_index_first]
                # Get position of the label for the second group
                label_index_second = x_tick_labels.index(second_group)
                specific_xtick_position_second_group = xtick_positions[label_index_second]

                # Get maximum value of the whole data
                max_value = max(data_list[label_index_first].max(), data_list[label_index_second].max())

                # The color of line and asterisk should be black
                color = "black"

                # Plot a line between the two groups
                axes.plot(
                    [specific_xtick_position_first_group, specific_xtick_position_second_group],
                    [max_value + distance + counter, max_value + distance + counter],
                    color=color,
                )

                # Add an asterisk in the middle of the line
                axes.text(
                    (specific_xtick_position_first_group + specific_xtick_position_second_group) / 2,
                    max_value + distance + counter,
                    "*",
                    fontsize=12,
                    color=color,
                )

                counter += 0.3 * max([data_list[0].max(), data_list[1].max(), data_list[2].max(), data_list[3].max()])

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% STEP 1. LOAD DATA
    # Create a dictionary for all datastreams
    data = {f"{variable}": [] for variable in datastreams["annotation"] + datastreams["physiological"]}
    # Load the annotation data
    annotations = pd.read_csv(annotation_dir / "all_subjects_task-AVR_beh_features.tsv", sep="\t")
    # Drop unnecessary columns
    annotations = annotations.drop(columns=["subject", "video"])

    for variable in datastreams["annotation"]:
        # Create a dataframe with the variable
        variable_data = pd.DataFrame(annotations[variable])
        # Add the timestamp as first column
        variable_data.insert(0, "timestamp", annotations["timestamp"])
        data[variable] = variable_data

    # Load the physiological data
    physiological = pd.read_csv(physiological_dir / "all_subjects_task-AVR_physio_features.tsv", sep="\t")
    # Drop unnecessary columns
    physiological = physiological.drop(columns=["subject", "video"])

    for variable in datastreams["physiological"]:
        # Create a dataframe with the variable
        variable_data = pd.DataFrame(physiological[variable])
        # Add the timestamp as first column
        variable_data.insert(0, "timestamp", physiological["timestamp"])
        data[variable] = variable_data

    # Get events
    events = pd.read_csv(events_dir / "events_experiment.tsv", sep="\t")
    timestamps_events = events["onset"]
    event_names = events["event_name"]

    # Create a list of the different phases that has the same length as the data
    for variable in data:
        variable_data = data[variable]
        video_column = pd.DataFrame(index=variable_data.index, columns=["video"])
        for video in videos:
            counter = 0
            for timestamp in variable_data["timestamp"].unique():
                # Get the timestamps of the start and end of the video
                timestamp_start_video = timestamps_events[event_names == f"start_{video}"].reset_index()["onset"][
                    counter
                ]
                timestamp_stop_video = timestamps_events[event_names == f"end_{video}"].reset_index()["onset"][counter]
                # Get the indices of the timestamp
                timestamp_indices = variable_data.index[variable_data["timestamp"] == timestamp]
                if timestamp >= timestamp_start_video and timestamp <= timestamp_stop_video:
                    for index in timestamp_indices:
                        video_column.loc[index, "video"] = video
                if video == "spaceship" and timestamp >= timestamp_stop_video:
                    counter += 1
        variable_data["video"] = video_column
        # Add the data to the dictionary
        data[variable] = variable_data

    # %% STEP 2. PLOT AVERAGE TIMESERIES
    # Create a figure with subplots for annotation ratings, for ecg features, and for eeg features
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(8, 1)
    sns.set(style="ticks")

    # Plot annotation ratings
    annotation_axis = fig.add_subplot(gs[0])
    annotation_data = pd.DataFrame()
    for datastream in datastreams["annotation"]:
        stream_data = data[f"{datastream}"]
        annotation_data = pd.concat([annotation_data, stream_data], axis=1)

    # Drop duplicate columns
    annotation_data = annotation_data.loc[:, ~annotation_data.columns.duplicated()]

    # Find NaN values and drop them
    annotation_data = annotation_data.dropna()

    # Format data for plotting
    annotation_data = pd.melt(
        annotation_data, id_vars=["timestamp", "video"], var_name="variable", value_name="rating"
    )
    plot_average_timeseries(annotation_data, "rating", colors["annotation"], colors_videos, annotation_axis)

    # Plot physiological features
    # ECG features
    ecg_features = pd.DataFrame()
    for datastream in datastreams["physiological"][0:4]:
        stream_data = data[f"{datastream}"]
        ecg_features = pd.concat([ecg_features, stream_data], axis=1)

    # Drop duplicate columns
    ecg_data = ecg_features.loc[:, ~ecg_features.columns.duplicated()]

    # Find NaN values and drop them
    ecg_data = ecg_data.dropna()

    colors_ecg = [colors["physiological"][ecg_feature] for ecg_feature in datastreams["physiological"][0:4]]

    # Format data for plotting
    ecg_data = pd.melt(ecg_data, id_vars=["timestamp", "video"], var_name="variable", value_name="value")
    # Plot IBI in one plot
    ibi_axis = fig.add_subplot(gs[1])
    ibi_data = ecg_data[ecg_data["variable"] == "ibi"]
    plot_average_timeseries(ibi_data, "value", [colors["physiological"]["ibi"]], colors_videos, ibi_axis)
    # Other ECG features in one plot
    ecg_axis = fig.add_subplot(gs[2])
    plot_average_timeseries(ecg_data[ecg_data["variable"] != "ibi"], "value", colors_ecg[1:], colors_videos, ecg_axis)

    # EEG features
    # Create one plot for each EEG feature
    eeg_features = pd.DataFrame()
    for datastream in datastreams["physiological"][4:]:
        stream_data = data[f"{datastream}"]
        eeg_features = pd.concat([eeg_features, stream_data], axis=1)

    # Drop duplicate columns
    eeg_features = eeg_features.loc[:, ~eeg_features.columns.duplicated()]

    # Find NaN values and drop them
    eeg_features = eeg_features.dropna()

    eeg_feature_names = datastreams["physiological"][4:]

    index = 3
    for eeg_feature in eeg_feature_names:
        colors_eeg = [colors["physiological"][eeg_feature]]
        eeg_data = data[eeg_feature]
        # Format data for plotting
        eeg_data = pd.melt(eeg_data, id_vars=["timestamp", "video"], var_name="variable", value_name="power")
        eeg_axis = fig.add_subplot(gs[index])
        plot_average_timeseries(eeg_data, "power", colors_eeg, colors_videos, eeg_axis)
        index += 1

    # Remove the x-axis label for all subplots except the last one
    for ax in fig.get_axes():
        if ax != eeg_axis:
            ax.set_xlabel("")

    # Increase space between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(results_dir_descriptives / "average_timeseries.svg", dpi=300)

    # Show the plot
    if show_plots:
        plt.show()

    # %% STEP 3. PLOT RAINCLOUD PLOTS FOR ANNOTATION RATINGS
    # Read in annotation data
    annotation_data = pd.read_csv(
        data_dir / "features" / "avg" / "beh" / "all_subjects_task-AVR_beh_features.tsv", sep="\t"
    )
    # Read in results of the post-hoc-tests
    posthoc_results = pd.read_csv(results_dir_descriptives / "stats" / "annotation_posthoc_results.tsv", sep="\t")
    # Get the significant differences
    significant_differences_dataframe = posthoc_results[posthoc_results["Significance"] == True]  # noqa: E712

    # Plot raincloud plots to visualize the differences in annotation ratings between the different videos
    # Create a figure with subplots for valence and arousal
    figure, axes = plt.subplots(2, 1, figsize=(8, 8))

    for variable in datastreams["annotation"]:
        if mark_significant_differences:
            # Get significant differences for the current variable
            significant_differences_current = significant_differences_dataframe[
                significant_differences_dataframe["Variable"] == variable
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

            # Sort so that the groups are in alphabetical order
            significant_differences = sorted(significant_differences)
        else:
            significant_differences = []

        # Plot raincloud plot
        plot_raincloudplot(
            annotation_data,
            variable,
            videos,
            "Rating",
            significant_differences,
            axes[datastreams["annotation"].index(variable)],
        )

    # More space between the two plots
    plt.tight_layout()

    # Save figure
    figure.savefig(Path(results_dir_descriptives) / "beh" / "raincloud_mean_annotation.svg")

    # Show plot
    if show_plots:
        plt.show()

    # %% STEP 4. PLOT RAINCLOUD PLOTS FOR PHYSIOLOGICAL FEATURES
    # Read in physiological data
    physiological_data = pd.read_csv(
        data_dir / "features" / "avg" / "eeg" / "all_subjects_task-AVR_physio_features.tsv", sep="\t"
    )

    # Read in results of the post-hoc-tests
    posthoc_results = pd.read_csv(results_dir_descriptives / "stats" / "physiological_posthoc_results.tsv", sep="\t")

    # Plot raincloud plots to visualize the differences in physiological features between the different videos
    # Create a figure with subplots for the different physiological features
    figure, axes = plt.subplots(9, 1, figsize=(8, 40))

    for variable in datastreams["physiological"]:
        if mark_significant_differences:
            # Get significant differences for the current variable
            significant_differences_current = posthoc_results[posthoc_results["Variable"] == variable][
                posthoc_results["Significance"] == True  # noqa: E712
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

            # Sort so that the groups are in alphabetical order
            significant_differences = sorted(significant_differences)
        else:
            significant_differences = []

        ylabel = "Value" if variable in ("ibi", "hrv", "lf_hrv", "hf_hrv") else "Power"

        # Plot raincloud plot
        plot_raincloudplot(
            physiological_data,
            variable,
            videos,
            ylabel,
            significant_differences,
            axes[datastreams["physiological"].index(variable)],
        )

    # More space between the two plots
    plt.tight_layout()

    # Save figure
    figure.savefig(Path(results_dir_descriptives) / "eeg" / "raincloud_mean_physiological.svg")

    # Show plot
    if show_plots:
        plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    plot_descriptives()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
