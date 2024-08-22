"""
Plotting hidden states from HMM analysis of the AVR data.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 20 August 2024
Last updated: 22 August 2024
"""

# %%
def plot_hidden_states(  # noqa: C901, PLR0915, PLR0912
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    show_plots=True,
):
    """
    Plot descriptive statistics for the AVR data.

    The following steps are performed:
        1. Load data.
        2. Plot example of HMM algorithm for methods.
        3. Create one set of plots for each HMM:
            3a. Violin plots for all four states for fractional occupancy, life times, interval times.
            3b. Mean valence and arousal ratings for each state on the Affect Grid.
            3c. Raincloud plots for each state for all ECG features with significant differences marked.
            3d. Topoplots for each state for all EEG features.
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # Path where the data is stored
    resultpath = Path(results_dir) / "phase3" / "AVR"
    events_dir = Path(data_dir) / "phase3" / "AVR" / "derivatives" / "preproc"

    # Which HMMs to plot
    models = ["cardiac"]
    #"neural", "integrated"]
    # Which features are used for which HMM
    models_features = {
        "cardiac": ["ibi", "hf-hrv"],
        "neural": ["posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
        "integrated": ["ibi", "hf-hrv", "posterior_alpha", "frontal_alpha", "frontal_theta", "beta", "gamma"],
    }

    # Which subject to use as example subject for HMM for methods
    example_subject = "031"
    time_window = [14.5, 16.5]    # Time window for the example plot in minutes

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

    # Colors of the hidden states
    colors_states = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]  #  dark orange, dark blue, green, pink

    mark_significant_differences = True  # if True, significant differences will be marked in the boxplots

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    def example_plot(  # noqa: PLR0913
        data, hidden_states, axis, title, legend
    ):
        """
        Plot the data with the hidden states.

        Args:
        ----
            data (np.array): The data to plot.
            hidden_states (np.array): The hidden states of the data.
            axis (matplotlib.axis): The axis to plot on.
            title (str): The title of the plot.
            legend (list): The legend of the plot.
        """
        # Define the intervals with the hidden states to be of different colors
        segments = []
        colors = []
        color_labels = []

        all_states = np.unique(hidden_states)
        all_states.sort()

        times = np.arange(len(data))

        for i in range(len(data) - 1):
            xi = times[i:i+2]
            yi = data[i:i+2]

            if hidden_states[xi[1]] == all_states[0]:
                colors.append(colors_states[0])
                color_labels.append(legend[0])
            elif hidden_states[xi[1]] == all_states[1]:
                colors.append(colors_states[1])
                color_labels.append(legend[1])
            elif hidden_states[xi[1]] == all_states[2]:
                colors.append(colors_states[2])
                color_labels.append(legend[2])
            elif hidden_states[xi[1]] == all_states[3]:
                colors.append(colors_states[3])
                color_labels.append(legend[3])

            segments.append(np.column_stack([xi, yi]))

        # Create a LineCollection from the segments and colors
        lc = LineCollection(segments, colors=colors, linewidth=2)

        # Add the LineCollection to the axis
        axis.add_collection(lc)

        # Set the title of the plot
        if title in ("ibi", "hf-hrv"):
            title = title.upper()
            ylabel = "Value (z-scored)"
        else:
            title = title.replace("_", " ").title()
            ylabel = "Power (z-scored)"

        axis.set_title(title, fontsize=14, fontweight="bold")

        # Adjust the limits of the plot
        axis.set_xlim(-1, len(data) + 1)
        axis.set_ylim(data.min()-1, data.max()+1)

        # Create custom legend handles
        legend_handles = [
            Line2D([0], [0], color=colors_states[i], lw=2, label=legend[i]) for i in range(len(all_states))
                    ]

        # Add the legend to the plot
        axis.legend(handles=legend_handles, loc="upper right")

        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel, labelpad=20)
    

    def catplot_states(hidden_states):
        """
        Create a categorical plot of the hidden states over time.

        Args:
        ----
            hidden_states (np.array): The hidden states to plot.
            axis (matplotlib.axis): The axis to plot on.
        
        Returns:
        -------
            fig (matplotlib.figure): The figure with the
                categorical plot of the hidden states.
        """
        list_states = hidden_states.unique()
        list_states.sort()

        # Add time to the hidden states
        hidden_states = pd.DataFrame(hidden_states)
        hidden_states["time"] = hidden_states.index

        # Create a subplot for each state
        fig, axis = plt.subplots(len(list_states), 1, figsize=(10, len(list_states)))
        
        sns.set(style="white")

        for state in list_states:
            # Transform the data to 1s and 0s
            hidden_states_binary = hidden_states.copy()
            for row in hidden_states_binary.iterrows():
                if row[1]["state"] == state:
                    hidden_states_binary.loc[row[0], "state"] = 1
                else:
                    hidden_states_binary.loc[row[0], "state"] = 0

            # Create the plot
            axis[state].plot(hidden_states_binary["time"], hidden_states_binary["state"], color=colors_states[state])
            # Fill the area under the curve
            axis[state].fill_between(hidden_states_binary["time"], hidden_states_binary["state"], color=colors_states[state], alpha=0.3)
            # Rotate the ylabel
            axis[state].set_ylabel(f"State {state+1}", rotation=360, labelpad=30)
            # Remove the x-label for all except the last subplot
            axis[state].set_xlabel("") if state != max(list_states) else axis[state].set_xlabel("Time (s)")

            # Remove the y-ticks
            axis[state].set_yticks([])

            # Adjust the limits of the plot
            axis[state].set_xlim(-1, len(hidden_states) + 1)

        return fig

    def create_violinplot(data, axis, variable, ylabel):
        """
        Create violin plots for global statistics of the hidden states.

        Args:
        ----
            data (pd.DataFrame): The data to plot.
            axis (matplotlib.axis): The axis to plot on.
            variable (str): The variable to plot.
            ylabel (str): The ylabel of the plot.
        """
        sns.violinplot(data=data, x="state", y=variable, palette=colors_states, ax=axis, inner=None)

        # Customizing the violin plots (color and transparency)
        for index, violin in enumerate(axis.collections[::1]):
            face_color = colors_states[index]
            violin.set_facecolor(plt.matplotlib.colors.to_rgba(face_color, alpha=0.3))
            violin.set_edgecolor(face_color)
            violin.set_linewidth(2)
        
        # Overlay the individual data points with stripplot
        sns.stripplot(data=data, x="state", y=variable, palette=colors_states, ax=axis, size=4, jitter=False)

        # Add labels
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_xlabel("")
        # Set y-limits
        if variable == "fractional_occupancy":
            axis.set_ylim(0, 1)
        # Change the x-ticks
        axis.set_xticklabels([f"State {i+1}" for i in range(4)], fontsize=12)
        # Add title
        title = " ".join(variable.split("_")).title() if variable == "fractional_occupancy" else variable.split("_")[1].capitalize()
        axis.set_title(title, fontsize=14, fontweight="bold")

    def create_affect_grid(data, data_mean, axis):
        """
        Create a plot of the mean valence and arousal ratings for each state on the Affect Grid.

        Args:
        ----
            data (pd.DataFrame): The data to plot.
            data_mean (pd.DataFrame): The mean data to plot.
            axis (matplotlib.axis): The axis to plot on.
        """
        # Create the plot
        sns.scatterplot(data=data, x="valence", y="arousal", hue="state", palette=colors_states, ax=axis, s=50, alpha=0.4)

        # Add the mean ratings
        sns.scatterplot(data=data_mean, x="valence", y="arousal", hue="state", palette=colors_states, ax=axis, s=150, alpha=1)

        # Add the standard deviation of the ratings as lines to the points
        for i in range(4):
            axis.errorbar(
                data_mean[data_mean["state"] == i]["valence"],
                data_mean[data_mean["state"] == i]["arousal"],
                xerr=data_mean[data_mean["state"] == i]["std_valence"],
                yerr=data_mean[data_mean["state"] == i]["std_arousal"],
                fmt="none",
                ecolor=colors_states[i],
                capsize=5,
                capthick=2,
                elinewidth=2)
        
        # Change the axes to be centered
        axis.spines["left"].set_position("center")
        axis.spines["bottom"].set_position("center")

        # Hide the top and right spines
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        # Add labels to the axes outside of the plot
        axis.set_xlabel("Valence", fontsize=12)
        axis.set_ylabel("Arousal", fontsize=12, rotation=360)

        # Move the x-axis label to the right middle
        axis.xaxis.set_label_coords(0.96, 0.53)
    
        # Move the y-axis label to the top middle
        axis.yaxis.set_label_coords(0.55, 0.98)

        # Set the limits of the plot
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)

        # Hide only the middle tick label on the x-axis and y-axis
        xticks = axis.xaxis.get_major_ticks()
        xticks[3].label1.set_visible(False)
        yticks = axis.yaxis.get_major_ticks()
        yticks[3].label1.set_visible(False)

        # Set custom legend labels
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_states[i], markersize=10, label=f"State {i+1}") for i in range(4)
        ]

        axis.legend(handles=legend_handles, loc="upper right")

        # Add grid lines
        plt.grid(True)

    def create_raincloud_plots(data, axis, variable, significant_differences):
        """
        Create raincloud plots for the ECG features with significant differences marked.

        Raincloud plots are made up of three different parts:
        - Boxplots.
        - Violin Plots.
        - Scatter Plots.

        Args:
        ----
            data (pd.DataFrame): The data to plot.
            axis (matplotlib.axis): The axis to plot on.
            variable (str): The variable to plot.
            significant_differences (list): The significant differences to mark.
        """
        # Divide data for each part of the plot
        data_list = []
        list_states = data["state"].unique()
        list_states.sort()
        for state in list_states:
            data_state = data[data["state"] == state][f"{variable}"]
            data_state = data_state.reset_index(drop=True)
            data_list.append(data_state)

        # Exclude nan values from the data
        data_list = [data[~np.isnan(data)] for data in data_list]

        # Boxplots
        # Boxplot data
        boxplot_data = axis.boxplot(
            data_list, widths=0.15, patch_artist=True, showfliers=False, medianprops=dict(color="black"))

        # Change to the desired color and add transparency
        for patch, color in zip(boxplot_data["boxes"], colors_states, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Violin Plots
        # Violin plot data
        violin_data = axis.violinplot(
            data_list, points=int(len(data)/len(list_states)), showmeans=False, showmedians=False, showextrema=False
        )


        for idx, body in enumerate(violin_data["bodies"]):
            # Get the center of the plot
            center = np.mean(body.get_paths()[0].vertices[:, 0])
            # Modify it so we only see the upper half of the violin plot
            body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], center, np.inf)
            # Change to the desired color
            body.set_color(colors_states[list_states[idx]])

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
            axis.scatter(x, features, s=3, c=colors_states[list_states[idx]])

        # Set labels
        axis.set_xticklabels([f"State {state+1}" for state in list_states])
        axis.set_ylabel("Value (z-scored)")

        # Set title
        axis.set_title(f"{variable.upper()}", fontsize=14, fontweight="bold", pad=20)

        if mark_significant_differences:
            # Mark significant differences with an asterisk and a line above the two groups
            counter = 0
            # Get distance between lines
            distance = 0.5

            for difference in significant_differences:
                first_group = f"State {difference[0]+1}"
                second_group = f"State {difference[1]+1}"
                # Get x-tick labels and positions
                x_tick_labels = [tick.get_text() for tick in axis.get_xticklabels()]
                xtick_positions = axis.get_xticks()

                # Get position of the label for the first group
                label_index_first = x_tick_labels.index(first_group)
                specific_xtick_position_first_group = xtick_positions[label_index_first]
                # Get position of the label for the second group
                label_index_second = x_tick_labels.index(second_group)
                specific_xtick_position_second_group = xtick_positions[label_index_second]

                # Get maximum value
                max_value = np.max(np.concatenate(data_list))

                # The color of line and asterisk should be black
                color = "black"

                # Plot a line between the two groups
                axis.plot(
                    [specific_xtick_position_first_group, specific_xtick_position_second_group],
                    [max_value + distance + counter, max_value + distance + counter],
                    color=color,
                )

                # Add an asterisk in the middle of the line
                axis.text(
                    (specific_xtick_position_first_group + specific_xtick_position_second_group) / 2,
                    max_value + distance + counter,
                    "*",
                    fontsize=12,
                    color=color,
                )

                counter += 0.3

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% STEP 1. LOAD DATA
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Plotting hidden states for {model} model...")
        print("+++++++++++++++++++++++++++++++++\n")

        hmm_path = resultpath / "avg" / "hmm" / model
        glm_path = resultpath / "avg" / "glm" / model

        features_string = "_".join(models_features[model])

        # Load the data
        hmm_data = pd.read_csv(hmm_path / f"all_subjects_task-AVR_{model}_model_data_{features_string}.tsv", sep="\t")
        hmm_feature_stats_all = pd.read_csv(hmm_path / f"all_subjects_task-AVR_{model}_model_states_stats.tsv", sep="\t")
        hmm_global_stats_all = pd.read_csv(hmm_path / f"all_subjects_task-AVR_{model}_model_states_global_stats.tsv", sep="\t")
        hmm_feature_stats_avg = pd.read_csv(hmm_path / f"avg_task-AVR_{model}_model_states_stats.tsv", sep="\t")
        hmm_global_stats_avg = pd.read_csv(hmm_path / f"avg_task-AVR_{model}_model_states_global_stats.tsv", sep="\t")

        # Read in the results of the post-hoc-tests
        posthoc_results = pd.read_csv(glm_path / f"avg_task-AVR_{model}_model_glm_results_posthoc_tests.tsv", sep="\t")

        events = pd.read_csv(events_dir / "events_experiment.tsv", sep="\t")
    
        # %% STEP 2. PLOT EXAMPLE OF HMM ALGORITHM FOR METHODS
        # Get data for example subject
        example_subject_data = hmm_data[hmm_data["subject"] == int(example_subject)]

        # Get the time window for the example plot
        example_data = example_subject_data[
            (example_subject_data["timestamp"] >= time_window[0] * 60) & (example_subject_data["timestamp"] <= time_window[1] * 60)
        ]

        example_data = example_data.reset_index(drop=True)
        hidden_states = example_data["state"]

        # Create the figure
        sns.set(style="ticks")
        fig, axs = plt.subplots(len(models_features[model]), 1, figsize=(10, 5 * len(models_features[model])))

        # Plot the example data
        for feature_index, feature in enumerate(models_features[model]):
            example_plot(
                example_data[feature],
                hidden_states,
                axs[feature_index] if len(models_features[model]) > 1 else axs,
                feature,
                [f"State {i+1}" for i in range(4)],
            )

        # Remove the x-labels from all but the last subplot
        for ax in axs[:-1]:
            ax.set_xlabel("")

        # Save the plot
        plot_file = f"example_{model}_hmm_{features_string}.svg"
        plt.savefig(hmm_path / plot_file)

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()

        # Plot the categorical plot of the hidden states
        fig = catplot_states(hidden_states)

        # Save the plot
        plot_file = f"example_{model}_hmm_states_{features_string}.svg"
        fig.savefig(hmm_path / plot_file)

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()

        # %% STEP 3. CREATE ONE SET OF PLOTS FOR EACH HMM
        # 3a. Violin plots for all four states for fractional occupancy, life times, interval times
        sns.set(style="ticks")
        figure, axes = plt.subplots(3, 1, figsize=(10, 15))

        for index, variable in enumerate(["fractional_occupancy", "mean_lifetime", "mean_intervaltime"]):
            create_violinplot(hmm_global_stats_all, axes[index], variable, ("Proportion" if variable == "fractional_occupancy" else "Time (s)"))

        # Save the plot
        plot_file = f"global_stats_{model}_model.svg"
        plt.savefig(hmm_path / plot_file)

        # Show the plot
        if show_plots:
            plt.show()
        
        plt.close()

        # 3b. Mean valence and arousal ratings for each state on the Affect Grid
        arousal_all_subjects = hmm_feature_stats_all[hmm_feature_stats_all["feature"] == "arousal"]
        arousal_all_subjects = arousal_all_subjects.drop(columns=["feature", "subject", "std", "min", "max"])
        arousal_all_subjects = arousal_all_subjects.rename(columns={"mean": "arousal"})
        arousal_all_subjects = arousal_all_subjects.reset_index(drop=True)

        valence_all_subjects = hmm_feature_stats_all[hmm_feature_stats_all["feature"] == "valence"]
        valence_all_subjects = valence_all_subjects.drop(columns=["feature", "subject", "std", "min", "max", "state"])
        valence_all_subjects = valence_all_subjects.rename(columns={"mean": "valence"})
        valence_all_subjects = valence_all_subjects.reset_index(drop=True)

        # Combine the data
        ratings_dataframe_all_subjects = pd.concat([valence_all_subjects, arousal_all_subjects], axis=1)

        # Get the mean ratings for each state
        arousal_mean = hmm_feature_stats_avg[hmm_feature_stats_avg["feature"] == "arousal"]
        arousal_mean = arousal_mean.drop(columns=["feature", "min", "max"])
        arousal_mean = arousal_mean.rename(columns={"mean": "arousal", "std": "std_arousal"})
        arousal_mean = arousal_mean.reset_index(drop=True)

        valence_mean = hmm_feature_stats_avg[hmm_feature_stats_avg["feature"] == "valence"]
        valence_mean = valence_mean.drop(columns=["feature", "min", "max", "state"])
        valence_mean = valence_mean.rename(columns={"mean": "valence", "std": "std_valence"})
        valence_mean = valence_mean.reset_index(drop=True)

        # Combine the data
        ratings_dataframe_avg = pd.concat([valence_mean, arousal_mean], axis=1)

        sns.set(style="ticks")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        create_affect_grid(ratings_dataframe_all_subjects, ratings_dataframe_avg, ax)

        # Save the plot
        plot_file = f"affect_grid_{model}_model.svg"
        plt.savefig(hmm_path / plot_file)

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()

        # 3c. Raincloud plots for each state for all ECG features with significant differences marked
        cardiac_features = models_features["cardiac"]

        # Get the significant differences
        significant_differences_dataframe = posthoc_results[posthoc_results["Significance"] == True]  # noqa: E712

        # Plot raincloud plots
        sns.set(style="ticks")

        fig, axes = plt.subplots(len(cardiac_features), 1, figsize=(10, 15))

        for index, feature in enumerate(cardiac_features):
            # Get the data for the current feature
            features_data = hmm_feature_stats_all[hmm_feature_stats_all["feature"] == feature]
            features_data = features_data.drop(columns=["feature", "subject", "std", "min", "max"])
            features_data = features_data.rename(columns={"mean": feature})
            features_data = features_data.reset_index(drop=True)

            # Get tjhe significant differences for the current feature
            significant_differences_current = significant_differences_dataframe[
                significant_differences_dataframe["Variable"] == feature
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

            create_raincloud_plots(features_data, axes[index], feature, significant_differences)

        # Save the plot
        plot_file = f"raincloud_plots_cardiac_features_{model}_model.svg"
        plt.savefig(hmm_path / plot_file)

        # Show the plot
        if show_plots:
            plt.show()

        plt.close()


        # 3d. Topoplots for each state for all EEG features


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    plot_hidden_states()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END


