"""
Plotting hidden states from HMM analysis of the AVR data.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 20 August 2024
Last updated: 20 August 2024
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
    models = ["cardiac", "neural", "integrated"]
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

    def create_violinplot(data, variable, ylabel):
        """
        Create violin plots for global statistics of the hidden states.

        Args:
        ----
            data (pd.DataFrame): The data to plot.
            variable (str): The variable to plot.
            ylabel (str): The ylabel of the plot.
        """
        fig, axis = plt.subplots(figsize=(10, 5))

        sns.violinplot(data=data, x="state", y=variable, palette=colors_states, ax=axis, inner=None)

        for index, violin in enumerate(plt.gca().collections[::1]):
            face_color = colors_states[index]
            violin.set_facecolor(plt.matplotlib.colors.to_rgba(face_color, alpha=0.3))
            violin.set_edgecolor(face_color)
            violin.set_linewidth(2)

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

        return fig

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% STEP 1. LOAD DATA
    # Loop over all models
    for model in models:
        print("+++++++++++++++++++++++++++++++++")
        print(f"Initiating {model} model...")
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
        for variable in ["fractional_occupancy", "mean_lifetime", "mean_intervaltime"]:
            sns.set(style="ticks")

            fig = create_violinplot(hmm_global_stats_all, variable, ("Proportion" if variable == "fractional_occupancy" else "Time (s)"))

            # Save the plot
            plot_file = f"global_stats_{model}_model_{variable}.svg"
            fig.savefig(hmm_path / plot_file)

            # Show the plot
            if show_plots:
                plt.show()
            
            plt.close()

        # 3b. Mean valence and arousal ratings for each state on the Affect Grid

        # 3c. Raincloud plots for each state for all ECG features with significant differences marked

        # 3d. Topoplots for each state for all EEG features


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    plot_hidden_states()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END


