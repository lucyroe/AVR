"""
Plotting descriptive statistics for the AVR data.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 12 August 2024
Last updated: 12 August 2024
"""

# %%
def plot_descriptives(  # noqa: C901, PLR0915
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    results_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
    show_plots=True,
):
    """
    Plot descriptive statistics for the AVR data.

    The following steps are performed:
        1. TODO: Add steps.
    """
    # %% Import
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # Path where the data is stored
    data_dir = Path(data_dir) / "phase3" / "AVR" / "derivatives"
    annotation_dir = Path(data_dir) / "preproc" / "avg" / "beh"
    physiological_dir = Path(data_dir) / "features" / "avg" / "eeg"
    events_dir = Path(data_dir) / "preproc"

    # Path where results should be saved
    results_dir_descriptives = Path(results_dir) / "phase3" / "AVR" / "avg"

    # List of datastreams
    datastreams = {"annotation": ["valence", "arousal"],
                "physiological": ["ibi", "hrv", "lf-hrv", "hf-hrv",
                "posterior_alpha", "frontal_alpha", "frontal_theta", "gamma", "beta"]}
    
    # List of videos
    videos = ["spaceship", "invasion", "asteroids", "underwood"]

    # Colors
    colors = {"annotation": {"valence": "#0072B2", "arousal": "#0096c6"},   # shades of dark blue
                "physiological": {"ibi": "#CC79A7", "hrv": "#f3849b", "lf-hrv": "#ff9789", "hf-hrv": "#ffb375",
                # shades of pink-orange
                "posterior_alpha": "#009E73", "frontal_alpha": "#66b974", "frontal_theta": "#a5d279",
                "gamma": "#00c787", "beta": "#85b082"}    # shades of green
    }

    colors_videos = {"spaceship": "#6C6C6C",    # grey
                    "invasion": "#56B4E9",      # shades of light blue
                    "asteroids": "#649ee1",
                    "underwood": "#7a86d1"}

    mark_significant_differences = True  # if True, significant differences will be marked in the boxplots

    # %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    def plot_average_timeseries(data, variable_name, colors, colors_videos, ax):
        fig = sns.lineplot(data=data, x="timestamp", y=variable_name,
                            hue="variable", palette=colors, ax=ax,
                            legend="brief",
                            lw=1.5)

        # Add shading for the different videos
        for video in videos:
            if video == "spaceship":
                timestamps_start_video = timestamps_events[event_names == f"start_{video}"]
                timestamps_stop_video = timestamps_events[event_names == f"end_{video}"]
                for timestamp_start_video, timestamp_stop_video in zip(timestamps_start_video, timestamps_stop_video):
                    ax.axvspan(timestamp_start_video, timestamp_stop_video, color=colors_videos[video], alpha=0.1)
            else:
                # Get the timestamps of the start and end of the video
                timestamp_start_video = float(timestamps_events[event_names == f"start_{video}"])
                timestamp_stop_video = float(timestamps_events[event_names == f"end_{video}"])
                # Add a text label for the video
                ax.text(timestamp_start_video + (timestamp_stop_video-timestamp_start_video)/3,
                        max(data[variable_name]), video.capitalize(), color="white",
                        fontsize=14, fontweight="bold", bbox=dict(facecolor=colors_videos[video], alpha=0.8))

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
        variables_text = " & ".join((" ".join(variable.title().split("_")) if variable_name != "value" else variable.upper()) for variable in variables)
        # Set the title
        fig.set_title(f"{variables_text}", fontsize=16, fontweight="bold", pad=20)




    def plot_boxplot(data, variable, color, ax):
        pass

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    # %% STEP 1. LOAD DATA
    # Create one dataframe for all datastreams
    data = pd.DataFrame()
    # Load the annotation data
    annotations = pd.read_csv(annotation_dir / "avg_task-AVR_beh_preprocessed.tsv", sep="\t")
    for variable in datastreams["annotation"]:
        data[variable] = annotations[variable]

    # Load the physiological data
    physiological = pd.read_csv(physiological_dir / "avg_task-AVR_physio_features.tsv", sep="\t")
    for variable in datastreams["physiological"]:
        # Check if the variable has the same length as the annotations
        for annotation_variable in datastreams["annotation"]:
            if len(physiological[variable]) != len(data[annotation_variable]):
                # Cut the annotation data to match the length of the physiological data
                data[annotation_variable] = data[annotation_variable][:len(physiological[variable])]
        data[variable] = physiological[variable]

    # Delete rows with NaN values
    data = data.dropna()

    # Add timestamp as first column
    data.insert(0, "timestamp", np.arange(len(data)))

    # Get events
    events = pd.read_csv(events_dir / "events_experiment.tsv", sep="\t")
    timestamps_events = events["onset"]
    event_names = events["event_name"]

    # Create a list of the different phases that has the same length as the data
    video_column = pd.DataFrame(index=data.index, columns=["video"])
    for video in videos:
        counter = 0
        for row in data.iterrows():
            # Get the timestamps of the start and end of the video
            timestamp_start_video = timestamps_events[event_names == f"start_{video}"].reset_index()["onset"][counter]
            timestamp_stop_video = timestamps_events[event_names == f"end_{video}"].reset_index()["onset"][counter]
            if row[1]["timestamp"] >= timestamp_start_video and row[1]["timestamp"] <= timestamp_stop_video:
                video_column.loc[row[0], "video"] = video
            if video == "spaceship" and row[1]["timestamp"] >= timestamp_stop_video:
                counter += 1
    data["video"] = video_column

    # %% STEP 2. PLOT AVERAGE TIMESERIES
    # Create a figure with subplots for annotation ratings, for ecg features, and for eeg features
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(8, 1)
    sns.set(style="whitegrid")

    # Plot annotation ratings
    annotation_axis = fig.add_subplot(gs[0])
    annotation_data = data[["timestamp"] + datastreams["annotation"]]
    # Format data for plotting
    annotation_data = pd.melt(annotation_data, id_vars="timestamp", var_name="variable", value_name="rating")
    plot_average_timeseries(annotation_data, "rating", colors["annotation"], colors_videos, annotation_axis)

    # Plot physiological features
    # ECG features
    ecg_features = datastreams["physiological"][0:4]
    colors_ecg = [colors["physiological"][ecg_feature] for ecg_feature in ecg_features]
    ecg_data = data[["timestamp"] + ecg_features]
    # Format data for plotting
    ecg_data = pd.melt(ecg_data, id_vars="timestamp", var_name="variable", value_name="value")
    # Plot IBI in one plot
    ibi_axis = fig.add_subplot(gs[1])
    ibi_data = ecg_data[ecg_data["variable"] == "ibi"]
    plot_average_timeseries(ibi_data, "value", [colors["physiological"]["ibi"]], colors_videos, ibi_axis)
    # Other ECG features in one plot
    ecg_axis = fig.add_subplot(gs[2])
    plot_average_timeseries(ecg_data[ecg_data["variable"] != "ibi"], "value", colors_ecg[1:], colors_videos, ecg_axis)

    # EEG features
    # Create one plot for each EEG feature
    eeg_features = datastreams["physiological"][4:]
    index = 3
    for eeg_feature in eeg_features:
        colors_eeg = [colors["physiological"][eeg_feature]]
        eeg_data = data[["timestamp", eeg_feature]]
        # Format data for plotting
        eeg_data = pd.melt(eeg_data, id_vars="timestamp", var_name="variable", value_name="power")
        eeg_axis = fig.add_subplot(gs[index])
        plot_average_timeseries(eeg_data, "power", colors_eeg, colors_videos, eeg_axis)
        index += 1
    
    # Remove the x-axis label for all subplots except the last one
    for ax in fig.get_axes():
        if ax != eeg_axis:
            ax.set_xlabel("")
    
    # Increase space between subplots
    plt.tight_layout()



# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    plot_descriptives()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
