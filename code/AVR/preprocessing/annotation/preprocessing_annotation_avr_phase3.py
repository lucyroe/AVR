"""
Script to preprocess annotation data for AVR phase 3.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 6 July 2024
Last update: 1 August 2024
"""

def preprocess_annotations(subjects=[],  # noqa: PLR0915, B006
            data_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
            results_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/",
            show_plots=False,
            debug=False):
    """
    Preprocess annotation data for AVR phase 3.

    Inputs: Raw annotation data
    Outputs: Preprocessed annotation data as tsv file (both for subjects separately and averaged over all subjects)

    Steps:
    1. LOAD DATA
    2. PREPROCESS DATA
        2a. Cutting data
        2b. Format data
        2c. Plot arousal and valence data
    3. AVERAGE OVER ALL PARTICIPANTS
        3a. Concatenate data of all participants into one dataframe
        3b. Save dataframe with all participants in tsv file ("all_subjects_task-{task}_beh_preprocessed.tsv")
        3c. Calculate averaged data
        3d. Save dataframe with mean arousal data in tsv file ("avg_task-{task}_beh_preprocessed.tsv")
        3e. Plot mean arousal and valence data as sanity check
    """
    # %% Import
    import gzip
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    task = "AVR"

    # Only analyze one subject when debug mode is on
    if debug:
        subjects = [subjects[0]]

    # Define whether missing values should be interpolates
    interpolate_missing_values = True

    # Specify the data path info (in BIDS format)
    # change with the directory of data storage
    data_dir = Path(data_dir)
    exp_name = "AVR"
    rawdata_name = "rawdata"  # rawdata folder
    derivative_name = "derivatives"  # derivates folder
    preprocessed_name = "preproc"  # preprocessed folder (inside derivatives)
    averaged_name = "avg"  # averaged data folder (inside preprocessed)
    datatype_name = "beh"  # data type specification
    results_dir = Path(results_dir)

    # Create the preprocessed data folder if it does not exist
    for subject in subjects:
        subject_preprocessed_folder = (
            data_dir / exp_name / derivative_name / preprocessed_name / f"sub-{subject}" / datatype_name
        )
        subject_preprocessed_folder.mkdir(parents=True, exist_ok=True)
    avg_preprocessed_folder = data_dir / exp_name / derivative_name / preprocessed_name / averaged_name / datatype_name
    avg_preprocessed_folder.mkdir(parents=True, exist_ok=True)
    avg_results_folder = results_dir / exp_name / averaged_name / datatype_name
    avg_results_folder.mkdir(parents=True, exist_ok=True)

    # Define if the first and last 2.5 seconds of the data should be cut off
    # To avoid any potential artifacts at the beginning and end of the experiment
    cut_off_seconds = 2.5

    sampling_frequency = 90  # Sampling frequency of the data in Hz
    downsampling_frequency = 1  # Downsampling frequency in Hz
    resample_rate = sampling_frequency // downsampling_frequency

    # Create color palette for plots
    colors = {
        "valence": "#0072B2",  # dark blue
        "arousal": "#E69F00" # light orange
    }

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
    # %% STEP 1. LOAD DATA
    # Initialize list to store data of all participants
    list_data_all = []
    # Loop over all subjects
    for subject_index, subject in enumerate(subjects):
        print(f"Processing subject {subject_index+1} (ID {subject}) of " + str(len(subjects)) + "...")

        # Get right file for the dataset
        directory = Path(data_dir) / exp_name / rawdata_name / f"sub-{subject}" / datatype_name
        file = directory / f"sub-{subject}_task-{task}_{datatype_name}.tsv.gz"

        # Unzip and read in data
        with gzip.open(file, "rt") as f:
            data = pd.read_csv(f, sep="\t")

        # Load event markers for subject
        event_markers = pd.read_csv(directory / f"sub-{subject}_task-{task}_events.tsv", sep="\t")

        # Load mapping for event markers to real events
        event_mapping = pd.read_csv(data_dir / exp_name / rawdata_name / "events_mapping.tsv", sep="\t")

        # Drop column with trial type
        event_markers = event_markers.drop(columns=["trial_type"])

        # Add column with event names to event markers
        events = pd.concat([event_markers, event_mapping], axis=1)

        # Drop unnecessary columns
        events = events.drop(columns=["duration"])

        # %% STEP 2. PREPROCESS DATA
        # 2a. Cutting data
        # Get start and end time of the experiment
        start_time = events[events["event_name"] == "start_spaceship"].reset_index()["onset"].tolist()[0]
        end_time = events[events["event_name"] == "end_spaceship"].reset_index()["onset"].tolist()[-1]

        # Get events for experiment (from start to end of experiment)
        events_experiment = events[(events["onset"] >= start_time) & (events["onset"] <= end_time)]

        # Cut data to start and end time
        # And remove first and last 2.5 seconds of data (if specified above)
        if cut_off_seconds > 0:
            data = data[
                (data["onset"] >= start_time + cut_off_seconds) & (data["onset"] <= end_time - cut_off_seconds)
            ]
            # Adjust event markers accordingly
            events_experiment["onset"] = events_experiment["onset"] - cut_off_seconds
        else:
            data = data[(data["onset"] >= start_time) & (data["onset"] <= end_time)]

        # 2b. Format data
        # Set event time to start at - cut_off_seconds
        events_experiment["onset"] = events_experiment["onset"] - events_experiment["onset"].iloc[0] - cut_off_seconds

        # Remove unnecessary columns
        data = data.drop(columns=["duration", "onset"])

        # Downsample data from 90 to 1 Hz
        data = data.iloc[::resample_rate]

        # Reset index
        data = data.reset_index(drop=True)

        # Add timestamp column
        data.insert(0, "timestamp", data.index)

        # Interpolate missing values
        if interpolate_missing_values:
            data = data.interpolate()   # linear interpolation

        # Add subject ID to data as first column
        data.insert(0, "subject", subject)

        # Save preprocessed data as tsv file
        data.to_csv(
            Path(data_dir)
            / exp_name
            / derivative_name
            / preprocessed_name
            / f"sub-{subject}"
            / datatype_name
            / f"sub-{subject}_task-{task}_{datatype_name}_preprocessed.tsv",
            sep="\t",
            index=False,
        )

        # Add participant's data to dataframe for all participants
        list_data_all.append(data)

        # 2c. Plot arousal and valence data
        plt.figure(figsize=(15, 5))
        plt.plot(data["timestamp"], data["arousal"], label="Arousal", color=colors["arousal"])
        plt.plot(data["timestamp"], data["valence"], label="Valence", color=colors["valence"])
        plt.title(f"Ratings for {task} phase 3 for subject {subject}")

        # Add vertical lines for event markers
        # Exclude first and last event markers
        # And only use every second event marker to avoid overlap
        for _, row in events_experiment.iloc[0:-1:2].iterrows():
            plt.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)
            plt.text(row["onset"] - 100, -1.5, row["event_name"], rotation=30, fontsize=8, color="gray")

        # Transform x-axis to minutes
        plt.xticks(
            ticks=range(0, int(data["timestamp"].max()) + 1, 60),
            labels=[str(int(t / 60)) for t in range(0, int(data["timestamp"].max()) + 1, 60)],
        )
        plt.xlabel("Time (min)")
        plt.ylabel("Ratings")
        plt.legend(["Arousal", "Valence"])
        plt.ylim(-1.2, 1.2)

        # Save plot
        plt.savefig(Path(results_dir) / exp_name  / f"sub-{subject}" / datatype_name /
        f"sub-{subject}_task-{task}_{datatype_name}_preprocessed.png")

        if show_plots:
            plt.show()


    # %% STEP 3. AVERAGE OVER ALL PARTICIPANTS

    print("Finished preprocessing data for all participants. Now averaging over all participants...")

    # 3a. Concatenate data of all participants into one dataframe
    data_all = pd.concat(list_data_all, ignore_index=True)

    # 3b. Save dataframe with all participants in tsv file
    data_all.to_csv(
        Path(data_dir)
        / exp_name
        / derivative_name
        / preprocessed_name
        / averaged_name
        / datatype_name
        / f"all_subjects_task-{task}_{datatype_name}_preprocessed.tsv",
        sep="\t",
        index=False,
    )

    # Drop column with subject number
    data_all = data_all.drop(columns=["subject"])

    # 3c. Calculate averaged data
    data_mean = data_all.groupby("timestamp").mean()
    # Add time column as first column
    data_mean.insert(0, "timestamp", data_mean.index)

    # 3d. Save dataframe with mean arousal data in tsv file
    data_mean.to_csv(
        Path(data_dir)
        / exp_name
        / derivative_name
        / preprocessed_name
        / averaged_name
        / datatype_name
        / f"avg_task-{task}_{datatype_name}_preprocessed.tsv",
        sep="\t",
        index=False,
    )

    # 3e. Plot mean arousal and valence data
    plt.figure(figsize=(15, 5))
    plt.plot(data_mean["timestamp"], data_mean["arousal"], label="Arousal", color=colors["arousal"])
    plt.plot(data_mean["timestamp"], data_mean["valence"], label="Valence", color=colors["valence"])
    plt.title(f"Mean ratings for {task} phase 3 (n={len(subjects)})")

    # Add vertical lines for event markers
    # Exclude first and last event markers
    # And only use every second event marker to avoid overlap
    for _, row in events_experiment.iloc[0:-1:2].iterrows():
        plt.axvline(row["onset"], color="gray", linestyle="--", alpha=0.5)
        plt.text(row["onset"] - 100, -1.5, row["event_name"], rotation=30, fontsize=8, color="gray")

    # Transform x-axis to minutes
    plt.xticks(
        ticks=range(0, int(data_mean["timestamp"].max()) + 1, 60),
        labels=[str(int(t / 60)) for t in range(0, int(data_mean["timestamp"].max()) + 1, 60)],
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Ratings")
    plt.legend(["Arousal", "Valence"])
    plt.ylim(-1.2, 1.2)

    # Save plot
    plt.savefig(Path(results_dir) / exp_name / averaged_name / datatype_name /
    f"avg_task-{task}_{datatype_name}_preprocessed.png")

    plt.show()

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    preprocess_annotations()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
