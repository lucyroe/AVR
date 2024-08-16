"""
Script to calculate averaged preprocessing metadata.

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 16 August 2024
Last update: 16 August 2024
"""


def preprocessing_metadata(
    subjects=[  # noqa: B006
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "009",
        "011",
        "012",
        "014",
        "015",
        "016",
        "017",
        "018",
        "019",
        "020",
        "021",
        "022",
        "023",
        "024",
        "025",
        "026",
        "027",
        "028",
        "029",
        "030",
        "031",
        "032",
        "033",
        "034",
        "035",
        "036",
        "037",
        "038",
        "039",
        "040",
        "041",
        "042",
        "043",
        "044",
        "045",
        "046",
        "047",
    ],
    data_dir="/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/",
    debug=False,
):
    """
    Calculate averaged preprocessing metadata.

    Inputs: json file with preprocessing metadata for each subject

    Outputs: averaged preprocessing metadata

    Steps:
    1.
    """
    # %% Import
    import json
    from pathlib import Path

    import pandas as pd

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    task = "AVR"

    # Define resultpath
    result_path = Path(data_dir) / "phase3" / task / "derivatives" / "preproc" / "avg" / "eeg"

    # Only analyze one subject when debug mode is on
    if debug:
        subjects = [subjects[0]]

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # %% STEP 1: LOAD METADATA
    # Initiate empty dataframe for metadata
    metadata = pd.DataFrame(
        columns=[
            "subject",
            "number_bad_channels_x_epochs",
            "number_bad_epochs",
            "number_interpolated_channels_x_epochs",
            "number_ICA_components",
            "number_eog_components",
            "number_ecg_components",
            "number_emg_components",
            "number_other_components",
            "explained_variance",
        ]
    )
    # Loop over subjects
    for subject_index, subject in enumerate(subjects):
        # Load metadata
        metadata_file = (
            Path(data_dir)
            / "phase3"
            / task
            / "derivatives"
            / "preproc"
            / f"sub-{subject}"
            / "eeg"
            / f"sub-{subject}_task-{task}_physio_preprocessing_metadata.json"
        )

        with metadata_file.open("r") as file:
            metadata_subject = json.load(file)

        # Append metadata to dataframe
        metadata.loc[subject_index, "subject"] = subject
        metadata.loc[subject_index, "number_bad_channels_x_epochs"] = metadata_subject[
            "number_of_bad_channels_x_epochs"
        ]
        metadata.loc[subject_index, "number_bad_epochs"] = metadata_subject["number_of_bad_epochs"]
        metadata.loc[subject_index, "number_interpolated_channels_x_epochs"] = metadata_subject[
            "number_of_interpolated_channels_x_epochs"
        ]
        metadata.loc[subject_index, "number_ICA_components"] = metadata_subject["number_of_all_components"]
        metadata.loc[subject_index, "number_eog_components"] = len(metadata_subject["eog_components"])
        metadata.loc[subject_index, "number_ecg_components"] = len(metadata_subject["ecg_components"])
        metadata.loc[subject_index, "number_emg_components"] = len(metadata_subject["emg_components"])
        metadata.loc[subject_index, "number_other_components"] = len(metadata_subject["other_components"])
        metadata.loc[subject_index, "explained_variance"] = metadata_subject["explained_variance_ratio"]

    # %% STEP 2: AVERAGE METADATA
    # Initialize empty dataframe for averaged metadata
    averaged_metadata = pd.DataFrame(
        columns=[
            "number_bad_channels_x_epochs",
            "number_bad_epochs",
            "number_interpolated_channels_x_epochs",
            "number_ICA_components",
            "number_eog_components",
            "number_ecg_components",
            "number_emg_components",
            "number_other_components",
            "explained_variance",
            "max_ICA_components",
            "min_ICA_components",
            "max_eog_components",
            "min_eog_components",
            "max_ecg_components",
            "min_ecg_components",
            "max_emg_components",
            "min_emg_components",
            "max_other_components",
            "min_other_components",
        ]
    )
    # Drop subject column
    metadata = metadata.drop(columns=["subject"])

    # Calculate mean
    averaged_metadata.loc["number_bad_channels_x_epochs"] = metadata["number_bad_channels_x_epochs"].mean()
    averaged_metadata.loc["number_bad_epochs"] = metadata["number_bad_epochs"].mean()
    averaged_metadata.loc["number_interpolated_channels_x_epochs"] = metadata[
        "number_interpolated_channels_x_epochs"
    ].mean()
    averaged_metadata.loc["number_ICA_components"] = metadata["number_ICA_components"].mean()
    averaged_metadata.loc["number_eog_components"] = metadata["number_eog_components"].mean()
    averaged_metadata.loc["number_ecg_components"] = metadata["number_ecg_components"].mean()
    averaged_metadata.loc["number_emg_components"] = metadata["number_emg_components"].mean()
    averaged_metadata.loc["number_other_components"] = metadata["number_other_components"].mean()
    averaged_metadata.loc["explained_variance"] = metadata["explained_variance"].mean()

    # Calculate min and max of components
    averaged_metadata.loc["max_ICA_components"] = metadata["number_ICA_components"].max()
    averaged_metadata.loc["min_ICA_components"] = metadata["number_ICA_components"].min()
    averaged_metadata.loc["max_eog_components"] = metadata["number_eog_components"].max()
    averaged_metadata.loc["min_eog_components"] = metadata["number_eog_components"].min()
    averaged_metadata.loc["max_ecg_components"] = metadata["number_ecg_components"].max()
    averaged_metadata.loc["min_ecg_components"] = metadata["number_ecg_components"].min()
    averaged_metadata.loc["max_emg_components"] = metadata["number_emg_components"].max()
    averaged_metadata.loc["min_emg_components"] = metadata["number_emg_components"].min()
    averaged_metadata.loc["max_other_components"] = metadata["number_other_components"].max()
    averaged_metadata.loc["min_other_components"] = metadata["number_other_components"].min()

    # %% STEP 3: SAVE AVERAGED METADATA
    # Save averaged metadata
    averaged_metadata_file = result_path / "preprocessing_metadata.json"
    averaged_metadata.to_csv(averaged_metadata_file, index=False, sep="\t")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    preprocessing_metadata()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
