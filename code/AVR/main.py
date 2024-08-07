"""
Main module for AVR project.

Required packages:  numpy, pandas, json, time, pathlib, pyxdf, gzip, sys,
                    mne, neurokit2, systole, autoreject,
                    fcwt, scipy, foof, statsmodels,
                    matplotlib, seaborn, IPython

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 1 August 2024
Last update: 7 August 2024
"""

def main():
    """
    Run the main function of the AVR project.

    This module is the entry point for the AVR project. It contains the main functions for loading the data,
    preprocessing the data, and running the analysis.

    The following steps are performed:
        1. Load data: Load the data from the data directory.
        2. Preprocess data: Preprocess the data, including annotations and physiological data.
        3. Extract features: Extract features from the physiological data.
    """
    # %% Import
    from AVR.datacomparison.compare_variability_phase1_phase3 import compare_variability_phase1_phase3
    from AVR.datavisualization.raincloud_plot import raincloud_plot
    from AVR.preprocessing.annotation.preprocessing_annotation_avr_phase3 import preprocess_annotations
    from AVR.preprocessing.physiological.feature_extraction import extract_features
    from AVR.preprocessing.physiological.preprocessing_physiological_avr_phase3 import (
        preprocess_physiological,
    )
    from AVR.preprocessing.read_xdf import read_xdf

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    steps = []    # Adjust as needed
    # "Load data", "Preprocess data", "Extract features", "Descriptive Statistics", "Plot results"

    subjects = ["001", "002", "003"]  # Adjust as needed

    # Only needed for comparison of phase 3 with phase 1
    subjects_phase1 = ["06", "08", "10", "12", "14", "16", "18", "19", "20",
                        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                        "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                        "41", "42", "43", "44", "45", "46", "47", "48", "49", "51",
                        "53", "55", "57", "59", "61", "63", "65", "67", "69", "71", "73", "75"]

    # Specify the data path info (in BIDS format)
    # change with the directory of data storage
    data_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"
    results_dir = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/"

    # Define if plots should be shown
    show_plots = False

    # Define whether manual cleaning of the data is required (cleaning of R-peaks for ECG data)
    manual_cleaning = False

    # Only analyze one subject when debug mode is on
    debug = False

    # %% Script  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
    for index, step in enumerate(steps):
        print("*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
        print(f"Running step: {step} (Step {index+1} of {len(steps)})")
        print("*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")

        if step == "Load data":
            read_xdf(subjects, data_dir, results_dir, show_plots, debug)

        elif step == "Preprocess data":
            print("\nPreprocessing annotations...\n")
            preprocess_annotations(subjects, data_dir, results_dir, show_plots, debug)
            print("\nPreprocessing physiological data...\n")
            preprocess_physiological(subjects, data_dir, results_dir, show_plots, debug, manual_cleaning)

        elif step == "Extract features":
            extract_features(subjects, data_dir, results_dir, show_plots, debug)

        elif step == "Descriptive Statistics":
            print("\nPerforming statistical comparison of variability between phase 1 and phase 3...\n")
            compare_variability_phase1_phase3(subjects, subjects_phase1, data_dir, results_dir, show_plots)
            print("\nCreating raincloud plots to compare variability between phase 1 and phase 3...\n")

        elif step == "Plot results":
            print("\nPlotting results...\n")
            raincloud_plot(data_dir, results_dir, show_plots)

        else:
            print(f"Step {step} not found")

    print("*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
    print("AVR pipeline completed successfully.")
    print("*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
if __name__ == "__main__":
    # Run the main function
    main()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
