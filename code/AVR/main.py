"""
Main module for AVR project.

Required packages:  numpy, pandas, json, time, pathlib, pyxdf, gzip, sys,
                    mne, neurokit2, systole, autoreject,
                    fcwt, scipy, foof, statsmodels,
                    matplotlib, seaborn, IPython

Author: Lucy Roellecke
Contact: lucy.roellecke[at]tuta.com
Created on: 1 August 2024
Last update: 12 August 2024
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
        4. Univariate statistics: Perform univariate statistical analysis.
        5. Modelling: Perform Hidden Markov Model (HMM) analysis.
        6. Plot results: Plot the results of the analysis.
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
    from AVR.statistics.univariate_statistics import univariate_statistics
    #from AVR.datacomparison.plot_descriptives import plot_descriptives

    # %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    steps = ["Preprocess data"]    # Adjust as needed
    # "Load data", "Preprocess data", "Extract features", "Univariate statistics", "Plot results", "Modelling"

    subjects = ['011', '017', '029', '033', '035', '036', '039', '044', '047']
                #"026", "027", "028", "029", "030",
                #"031", "032", "033", "034", "035", "036", "037", "038", "039", "040",
                #"041", "042", "043", "044", "045", "046", "047"]
    # "001", "002", "003","004", "005", "006", "007", "009",    # Already done
    # "011", "012", "014", "015", "016", "017", "018", "019",   # Already done
    # "020", "021", "022", "024", "025"                        # Already done

    # subjects "008", "010", "013" were excluded due to missing data
    # subject "023" was excluded because of bad quality of ECG data

    # Only needed for comparison of phase 3 with phase 1
    subjects_phase1 = ["06", "08", "10", "12", "14", "16", "18", "19", "20",
                        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                        "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                        "41", "42", "43", "44", "45", "46", "47", "48", "49", "51",
                        "53", "55", "57", "59", "61", "63", "65", "67", "69", "71", "73", "75"]

    # Specify the data path info (in BIDS format)
    # change with the directory of data storage
    data_dir = "E:/AffectiveVR/Phase_3/Data/"
    results_dir = "E:/AffectiveVR/Phase_3/Results/"

    # Define if plots should be shown
    show_plots = True

    # Define whether manual cleaning of the data is required (cleaning of R-peaks for ECG data)
    manual_cleaning = True

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
            #print("\nPreprocessing annotations...\n")
            #preprocess_annotations(subjects, data_dir, results_dir, show_plots, debug)
            print("\nPreprocessing physiological data...\n")
            preprocess_physiological(subjects, data_dir, results_dir, show_plots, debug, manual_cleaning)

        elif step == "Extract features":
            extract_features(subjects, data_dir, results_dir, show_plots, debug)

        elif step == "Univariate statistics":
            print("\nPerforming univariate statistical analysis...\n")
            univariate_statistics(subjects, data_dir, results_dir, show_plots, debug)
            print("\nPerforming statistical comparison of variability in ratings between phase 1 and phase 3...\n")
            compare_variability_phase1_phase3(subjects, subjects_phase1, data_dir, results_dir, show_plots)

        elif step == "Plot results":
            print("\nPlotting results...\n")
            print("\nCreating descriptives plots...\n")
            plot_descriptives(data_dir, results_dir, show_plots)
            print("\nCreating raincloud plots to compare variability in ratings between phase 1 and phase 3...\n")
            raincloud_plot(data_dir, results_dir, show_plots)

        elif step == "Modelling":
            print("\nPerforming Hidden Markov Model (HMM) analysis...\n")

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

# %%
