########################################################################################################################
# Script to calculate summary statistics for CR data from CEAP dataset after preprocessing
#
# Input:        csv files created by 01_preprocessing.py with preprocessed data for each participant
#               for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv)
#
# Step 2a:      creates csv files with descriptive statistics for each quadrant separately for each participant
#               Outputs:      cr_affectivevr_descriptive_individual.csv, cr_ceap_descriptive_individual.csv
# Step 2b:      creates csv file with descriptive statistics for each quadrant averaged over participants
#               Outputs:      cr_affectivevr_descriptive.csv, cr_ceap_descriptive.csv
# Step 2c:      creates csv file with descriptive statistics averaged over quadrants for each participant
#               Outputs:      cr_affectivevr_descriptive_avgvideos_individual.csv, cr_ceap_descriptive_avgvideos_individual.csv
# Step 2d:      creates csv file with descriptive statistics averaged over quadrants averaged over participants
#               Outputs:      cr_affectivevr_descriptive_avgvideos.csv, cr_ceap_descriptive_avgvideos.csv
#
# Step 2e:      creates csv file with arousal and valence values for all timepoints, quadrants and rating methods averaged across participants
#               Outputs:      cr_affectivevr_all.csv, cr_ceap_all.csv
#
# Author:       Lucy Roellecke (lucy.roellecke[at]tuta.com)
# Last version: 02.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES ------------------------
import pandas as pd
import numpy as np
import os
from scipy import stats

# ------------------------- SETUP ------------------------------
# change the data_path to the path where you saved the preprocessed data
data_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CEAP/data/CEAP-360VR/3_AnnotationData/"

# change the results_path to the path where you want to save the results
results_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CEAP/results/descriptives/"

# get the file path for the file with the preprocessed AffectiveVR and CEAP data
preprocessed_file = os.path.join(data_path, "all_participants_ceap_affectivevr.csv")

# datasets to analyze
datasets = ["affectivevr"]
# "ceap"

# analysis steps to perform
steps = ["2e"]
# "2a", "2b", "2c", "2d"

# quadrants
quadrants = ["HP", "LP", "LN", "HN"]


# ----------------------- FUNCTIONS ----------------------------
# step 2a
# function to calculate descriptive statistics for each quadrant separately for each participant
def descriptives_individual(
    data: pd.DataFrame,
    quadrants: list[str],
    subjects: list[str],
    rating_methods: list[str],
) -> pd.DataFrame:
    # create new dataframe with summary descriptives
    cr_descriptives_individual = pd.DataFrame(
        {
            "sj_id": [],  # subject ID
            "rating_method": [],  # rating method
            "quadrant": [],  # quadrant
            "cr_mean_v": [],  # mean valence
            "cr_mean_a": [],  # mean arousal
            "cr_mean_dist": [],  # mean distance
            "cr_mean_angle": [],  # mean angle
            "cr_std_v": [],  # standard deviation valence
            "cr_std_a": [],  # standard deviation arousal
            "cr_std_dist": [],  # standard deviation distance
            "cr_std_angle": [],  # standard deviation angle
            "cr_skew_v": [],  # skewness valence
            "cr_skew_a": [],  # skewness arousal
            "cr_skew_dist": [],  # skewness distance
            "cr_skew_angle": [],  # skewness angle
            "cr_kurtosis_v": [],  # kurtosis valence
            "cr_kurtosis_a": [],  # kurtosis arousal
            "cr_kurtosis_dist": [],  # kurtosis distance
            "cr_kurtosis_angle": [],  # kurtosis angle
            "cr_auc_v": [],  # area under curve valence
            "cr_auc_a": [],  # area under curve arousal
            "cr_auc_dist": [],  # area under curve distance
            "cr_auc_angle": [],  # area under curve angle
        }
    )
    # loop over participants
    for subject_index, subject in enumerate(subjects):
        for rating_index, rating_method in enumerate(rating_methods):
            # get data for one subject for one rating method
            if len(rating_methods) > 3:
                if rating_method == "mean":
                    subject_data = data[data["sj_id"] == subject]
                else:
                    subject_data = data[data["sj_id"] == subject][
                        data["video_id"] == rating_method
                    ]
            else:
                subject_data = data[data["sj_id"] == subject][
                    data["rating_method"] == rating_method
                ]

            # loop over quadrants
            for row_index, quadrant in enumerate(quadrants):
                # mean
                cr_mean_v = subject_data["cr_v"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].mean(skipna=True)
                cr_mean_a = subject_data["cr_a"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].mean(skipna=True)
                cr_mean_dist = float(
                    np.nanmean(
                        subject_data["cr_dist"][
                            subject_data["quadrant"].str.contains(quadrant)
                        ]
                    )
                )
                cr_mean_angle = float(
                    np.nanmean(
                        subject_data["cr_angle"][
                            subject_data["quadrant"].str.contains(quadrant)
                        ]
                    )
                )

                # std
                cr_std_v = subject_data["cr_v"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].std(skipna=True)
                cr_std_a = subject_data["cr_a"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].std(skipna=True)
                cr_std_dist = float(
                    np.nanstd(
                        subject_data["cr_dist"][
                            subject_data["quadrant"].str.contains(quadrant)
                        ]
                    )
                )
                cr_std_angle = float(
                    np.nanstd(
                        subject_data["cr_angle"][
                            subject_data["quadrant"].str.contains(quadrant)
                        ]
                    )
                )

                # skewness
                cr_skew_v = subject_data["cr_v"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].skew(skipna=True)
                cr_skew_a = subject_data["cr_v"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].skew(skipna=True)
                cr_skew_dist = stats.skew(
                    subject_data["cr_dist"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ],
                    nan_policy="omit",
                )
                cr_skew_angle = stats.skew(
                    subject_data["cr_angle"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ],
                    nan_policy="omit",
                )

                # kutosis
                cr_kurtosis_v = subject_data["cr_v"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].kurtosis(skipna=True)
                cr_kurtosis_a = subject_data["cr_v"][
                    subject_data["quadrant"].str.contains(quadrant)
                ].kurtosis(skipna=True)
                cr_kurtosis_dist = stats.kurtosis(
                    subject_data["cr_dist"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ],
                    nan_policy="omit",
                )
                cr_kurtosis_angle = stats.kurtosis(
                    subject_data["cr_angle"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ],
                    nan_policy="omit",
                )

                # area under curve - use trapz or sum? ~ same results
                cr_auc_v = np.trapz(
                    subject_data["cr_v"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ].dropna(),
                    dx=1,
                )
                cr_auc_a = np.trapz(
                    subject_data["cr_v"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ].dropna(),
                    dx=1,
                )
                cr_auc_dist = np.trapz(
                    subject_data["cr_dist"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ][
                        ~np.isnan(
                            subject_data["cr_dist"][
                                subject_data["quadrant"].str.contains(quadrant)
                            ]
                        )
                    ],
                    dx=1,
                )
                cr_auc_angle = np.trapz(
                    subject_data["cr_angle"][
                        subject_data["quadrant"].str.contains(quadrant)
                    ][
                        ~np.isnan(
                            subject_data["cr_angle"][
                                subject_data["quadrant"].str.contains(quadrant)
                            ]
                        )
                    ],
                    dx=1,
                )

                # add descriptive stats of one quadrant to cr_descriptives_individual dataframe as a new row
                new_row_index = (
                    subject_index * len(quadrants) * len(rating_methods)
                    + rating_index * len(quadrants)
                    + row_index
                )
                cr_descriptives_individual.loc[new_row_index, "sj_id"] = subject
                cr_descriptives_individual.loc[
                    new_row_index, "rating_method"
                ] = rating_method
                cr_descriptives_individual.loc[new_row_index, "quadrant"] = quadrant
                cr_descriptives_individual.loc[new_row_index, "cr_mean_v"] = cr_mean_v
                cr_descriptives_individual.loc[new_row_index, "cr_mean_a"] = cr_mean_a
                cr_descriptives_individual.loc[
                    new_row_index, "cr_mean_dist"
                ] = cr_mean_dist
                cr_descriptives_individual.loc[
                    new_row_index, "cr_mean_angle"
                ] = cr_mean_angle
                cr_descriptives_individual.loc[new_row_index, "cr_std_v"] = cr_std_v
                cr_descriptives_individual.loc[new_row_index, "cr_std_a"] = cr_std_a
                cr_descriptives_individual.loc[
                    new_row_index, "cr_std_dist"
                ] = cr_std_dist
                cr_descriptives_individual.loc[
                    new_row_index, "cr_std_angle"
                ] = cr_std_angle
                cr_descriptives_individual.loc[new_row_index, "cr_skew_v"] = cr_skew_v
                cr_descriptives_individual.loc[new_row_index, "cr_skew_a"] = cr_skew_a
                cr_descriptives_individual.loc[
                    new_row_index, "cr_skew_dist"
                ] = cr_skew_dist
                cr_descriptives_individual.loc[
                    new_row_index, "cr_skew_angle"
                ] = cr_skew_angle
                cr_descriptives_individual.loc[
                    new_row_index, "cr_kurtosis_v"
                ] = cr_kurtosis_v
                cr_descriptives_individual.loc[
                    new_row_index, "cr_kurtosis_a"
                ] = cr_kurtosis_a
                cr_descriptives_individual.loc[
                    new_row_index, "cr_kurtosis_dist"
                ] = cr_kurtosis_dist
                cr_descriptives_individual.loc[
                    new_row_index, "cr_kurtosis_angle"
                ] = cr_kurtosis_angle
                cr_descriptives_individual.loc[new_row_index, "cr_auc_v"] = cr_auc_v
                cr_descriptives_individual.loc[new_row_index, "cr_auc_a"] = cr_auc_a
                cr_descriptives_individual.loc[
                    new_row_index, "cr_auc_dist"
                ] = cr_auc_dist
                cr_descriptives_individual.loc[
                    new_row_index, "cr_auc_angle"
                ] = cr_auc_angle

    # delete all rows with nan values for mean
    nan_mean_rows = np.isnan(cr_descriptives_individual["cr_mean_v"])
    cr_descriptives_individual = cr_descriptives_individual[~nan_mean_rows]

    return cr_descriptives_individual


# step 2b
# function to calculate descriptive statistics for each quadrant separately averaged over participants
def descriptives(
    data: pd.DataFrame, quadrants: list[str], rating_methods: list[str]
) -> pd.DataFrame:
    # create new dataframe with summary descriptives
    cr_descriptives = pd.DataFrame(
        {
            "rating_method": [],  # rating method
            "quadrant": [],  # quadrant
            "cr_mean_v": [],  # mean valence
            "cr_mean_a": [],  # mean arousal
            "cr_mean_dist": [],  # mean distance
            "cr_mean_angle": [],  # mean angle
            "cr_std_v": [],  # standard deviation valence
            "cr_std_a": [],  # standard deviation arousal
            "cr_std_dist": [],  # standard deviation distance
            "cr_std_angle": [],  # standard deviation angle
            "cr_skew_v": [],  # skewness valence
            "cr_skew_a": [],  # skewness arousal
            "cr_skew_dist": [],  # skewness distance
            "cr_skew_angle": [],  # skewness angle
            "cr_kurtosis_v": [],  # kurtosis valence
            "cr_kurtosis_a": [],  # kurtosis arousal
            "cr_kurtosis_dist": [],  # kurtosis distance
            "cr_kurtosis_angle": [],  # kurtosis angle
            "cr_auc_v": [],  # area under curve valence
            "cr_auc_a": [],  # area under curve arousal
            "cr_auc_dist": [],  # area under curve distance
            "cr_auc_angle": [],  # area under curve angle
        }
    )

    for rating_index, rating_method in enumerate(rating_methods):
        # get data for one rating method
        if len(rating_methods) > 3:
            if rating_method == "mean":
                method_data = data
            else:
                method_data = data[data["video_id"] == rating_method]
        else:
            method_data = data[data["rating_method"] == rating_method]

        # loop over quadrants
        for row_index, quadrant in enumerate(quadrants):
            # mean
            cr_mean_v = method_data["cr_v"][
                method_data["quadrant"].str.contains(quadrant)
            ].mean(skipna=True)
            cr_mean_a = method_data["cr_a"][
                method_data["quadrant"].str.contains(quadrant)
            ].mean(skipna=True)
            cr_mean_dist = float(
                np.nanmean(
                    method_data["cr_dist"][
                        method_data["quadrant"].str.contains(quadrant)
                    ]
                )
            )
            cr_mean_angle = float(
                np.nanmean(
                    method_data["cr_angle"][
                        method_data["quadrant"].str.contains(quadrant)
                    ]
                )
            )

            # std
            cr_std_v = method_data["cr_v"][
                method_data["quadrant"].str.contains(quadrant)
            ].std(skipna=True)
            cr_std_a = method_data["cr_a"][
                method_data["quadrant"].str.contains(quadrant)
            ].std(skipna=True)
            cr_std_dist = float(
                np.nanstd(
                    method_data["cr_dist"][
                        method_data["quadrant"].str.contains(quadrant)
                    ]
                )
            )
            cr_std_angle = float(
                np.nanstd(
                    method_data["cr_angle"][
                        method_data["quadrant"].str.contains(quadrant)
                    ]
                )
            )

            # skewness
            cr_skew_v = method_data["cr_v"][
                method_data["quadrant"].str.contains(quadrant)
            ].skew(skipna=True)
            cr_skew_a = method_data["cr_v"][
                method_data["quadrant"].str.contains(quadrant)
            ].skew(skipna=True)
            cr_skew_dist = stats.skew(
                method_data["cr_dist"][method_data["quadrant"].str.contains(quadrant)],
                nan_policy="omit",
            )
            cr_skew_angle = stats.skew(
                method_data["cr_angle"][method_data["quadrant"].str.contains(quadrant)],
                nan_policy="omit",
            )

            # kutosis
            cr_kurtosis_v = method_data["cr_v"][
                method_data["quadrant"].str.contains(quadrant)
            ].kurtosis(skipna=True)
            cr_kurtosis_a = method_data["cr_v"][
                method_data["quadrant"].str.contains(quadrant)
            ].kurtosis(skipna=True)
            cr_kurtosis_dist = stats.kurtosis(
                method_data["cr_dist"][method_data["quadrant"].str.contains(quadrant)],
                nan_policy="omit",
            )
            cr_kurtosis_angle = stats.kurtosis(
                method_data["cr_angle"][method_data["quadrant"].str.contains(quadrant)],
                nan_policy="omit",
            )

            # area under curve - use trapz or sum? ~ same results
            cr_auc_v = np.trapz(
                method_data["cr_v"][
                    method_data["quadrant"].str.contains(quadrant)
                ].dropna(),
                dx=1,
            )
            cr_auc_a = np.trapz(
                method_data["cr_v"][
                    method_data["quadrant"].str.contains(quadrant)
                ].dropna(),
                dx=1,
            )
            cr_auc_dist = np.trapz(
                method_data["cr_dist"][method_data["quadrant"].str.contains(quadrant)][
                    ~np.isnan(
                        method_data["cr_dist"][
                            method_data["quadrant"].str.contains(quadrant)
                        ]
                    )
                ],
                dx=1,
            )
            cr_auc_angle = np.trapz(
                method_data["cr_angle"][method_data["quadrant"].str.contains(quadrant)][
                    ~np.isnan(
                        method_data["cr_angle"][
                            method_data["quadrant"].str.contains(quadrant)
                        ]
                    )
                ],
                dx=1,
            )

            # add descriptive stats of one quadrant to cr_descriptives dataframe as a new row
            new_row_index = rating_index * len(quadrants) + row_index
            cr_descriptives.loc[new_row_index, "rating_method"] = rating_method
            cr_descriptives.loc[new_row_index, "quadrant"] = quadrant
            cr_descriptives.loc[new_row_index, "cr_mean_v"] = cr_mean_v
            cr_descriptives.loc[new_row_index, "cr_mean_a"] = cr_mean_a
            cr_descriptives.loc[new_row_index, "cr_mean_dist"] = cr_mean_dist
            cr_descriptives.loc[new_row_index, "cr_mean_angle"] = cr_mean_angle
            cr_descriptives.loc[new_row_index, "cr_std_v"] = cr_std_v
            cr_descriptives.loc[new_row_index, "cr_std_a"] = cr_std_a
            cr_descriptives.loc[new_row_index, "cr_std_dist"] = cr_std_dist
            cr_descriptives.loc[new_row_index, "cr_std_angle"] = cr_std_angle
            cr_descriptives.loc[new_row_index, "cr_skew_v"] = cr_skew_v
            cr_descriptives.loc[new_row_index, "cr_skew_a"] = cr_skew_a
            cr_descriptives.loc[new_row_index, "cr_skew_dist"] = cr_skew_dist
            cr_descriptives.loc[new_row_index, "cr_skew_angle"] = cr_skew_angle
            cr_descriptives.loc[new_row_index, "cr_kurtosis_v"] = cr_kurtosis_v
            cr_descriptives.loc[new_row_index, "cr_kurtosis_a"] = cr_kurtosis_a
            cr_descriptives.loc[new_row_index, "cr_kurtosis_dist"] = cr_kurtosis_dist
            cr_descriptives.loc[new_row_index, "cr_kurtosis_angle"] = cr_kurtosis_angle
            cr_descriptives.loc[new_row_index, "cr_auc_v"] = cr_auc_v
            cr_descriptives.loc[new_row_index, "cr_auc_a"] = cr_auc_a
            cr_descriptives.loc[new_row_index, "cr_auc_dist"] = cr_auc_dist
            cr_descriptives.loc[new_row_index, "cr_auc_angle"] = cr_auc_angle

    # delete all rows with nan values for mean
    nan_mean_rows = np.isnan(cr_descriptives["cr_mean_v"])
    cr_descriptives = cr_descriptives[~nan_mean_rows]

    return cr_descriptives


# step 2c
# function to calculate descriptive statistics averaged over quadrants for each participant
def descriptives_avgvideos_individual(
    data: pd.DataFrame, subjects: list[str], rating_methods: list[str]
) -> pd.DataFrame:
    # create new dataframe with summary descriptives
    cr_descriptives_avgvideos_individual = pd.DataFrame(
        {
            "sj_id": [],  # subject ID
            "rating_method": [],  # rating method
            "cr_mean_v": [],  # mean valence
            "cr_mean_a": [],  # mean arousal
            "cr_mean_dist": [],  # mean distance
            "cr_mean_angle": [],  # mean angle
            "cr_std_v": [],  # standard deviation valence
            "cr_std_a": [],  # standard deviation arousal
            "cr_std_dist": [],  # standard deviation distance
            "cr_std_angle": [],  # standard deviation angle
            "cr_skew_v": [],  # skewness valence
            "cr_skew_a": [],  # skewness arousal
            "cr_skew_dist": [],  # skewness distance
            "cr_skew_angle": [],  # skewness angle
            "cr_kurtosis_v": [],  # kurtosis valence
            "cr_kurtosis_a": [],  # kurtosis arousal
            "cr_kurtosis_dist": [],  # kurtosis distance
            "cr_kurtosis_angle": [],  # kurtosis angle
            "cr_auc_v": [],  # area under curve valence
            "cr_auc_a": [],  # area under curve arousal
            "cr_auc_dist": [],  # area under curve distance
            "cr_auc_angle": [],  # area under curve angle
        }
    )

    if dataset == "ceap":
        # convert rating_method videos in CEAP dataset to V1 / V2 to make it easier to plot
        for row in data["video_id"].index:
            if data["video_id"][row] == "V1":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V2":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V3":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V4":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V5":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V6":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V7":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V8":
                data["video_id"][row] = "V2"
            else:
                continue

    # loop over participants
    for subject_index, subject in enumerate(subjects):
        for rating_index, rating_method in enumerate(rating_methods):
            # get data for one subject for one rating method
            if dataset == "ceap":
                if rating_method == "mean":
                    subject_data = data[data["sj_id"] == subject]
                else:
                    subject_data = data[data["sj_id"] == subject][
                        data["video_id"] == rating_method
                    ]
            else:
                subject_data = data[data["sj_id"] == subject][
                    data["rating_method"] == rating_method
                ]

            # mean
            cr_mean_v = subject_data["cr_v"].mean(skipna=True)
            cr_mean_a = subject_data["cr_a"].mean(skipna=True)
            cr_mean_dist = float(np.nanmean(subject_data["cr_dist"]))
            cr_mean_angle = float(np.nanmean(subject_data["cr_angle"]))

            # std
            cr_std_v = subject_data["cr_v"].std(skipna=True)
            cr_std_a = subject_data["cr_a"].std(skipna=True)
            cr_std_dist = float(np.nanstd(subject_data["cr_dist"]))
            cr_std_angle = float(np.nanstd(subject_data["cr_angle"]))

            # skewness
            cr_skew_v = subject_data["cr_v"].skew(skipna=True)
            cr_skew_a = subject_data["cr_v"].skew(skipna=True)
            cr_skew_dist = stats.skew(subject_data["cr_dist"], nan_policy="omit")
            cr_skew_angle = stats.skew(subject_data["cr_angle"], nan_policy="omit")

            # kutosis
            cr_kurtosis_v = subject_data["cr_v"].kurtosis(skipna=True)
            cr_kurtosis_a = subject_data["cr_v"].kurtosis(skipna=True)
            cr_kurtosis_dist = stats.kurtosis(
                subject_data["cr_dist"], nan_policy="omit"
            )
            cr_kurtosis_angle = stats.kurtosis(
                subject_data["cr_angle"], nan_policy="omit"
            )

            # area under curve - use trapz or sum? ~ same results
            cr_auc_v = np.trapz(subject_data["cr_v"].dropna(), dx=1)
            cr_auc_a = np.trapz(subject_data["cr_v"].dropna(), dx=1)
            cr_auc_dist = np.trapz(
                subject_data["cr_dist"][~np.isnan(subject_data["cr_dist"])], dx=1
            )
            cr_auc_angle = np.trapz(
                subject_data["cr_angle"][~np.isnan(subject_data["cr_angle"])], dx=1
            )

            # add descriptive stats to cr_descriptives_individual dataframe as a new row
            new_row_index = subject_index * len(rating_methods) + rating_index
            cr_descriptives_avgvideos_individual.loc[new_row_index, "sj_id"] = subject
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "rating_method"
            ] = rating_method
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_mean_v"
            ] = cr_mean_v
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_mean_a"
            ] = cr_mean_a
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_mean_dist"
            ] = cr_mean_dist
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_mean_angle"
            ] = cr_mean_angle
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_std_v"
            ] = cr_std_v
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_std_a"
            ] = cr_std_a
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_std_dist"
            ] = cr_std_dist
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_std_angle"
            ] = cr_std_angle
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_skew_v"
            ] = cr_skew_v
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_skew_a"
            ] = cr_skew_a
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_skew_dist"
            ] = cr_skew_dist
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_skew_angle"
            ] = cr_skew_angle
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_kurtosis_v"
            ] = cr_kurtosis_v
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_kurtosis_a"
            ] = cr_kurtosis_a
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_kurtosis_dist"
            ] = cr_kurtosis_dist
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_kurtosis_angle"
            ] = cr_kurtosis_angle
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_auc_v"
            ] = cr_auc_v
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_auc_a"
            ] = cr_auc_a
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_auc_dist"
            ] = cr_auc_dist
            cr_descriptives_avgvideos_individual.loc[
                new_row_index, "cr_auc_angle"
            ] = cr_auc_angle

    # delete all rows with nan values for mean
    nan_mean_rows = np.isnan(cr_descriptives_avgvideos_individual["cr_mean_v"])
    cr_descriptives_avgvideos_individual = cr_descriptives_avgvideos_individual[
        ~nan_mean_rows
    ]

    return cr_descriptives_avgvideos_individual


# step 2d
# function to calculate descriptive statistics averaged over quadrants averaged over participants
def descriptives_avgvideos(
    data: pd.DataFrame, rating_methods: list[str]
) -> pd.DataFrame:
    # create new dataframe with summary descriptives
    cr_descriptives_avgvideos = pd.DataFrame(
        {
            "rating_method": [],  # rating method
            "cr_mean_v": [],  # mean valence
            "cr_mean_a": [],  # mean arousal
            "cr_mean_dist": [],  # mean distance
            "cr_mean_angle": [],  # mean angle
            "cr_std_v": [],  # standard deviation valence
            "cr_std_a": [],  # standard deviation arousal
            "cr_std_dist": [],  # standard deviation distance
            "cr_std_angle": [],  # standard deviation angle
            "cr_skew_v": [],  # skewness valence
            "cr_skew_a": [],  # skewness arousal
            "cr_skew_dist": [],  # skewness distance
            "cr_skew_angle": [],  # skewness angle
            "cr_kurtosis_v": [],  # kurtosis valence
            "cr_kurtosis_a": [],  # kurtosis arousal
            "cr_kurtosis_dist": [],  # kurtosis distance
            "cr_kurtosis_angle": [],  # kurtosis angle
            "cr_auc_v": [],  # area under curve valence
            "cr_auc_a": [],  # area under curve arousal
            "cr_auc_dist": [],  # area under curve distance
            "cr_auc_angle": [],  # area under curve angle
        }
    )

    if dataset == "ceap":
        # convert rating_method videos in CEAP dataset to V1 / V2 to make it easier to plot
        for row in data["video_id"].index:
            if data["video_id"][row] == "V1":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V2":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V3":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V4":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V5":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V6":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V7":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V8":
                data["video_id"][row] = "V2"
            else:
                continue

    for rating_index, rating_method in enumerate(rating_methods):
        # get data for one rating method
        if dataset == "ceap":
            if rating_method == "mean":
                method_data = data
            else:
                method_data = data[data["video_id"] == rating_method]
        else:
            method_data = data[data["rating_method"] == rating_method]

        # mean
        cr_mean_v = method_data["cr_v"].mean(skipna=True)
        cr_mean_a = method_data["cr_a"].mean(skipna=True)
        cr_mean_dist = float(np.nanmean(method_data["cr_dist"]))
        cr_mean_angle = float(np.nanmean(method_data["cr_angle"]))

        # std
        cr_std_v = method_data["cr_v"].std(skipna=True)
        cr_std_a = method_data["cr_a"].std(skipna=True)
        cr_std_dist = float(np.nanstd(method_data["cr_dist"]))
        cr_std_angle = float(np.nanstd(method_data["cr_angle"]))

        # skewness
        cr_skew_v = method_data["cr_v"].skew(skipna=True)
        cr_skew_a = method_data["cr_v"].skew(skipna=True)
        cr_skew_dist = stats.skew(method_data["cr_dist"], nan_policy="omit")
        cr_skew_angle = stats.skew(method_data["cr_angle"], nan_policy="omit")

        # kutosis
        cr_kurtosis_v = method_data["cr_v"].kurtosis(skipna=True)
        cr_kurtosis_a = method_data["cr_v"].kurtosis(skipna=True)
        cr_kurtosis_dist = stats.kurtosis(method_data["cr_dist"], nan_policy="omit")
        cr_kurtosis_angle = stats.kurtosis(method_data["cr_angle"], nan_policy="omit")

        # area under curve - use trapz or sum? ~ same results
        cr_auc_v = np.trapz(method_data["cr_v"].dropna(), dx=1)
        cr_auc_a = np.trapz(method_data["cr_v"].dropna(), dx=1)
        cr_auc_dist = np.trapz(
            method_data["cr_dist"][~np.isnan(method_data["cr_dist"])], dx=1
        )
        cr_auc_angle = np.trapz(
            method_data["cr_angle"][~np.isnan(method_data["cr_angle"])], dx=1
        )

        # add descriptive stats to cr_descriptives_individual dataframe as a new row
        cr_descriptives_avgvideos.loc[rating_index, "rating_method"] = rating_method
        cr_descriptives_avgvideos.loc[rating_index, "cr_mean_v"] = cr_mean_v
        cr_descriptives_avgvideos.loc[rating_index, "cr_mean_a"] = cr_mean_a
        cr_descriptives_avgvideos.loc[rating_index, "cr_mean_dist"] = cr_mean_dist
        cr_descriptives_avgvideos.loc[rating_index, "cr_mean_angle"] = cr_mean_angle
        cr_descriptives_avgvideos.loc[rating_index, "cr_std_v"] = cr_std_v
        cr_descriptives_avgvideos.loc[rating_index, "cr_std_a"] = cr_std_a
        cr_descriptives_avgvideos.loc[rating_index, "cr_std_dist"] = cr_std_dist
        cr_descriptives_avgvideos.loc[rating_index, "cr_std_angle"] = cr_std_angle
        cr_descriptives_avgvideos.loc[rating_index, "cr_skew_v"] = cr_skew_v
        cr_descriptives_avgvideos.loc[rating_index, "cr_skew_a"] = cr_skew_a
        cr_descriptives_avgvideos.loc[rating_index, "cr_skew_dist"] = cr_skew_dist
        cr_descriptives_avgvideos.loc[rating_index, "cr_skew_angle"] = cr_skew_angle
        cr_descriptives_avgvideos.loc[rating_index, "cr_kurtosis_v"] = cr_kurtosis_v
        cr_descriptives_avgvideos.loc[rating_index, "cr_kurtosis_a"] = cr_kurtosis_a
        cr_descriptives_avgvideos.loc[
            rating_index, "cr_kurtosis_dist"
        ] = cr_kurtosis_dist
        cr_descriptives_avgvideos.loc[
            rating_index, "cr_kurtosis_angle"
        ] = cr_kurtosis_angle
        cr_descriptives_avgvideos.loc[rating_index, "cr_auc_v"] = cr_auc_v
        cr_descriptives_avgvideos.loc[rating_index, "cr_auc_a"] = cr_auc_a
        cr_descriptives_avgvideos.loc[rating_index, "cr_auc_dist"] = cr_auc_dist
        cr_descriptives_avgvideos.loc[rating_index, "cr_auc_angle"] = cr_auc_angle

    # delete all rows with nan values for mean
    nan_mean_rows = np.isnan(cr_descriptives_avgvideos["cr_mean_v"])
    cr_descriptives_avgvideos = cr_descriptives_avgvideos[~nan_mean_rows]

    return cr_descriptives_avgvideos


# step 2e
# function to calculate average valence and arousal values for all timepoints, quadrants and rating methods averaged over participants
def all_timepoints_average(
    data: pd.DataFrame, quadrants: list[str], rating_methods: list[str]
) -> pd.DataFrame:
    # create new dataframe with values
    cr_all_timepoints_average = pd.DataFrame(
        {
            "rating_method": [],  # rating method
            "quadrant": [],  # quadrant
            "cr_v": [],  # mean valence across participants for the timepoint
            "cr_a": [],  # mean arousal across participants for the timepoint
            "cr_dist": [],  # mean distance across participants for the timepoint
            "cr_angle": [],  # mean angle across participants for the timepoint
            "cr_time": [],  # timepoint
        }
    )

    if dataset == "ceap":
        # convert rating_method videos in CEAP dataset to V1 / V2 to make it easier to plot
        for row in data["video_id"].index:
            if data["video_id"][row] == "V1":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V2":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V3":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V4":
                data["video_id"][row] = "V1"
            elif data["video_id"][row] == "V5":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V6":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V7":
                data["video_id"][row] = "V2"
            elif data["video_id"][row] == "V8":
                data["video_id"][row] = "V2"
            else:
                continue

    # count all unique timepoints (5s to 60s)
    timepoints = data["cr_time"].unique()

    # loop over quadrants
    for counter, quadrant in enumerate(quadrants):
        # select only data from one quadrant
        quadrant_data = data[data["quadrant"].str.contains(quadrant)]

        # loop over rating methods
        for rating_index, rating_method in enumerate(rating_methods):
            # get data for one rating method
            if dataset == "ceap":
                if rating_method == "mean":
                    method_data = quadrant_data
                else:
                    method_data = quadrant_data[
                        quadrant_data["video_id"] == rating_method
                    ]
            else:
                method_data = quadrant_data[quadrant_data["rating_method"] == rating_method]

            # loop over all unique timepoints
            for index, timepoint in enumerate(timepoints):
                # mean valence across participants for the timepoint
                cr_v = method_data["cr_v"][method_data["cr_time"] == timepoint].mean(
                    skipna=True
                )
                # mean arousal across participants for the timepoint
                cr_a = method_data["cr_a"][method_data["cr_time"] == timepoint].mean(
                    skipna=True
                )
                # mean distance across participants for the timepoint
                cr_dist = float(
                    np.nanmean(
                        method_data["cr_dist"][method_data["cr_time"] == timepoint]
                    )
                )
                # mean angle across participants for the timepoint
                cr_angle = float(
                    np.nanmean(
                        method_data["cr_angle"][method_data["cr_time"] == timepoint]
                    )
                )

                # add values for one timepoint to cr_all_timepoints_average dataframe as a new row
                new_row_index = (
                    rating_index * len(quadrants) * len(timepoints)
                    + counter * len(timepoints)
                    + index
                )

                cr_all_timepoints_average.loc[
                    new_row_index, "rating_method"
                ] = rating_method
                cr_all_timepoints_average.loc[new_row_index, "quadrant"] = quadrant
                cr_all_timepoints_average.loc[new_row_index, "cr_v"] = cr_v
                cr_all_timepoints_average.loc[new_row_index, "cr_a"] = cr_a
                cr_all_timepoints_average.loc[new_row_index, "cr_dist"] = cr_dist
                cr_all_timepoints_average.loc[new_row_index, "cr_angle"] = cr_angle
                cr_all_timepoints_average.loc[new_row_index, "cr_time"] = timepoint

    return cr_all_timepoints_average


# ----------------------- ANALYSIS ----------------------------
for dataset in datasets:
    # read in part of the file corresponding to the dataset
    if dataset == "affectivevr":
        # read in only rows where rating_method is not "Joystick" as those are the ones corresponding to the AffectiveVR dataset
        df = pd.read_csv(preprocessed_file)
        df = df[~df["rating_method"].str.contains("Joystick")]
        # get list of subjects for that dataset
        subjects = df["sj_id"].unique().tolist()
        # get list of rating methods for that dataset
        rating_methods = df["rating_method"].unique().tolist()
    else:
        # read in only the rows where rating_method == "Joystick" as those are the ones corresponding to the CEAP dataset
        df = pd.read_csv(preprocessed_file)
        df = df[df["rating_method"].str.contains("Joystick")]
        # get list of subjects for that dataset
        subjects = df["sj_id"].unique().tolist()
        # get list of rating methods for that dataset and add "mean" to the list
        rating_methods = df["video_id"].unique().tolist()
        rating_methods.append("mean")

    for step in steps:
        # ------------------------ STEP 2A -----------------------------
        if step == "2a":
            # calculate descriptives using the descriptives_individual function
            cr_descriptives_individual = descriptives_individual(
                df, quadrants, subjects, rating_methods
            )
            if dataset == "ceap":
                # convert rating_method videos in CEAP dataset to V1 / V2 to make it easier to plot
                for row in cr_descriptives_individual["rating_method"].index:
                    if cr_descriptives_individual["rating_method"][row] == "V1":
                        cr_descriptives_individual["rating_method"][row] = "V1"
                    elif cr_descriptives_individual["rating_method"][row] == "V2":
                        cr_descriptives_individual["rating_method"][row] = "V1"
                    elif cr_descriptives_individual["rating_method"][row] == "V3":
                        cr_descriptives_individual["rating_method"][row] = "V1"
                    elif cr_descriptives_individual["rating_method"][row] == "V4":
                        cr_descriptives_individual["rating_method"][row] = "V1"
                    elif cr_descriptives_individual["rating_method"][row] == "V5":
                        cr_descriptives_individual["rating_method"][row] = "V2"
                    elif cr_descriptives_individual["rating_method"][row] == "V6":
                        cr_descriptives_individual["rating_method"][row] = "V2"
                    elif cr_descriptives_individual["rating_method"][row] == "V7":
                        cr_descriptives_individual["rating_method"][row] = "V2"
                    elif cr_descriptives_individual["rating_method"][row] == "V8":
                        cr_descriptives_individual["rating_method"][row] = "V2"
                    else:
                        continue

            # save as one data file
            cr_descriptives_individual.to_csv(
                os.path.join(
                    results_path, "cr_{}_descriptive_individual.csv".format(dataset)
                ),
                index=False,
            )

        # ------------------------ STEP 2B -----------------------------
        elif step == "2b":
            # calculate descriptives using the descriptives function
            cr_descriptives = descriptives(df, quadrants, rating_methods)

            if dataset == "ceap":
                # convert rating_method videos in CEAP dataset to V1 / V2 to make it easier to plot
                for row in cr_descriptives["rating_method"].index:
                    if cr_descriptives["rating_method"][row] == "V1":
                        cr_descriptives["rating_method"][row] = "V1"
                    elif cr_descriptives["rating_method"][row] == "V2":
                        cr_descriptives["rating_method"][row] = "V1"
                    elif cr_descriptives["rating_method"][row] == "V3":
                        cr_descriptives["rating_method"][row] = "V1"
                    elif cr_descriptives["rating_method"][row] == "V4":
                        cr_descriptives["rating_method"][row] = "V1"
                    elif cr_descriptives["rating_method"][row] == "V5":
                        cr_descriptives["rating_method"][row] = "V2"
                    elif cr_descriptives["rating_method"][row] == "V6":
                        cr_descriptives["rating_method"][row] = "V2"
                    elif cr_descriptives["rating_method"][row] == "V7":
                        cr_descriptives["rating_method"][row] = "V2"
                    elif cr_descriptives["rating_method"][row] == "V8":
                        cr_descriptives["rating_method"][row] = "V2"
                    else:
                        continue

            # save as one data file
            cr_descriptives.to_csv(
                os.path.join(results_path, "cr_{}_descriptive.csv".format(dataset)),
                index=False,
            )

        # ------------------------ STEP 2C -----------------------------
        elif step == "2c":
            if dataset == "ceap":
                rating_methods = ["V1", "V2", "mean"]

            # calculate descriptives using the descriptives_avgvideos_individual function
            cr_descriptives_avgvideos_individual = descriptives_avgvideos_individual(
                df, subjects, rating_methods
            )
            # save as one data file
            cr_descriptives_avgvideos_individual.to_csv(
                os.path.join(
                    results_path,
                    "cr_{}_descriptive_avgvideos_individual.csv".format(dataset),
                ),
                index=False,
            )

        # ------------------------ STEP 2D -----------------------------
        elif step == "2d":
            if dataset == "ceap":
                rating_methods = ["V1", "V2", "mean"]

            # calculate descriptives using the descriptives_avgvideos function
            cr_descriptives_avgvideos = descriptives_avgvideos(df, rating_methods)
            # save as one data file
            cr_descriptives_avgvideos.to_csv(
                os.path.join(
                    results_path, "cr_{}_descriptive_avgvideos.csv".format(dataset)
                ),
                index=False,
            )

        # ------------------------ STEP 2E -----------------------------
        elif step == "2e":
            if dataset == "ceap":
                rating_methods = ["V1", "V2", "mean"]

            # calculate values using the all_timepoints_average function
            cr_all_timepoints_average = all_timepoints_average(
                df, quadrants, rating_methods
            )
            # save as one data file
            cr_all_timepoints_average.to_csv(
                os.path.join(
                    results_path, "cr_{}_all_timepoints_average.csv".format(dataset)
                ),
                index=False,
            )

        else:
            print("Step not found")
