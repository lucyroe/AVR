########################################################################################################################
# Script to plot preprocessed CEAP data against preprocessed AffecitveVR data
#
# Step 3a:      Create descriptive plots for CEAP data against AffectiveVR data
#               Inputs:       csv file created by 01_preprocessing.py with preprocessed data from CEAP and AffectiveVR dataset
#               Outputs:      2 x 2 plot (four quadrants) for each dependent variable (Arousal, Valence, Distance, Angle)
#
# Step 3b:      Create descriptive plot for mean of CEAP data against mean of AffectiveVR data
#               Inputs:       two csv files created by 02_summary_stats.py with descriptive statistics averaged across timepoints and participants
#                             cr_affectivevr_descriptive.csv and cr_ceap_descriptive.csv
#               Output:       plot with coordinate system and mean arousal + valence values for all quadrants
#
# Step 3c:      Create descriptive plots for CEAP / AffectiveVR data for all videos separately averaged across participants
#               Inputs:       csv files created by 02_summary_stats.py with averaged data across participants (cr_affectivevr_all.csv, cr_ceap_all.csv)
#               Outputs:      2 plots with coordinate system and averaged arousal + valence values for each video, one for CEAP and one for AffectiveVR
#
# Step 3d:      Create descriptive plots for CEAP / AffectiveVR data for each participant's mean values
#               Inputs:       csv files created by 02_summary_stats.py with averaged data for individual participants (cr_affectivevr_descriptive_individual.csv, cr_ceap_descriptive_individual.csv)
#               Outputs:      2 plots with coordinate system and averaged arousal + valence values for each participant, one for CEAP and one for AffectiveVR
#
# Step 3e:      Create violin plot to compare summary statistics of CEAP and AffectiveVR datasets
#               Inputs:       csv files created by 02_summary_stats.py with averaged data for individual participants (cr_affectivevr_descriptive_individual.csv, cr_ceap_descriptive_individual.csv)
#               Outputs:      violin plots with summary statistics (mean, std) for all dimensions (valence, arousal, distance, angle))
#
# Author:       Lucy Roellecke (lucy.roellecke[at]tuta.com)
# Last version: 08.11.2023
########################################################################################################################

# -------------------- IMPORT PACKAGES ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import os
import seaborn as sns

# ------------------------- SETUP ------------------------------
# change the data_path to the path where you saved the preprocessed CEAP data
data_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CEAP/data/CEAP-360VR/3_AnnotationData/"
descriptive_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CEAP/results/descriptives/"
# change the plot_path to the path where you want to save the plots
plot_path = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/CEAP/results/"

# plotting steps to perform
steps = ["3e"]
# "3a", "3b", "3c", "3d"

# datasets to plot
datasets = ["affectivevr", "ceap"]

# quadrants
quadrants = ["HP", "LP", "LN", "HN"]

# statistics
statistics = ["mean", "std"]
# "skew", "kurtosis"

# dimensions
dimensions = ["valence", "arousal", "distance", "angle"]

# create color scheme for rating methods for mean arousal and valence plots
color_scheme_mean = [
    "#56b4e9",
    "#0072b2",
    "#009e73",  # blue tones for AffectiveVR
    "#e69f00",
    "#cc79a7",
    "#d55e00",
]  # pink tones for CEAP
# checked color-blind safety on: https://www.color-blindness.com/coblis-color-blindness-simulator/

# create color scheme for rating methods / videos for arousal and valence plots averaged across participants
color_scheme_avg = [
    "#f2c200",  # yellow tones for HP
    "#f4a504",
    "#ee8100",
    "#89b300",  # green tones for LP
    "#018756",
    "#2b3c27",
    "#9ec8ed",  # blue tones for LN
    "#5b4c90",
    "#0366a3",
    "#dc5621",  # red tones for HN
    "#be0233",
    "#b3456d",
]
# TODO: check color-blind safety

quadrant_color_mapping = {
    "HP": color_scheme_avg[0:3],
    "LP": color_scheme_avg[3:6],
    "LN": color_scheme_avg[6:9],
    "HN": color_scheme_avg[9:12],
}

rating_method_color_mapping = {
    "Grid": 0,
    "Flubber": 1,
    "Proprioceptive": 2,
    "Joystick V1": 0,
    "Joystick V2": 1,
    "Joystick mean": 2,
}

# create marker scheme for rating methods
marker_scheme = [
    "^",
    "^",
    "^",  # triangles for AffectiveVR
    "o",
    "o",
    "o",
]  # circles for CEAP

for step in steps:
    # ------------------------ STEP 3A -----------------------------
    if step == "3a":
        # open the "all_participants_both_studies.csv" file
        data = pd.read_csv(data_path + "all_participants_ceap_affectivevr.csv")

        # set dependent variables
        dependent_variables = ["cr_a", "cr_v", "cr_dist", "cr_angle"]
        dependent_variables_names = ["Arousal", "Valence", "Distance", "Angle"]

        # quadrant order to mirror affective grid
        quadrant_order = ["HN", "HP", "LN", "LP"]

        # video quadrant mapping for joystick
        video_quadrant_mapping = {
            "V1": "HP",
            "V2": "LP",
            "V3": "HN",
            "V4": "LN",
            "V5": "HP",
            "V6": "LP",
            "V7": "HN",
            "V8": "LN",
        }

        # creates a 2 x 2 plot for each dependent variable
        for i, dependent_variable in enumerate(dependent_variables):
            # create a new figure with a 2 x 2 subplot layout
            figure, axes = plt.subplots(2, 2, figsize=(10, 8))

            # add main title
            figure.suptitle(
                f"{dependent_variables_names[i]}", fontsize=16, fontweight="bold"
            )

            # loop over each quadrant
            for j, quadrant in enumerate(quadrant_order):
                # filter dataframe for the current quadrant value
                quadrant_dataframe = data[data["quadrant"] == quadrant]

                # determine the subplot position based on the desired order
                if j == 0:
                    row = 0
                    column = 0
                elif j == 1:
                    row = 0
                    column = 1
                elif j == 2:
                    row = 1
                    column = 0
                elif j == 3:
                    row = 1
                    column = 1
                else:
                    print("Error: Quadrant value not found.")
                    continue

                # create respective subplot per iteration (line plot for the mean ratings over time)
                axis = axes[row, column]
                axis.set_ylim(
                    [data[dependent_variable].min(), data[dependent_variable].max()]
                )  # set y-axis limits
                axis.set_xlim([0, 60])  # set x-axis limits

                counter = 0
                # loop over each rating method
                for rating_method in quadrant_dataframe["rating_method"].unique():
                    rating_dataframe = quadrant_dataframe[
                        quadrant_dataframe["rating_method"] == rating_method
                    ]

                    if rating_method == "Joystick":
                        # iterate over the video values for the current quadrant
                        for video in video_quadrant_mapping.keys():
                            if video_quadrant_mapping[video] == quadrant:
                                video_mean = (
                                    rating_dataframe[
                                        rating_dataframe["video_id"] == video
                                    ][dependent_variable]
                                    .groupby(rating_dataframe["cr_time"])
                                    .mean()
                                )
                                # plot values for rating_method over time
                                axis.plot(
                                    video_mean.index,
                                    video_mean.values,
                                    label=f"Joystick {video}",
                                    linestyle="-",
                                    c=color_scheme_mean[counter],
                                )
                                counter += 1

                        # calculate and plot the mean of all joystick videos in the quadrant
                        joystick_mean = (
                            rating_dataframe[dependent_variable]
                            .groupby(rating_dataframe["cr_time"])
                            .mean()
                        )
                        axis.plot(
                            joystick_mean.index,
                            joystick_mean.values,
                            label="Joystick Mean",
                            linestyle="--",
                            c=color_scheme_mean[counter],
                        )

                    else:
                        mean = (
                            rating_dataframe[dependent_variable]
                            .groupby(rating_dataframe["cr_time"])
                            .mean()
                        )
                        # plot values for rating_method over time
                        axis.plot(
                            mean.index,
                            mean.values,
                            label=rating_method,
                            linestyle="-",
                            c=color_scheme_mean[counter],
                        )

                        counter += 1

                # add subplot title and labels
                axis.set_title("Mean Ratings for " + quadrant + " Stimuli")
                axis.set_xlabel("Time")
                axis.set_ylabel(dependent_variables_names[i])

            # adjust the spacing between subplots
            plt.tight_layout()

            # create a single legend for all subplots
            handles, labels = axes[0, 0].get_legend_handles_labels()
            legend = figure.legend(
                handles, labels, loc="center right", title="Rating Methods"
            )
            figure.tight_layout(rect=(0, 0, 0.80, 1))
            legend.get_title().set_fontsize(11)
            legend.get_title().set_fontweight("bold")

            # save the figure as a PDF file
            plt.savefig(plot_path + dependent_variable + ".pdf")

            # close the current figure
            plt.close(figure)

    # ------------------------ STEP 3B -----------------------------
    elif step == "3b":
        # open the descriptive statistics files
        affectivevr_data = pd.read_csv(
            descriptive_path + "cr_affectivevr_descriptive.csv"
        )
        ceap_data = pd.read_csv(descriptive_path + "cr_ceap_descriptive.csv")

        # comebine two dataframes into a list
        data = [affectivevr_data, ceap_data]

        # create a new plot with a coordinate system
        figure, axes = plt.subplots(figsize=(10, 10))

        # loop over both datasets
        counter = 0
        for index, dataframe in enumerate(data):
            # loop over rating methods and plot values
            for label, dataset in dataframe.groupby("rating_method"):
                if index == 1:  # configure label for rating method of CEAP dataset
                    label = "Joystick " + str(label)

                marker = mmarkers.MarkerStyle(marker_scheme[counter])
                axes.scatter(
                    dataset["cr_mean_v"],
                    dataset["cr_mean_a"],
                    label=label,
                    marker=marker,
                    s=40,
                    c=color_scheme_mean[counter],
                )
                axes.set_ylim(-1, 1)  # set y-axis limits
                axes.set_xlim(-1, 1)  # set x-axis limits

                axes.yaxis.set_ticks(
                    [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                )  # set y-axis ticks
                axes.xaxis.set_ticks(
                    [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                )  # set x-axis ticks

                axes.yaxis.set_ticklabels(
                    ["-1", "", "-0.5", "", "0", "", "0.5", "", "1"]
                )  # set y-axis tick labels
                axes.xaxis.set_ticklabels(
                    ["-1", "", "-0.5", "", "0", "", "0.5", "", "1"]
                )  # set x-axis tick labels

                # add a horizontal and a vertical line through the origin
                axes.axvline(x=0, color="black", linewidth=1, linestyle="-")
                axes.axhline(y=0, color="black", linewidth=1, linestyle="-")

                # draw grid lines
                axes.grid(
                    which="major", color="grey", linewidth=1, linestyle="-", alpha=0.2
                )

                counter += 1

        # add legend, labels and title
        plt.legend()
        plt.legend(loc="upper right", title="Rating Method", title_fontsize=14)
        plt.xlabel("Mean Valence")
        plt.ylabel("Mean Arousal")
        plt.title(
            "Mean Valence and Arousal Ratings across participants",
            fontsize=16,
            fontweight="bold",
        )

        # show figure
        # plt.show()

        # save the figure as a PDF file
        plt.savefig(plot_path + "mean_valence_arousal.pdf")

        # close the current figure
        plt.close(figure)

    # ------------------------ STEP 3C -----------------------------
    elif step == "3c":
        for dataset in datasets:
            # open the preprocessed data file
            preprocessed_data = pd.read_csv(
                descriptive_path + "cr_{}_all_timepoints_average.csv".format(dataset)
            )

            # create a new plot with a coordinate system
            figure, axes = plt.subplots(figsize=(10, 10))

            # group data by rating methods and quadrants
            grouped_data = preprocessed_data.groupby(["quadrant", "rating_method"])

            counter = 0

            # loop over rating methods and quadrants and plot values
            for (quadrant, rating_method), group in grouped_data:
                if (
                    dataset == "ceap"
                ):  # configure label for rating method of CEAP dataset
                    rating_method = "Joystick " + str(rating_method)

                # get color for that rating method and quadrant
                color = quadrant_color_mapping[quadrant][
                    rating_method_color_mapping[rating_method]
                ]

                # plot values for rating_method
                axes.scatter(
                    group["cr_v"], group["cr_a"], label=rating_method, c=color, s=10
                )

                axes.set_ylim(-1, 1)  # set y-axis limits
                axes.set_xlim(-1, 1)  # set x-axis limits

                axes.yaxis.set_ticks(
                    [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                )  # set y-axis ticks
                axes.xaxis.set_ticks(
                    [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                )  # set x-axis ticks

                axes.yaxis.set_ticklabels(
                    ["-1", "", "-0.5", "", "0", "", "0.5", "", "1"]
                )  # set y-axis tick labels
                axes.xaxis.set_ticklabels(
                    ["-1", "", "-0.5", "", "0", "", "0.5", "", "1"]
                )  # set x-axis tick labels

                # add a horizontal and a vertical line through the origin
                axes.axvline(x=0, color="black", linewidth=1, linestyle="-")
                axes.axhline(y=0, color="black", linewidth=1, linestyle="-")

                # draw grid lines
                axes.grid(
                    which="major", color="grey", linewidth=1, linestyle="-", alpha=0.2
                )
                counter += 1

            # add legend, labels and title
            plt.legend()
            plt.legend(loc="upper right", title="Rating Method", title_fontsize=14)
            plt.xlabel("Mean Valence")
            plt.ylabel("Mean Arousal")
            plt.title(
                "Arousal and Valence Ratings averaged across participants",
                fontsize=16,
                fontweight="bold",
            )

            # show figure
            # plt.show()

            # save the figure as a PDF file
            plt.savefig(plot_path + "valence_arousal_{}.pdf".format(dataset))

            # close the current figure
            plt.close(figure)

    # ------------------------ STEP 3D -----------------------------
    elif step == "3d":
        # open the descriptive statistics files
        affectivevr_data = pd.read_csv(
            descriptive_path + "cr_affectivevr_descriptive_individual.csv"
        )
        ceap_data = pd.read_csv(descriptive_path + "cr_ceap_descriptive_individual.csv")

        # comebine two dataframes into a list
        data = [affectivevr_data, ceap_data]

        # create a new plot with a coordinate system
        figure, axes = plt.subplots(figsize=(10, 10))

        # loop over both datasets
        counter = 0
        for index, dataframe in enumerate(data):
            # loop over rating methods and plot values
            for label, dataset in dataframe.groupby("rating_method"):
                if index == 1:  # configure label for rating method of CEAP dataset
                    label = "Joystick " + str(label)

                marker = mmarkers.MarkerStyle(marker_scheme[counter])
                axes.scatter(
                    dataset["cr_mean_v"],
                    dataset["cr_mean_a"],
                    label=label,
                    marker=marker,
                    s=40,
                    c=color_scheme_mean[counter],
                )
                axes.set_ylim(-1, 1)  # set y-axis limits
                axes.set_xlim(-1, 1)  # set x-axis limits

                axes.yaxis.set_ticks(
                    [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                )  # set y-axis ticks
                axes.xaxis.set_ticks(
                    [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                )  # set x-axis ticks

                axes.yaxis.set_ticklabels(
                    ["-1", "", "-0.5", "", "0", "", "0.5", "", "1"]
                )  # set y-axis tick labels
                axes.xaxis.set_ticklabels(
                    ["-1", "", "-0.5", "", "0", "", "0.5", "", "1"]
                )  # set x-axis tick labels

                # add a horizontal and a vertical line through the origin
                axes.axvline(x=0, color="black", linewidth=1, linestyle="-")
                axes.axhline(y=0, color="black", linewidth=1, linestyle="-")

                # draw grid lines
                axes.grid(
                    which="major", color="grey", linewidth=1, linestyle="-", alpha=0.2
                )

                counter += 1

        # add legend, labels and title
        plt.legend()
        plt.legend(loc="upper right", title="Rating Method", title_fontsize=14)
        plt.xlabel("Mean Valence")
        plt.ylabel("Mean Arousal")
        plt.title(
            "Mean Valence and Arousal Ratings for individual participants",
            fontsize=16,
            fontweight="bold",
        )

        # show figure
        # plt.show()

        # save the figure as a PDF file
        plt.savefig(plot_path + "mean_valence_arousal_individual.pdf")

        # close the current figure
        plt.close(figure)

    # ------------------------ STEP 3E -----------------------------
    elif step == "3e":
        # format data for violin plots
        # create empty dataframe to store values for violin plot
        violin_dataset = pd.DataFrame(
            {
                "dataset": [],  # dataset
                "sj_id": [],  # subject ID
                "quadrant": [],  # quadrant
                "dimension": [],  # dimension
                "statistic": [],  # statistic
                "values": [],  # value
            }
        )

        # loop over datasets
        for index, dataset in enumerate(datasets):
            # open the preprocessed data file
            preprocessed_data = pd.read_csv(
                descriptive_path + "cr_{}_descriptive_individual.csv".format(dataset)
            )

            for index_sub, subject in enumerate(preprocessed_data["sj_id"].unique()):
                data_subject = preprocessed_data[preprocessed_data["sj_id"] == subject]

                for index_quadrant, quadrant in enumerate(
                    data_subject["quadrant"].unique()
                ):
                    data_quadrant = data_subject[data_subject["quadrant"] == quadrant]

                    # average over rating methods
                    averaged_data_subject = data_quadrant.mean()

                    for index_dim, dimension in enumerate(dimensions):
                        for index_stat, statistic in enumerate(statistics):
                            index_row = (
                                index
                                * 51
                                * (
                                    (index_sub + 1)
                                    * len(statistics * len(dimensions) * len(quadrants))
                                    + index_quadrant * len(statistics) * len(dimensions)
                                    + index_dim * len(statistics)
                                    + index_stat
                                )
                                + (index_sub + 1)
                                * len(statistics)
                                * len(dimensions)
                                * len(quadrants)
                                + index_quadrant * len(statistics) * len(dimensions)
                                + index_dim * len(statistics)
                                + index_stat
                            )

                            # get corresponding value
                            if dimension == "valence":
                                if statistic == "mean":
                                    value = averaged_data_subject["cr_mean_v"]
                                elif statistic == "std":
                                    value = averaged_data_subject["cr_std_v"]
                                elif statistic == "skew":
                                    value = averaged_data_subject["cr_skew_v"]
                                elif statistic == "kurtosis":
                                    value = averaged_data_subject["cr_kurtosis_v"]
                                else:
                                    value = np.nan
                            elif dimension == "arousal":
                                if statistic == "mean":
                                    value = averaged_data_subject["cr_mean_a"]
                                elif statistic == "std":
                                    value = averaged_data_subject["cr_std_a"]
                                elif statistic == "skew":
                                    value = averaged_data_subject["cr_skew_a"]
                                elif statistic == "kurtosis":
                                    value = averaged_data_subject["cr_kurtosis_a"]
                                else:
                                    value = np.nan
                            elif dimension == "distance":
                                if statistic == "mean":
                                    value = averaged_data_subject["cr_mean_dist"]
                                elif statistic == "std":
                                    value = averaged_data_subject["cr_std_dist"]
                                elif statistic == "skew":
                                    value = averaged_data_subject["cr_skew_dist"]
                                elif statistic == "kurtosis":
                                    value = averaged_data_subject["cr_kurtosis_dist"]
                                else:
                                    value = np.nan
                            elif dimension == "angle":
                                if statistic == "mean":
                                    value = averaged_data_subject["cr_mean_angle"]
                                elif statistic == "std":
                                    value = averaged_data_subject["cr_std_angle"]
                                elif statistic == "skew":
                                    value = averaged_data_subject["cr_skew_angle"]
                                elif statistic == "kurtosis":
                                    value = averaged_data_subject["cr_kurtosis_angle"]
                                else:
                                    value = np.nan
                            else:
                                value = np.nan

                            violin_dataset.loc[index_row, "dataset"] = dataset
                            violin_dataset.loc[index_row, "sj_id"] = subject
                            violin_dataset.loc[index_row, "quadrant"] = quadrant
                            violin_dataset.loc[index_row, "dimension"] = dimension
                            violin_dataset.loc[index_row, "statistic"] = statistic
                            violin_dataset.loc[index_row, "values"] = value

        # save as one data file
        violin_dataset.to_csv(
            os.path.join(
                os.path.join(plot_path, "descriptives"), "cr_violin_dataset.csv"
            ),
            index=False,
        )

        # 4 violin plots for each quadrant, averaged over rating methods, depicting AVR and CEAP in two different colors
        # with two subplots: mean, std
        # (does not make sense to depict skew and kurtosis in the same plot as scale is different)
        # x 4 dimensions: valence, arousal, distance, angle = 16 violin plots in total

        for quadrant in quadrants:
            data_quadrant = violin_dataset[violin_dataset["quadrant"] == quadrant]
            for dimension in dimensions:
                data_quadrant_dimension = data_quadrant[
                    data_quadrant["dimension"] == dimension
                ]

                # create a violin plot
                sns.violinplot(
                    data=data_quadrant_dimension,
                    x="statistic",
                    y="values",
                    hue="dataset",
                    split=True,
                    inner="quart",
                    linewidth=1,
                )

                # add legend, labels and title
                plt.xlabel("Statistic")
                plt.ylabel("Rating")
                plt.title(
                    "Quadrant " + quadrant + " - " + dimension,
                    fontsize=16,
                    fontweight="bold",
                )
                # show figure
                # plt.show()

                # save the figure as a PDF file
                plt.savefig(
                    plot_path + "violin_{}.pdf".format(dimension + "_" + quadrant)
                )

                # close the current figure
                plt.close()

    else:
        print("Error: Step not found.")
