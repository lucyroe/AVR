"""
Plotting radar chart for AVR questionnaire data.

The following steps are performed:
    - Normalization of data (so that all scales are the same)
    - Plotting of radar chart with all questionnaires combined
    - Saving of radar chart as .pdf file

Author: Lucy Roellecke
Created on: 27 February 2024
Last updated: 27 February 2024
"""

# %% Import
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
datapath = "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/data/"  # path where data is saved
filename = "assessment_preprocessed.csv"  # name of the file containing the preprocessed questionnaire data
resultpath = (
    "/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AVR/results/"
)  # path where results should be saved
phases = ["phase1"]  # phases for which the radar chart should be plotted
# "phase2"
questionnaires = [
    "emo_rep",
    "invasiveness",
    "presence_score",
    "satisfaction",
    "sus_score",
]  # questionnaires to be included in the radar chart
scales = {
    "emo_rep": [0, 6],
    "invasiveness": [0, 6],
    "presence_score": [0, 6],
    "satisfaction": [0, 6],
    "sus_score": [0, 100],
}  # original scales of the questionnaires
rating_methods = ["Baseline", "Flubber", "Grid", "Proprioceptive"]  # rating methods to be included in the radar chart


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def normalize(value: float, scale: list) -> float:
    """
    Normalize a value to a range of 0 to 100.

    Args:
    ----
    value: value to be normalized
    scale: list with the minimum and maximum value of the old scale

    Returns:
    -------
    normalized value

    """
    return ((value - scale[0]) / (scale[1] - scale[0])) * 100


def plot_radar(data: pd.DataFrame, questionnaires: list, rating_methods: list) -> plt.figure():
    """
    Plot a radar chart for the given data.

    Args:
    ----
    data: dataframe with the normalized mean values for all questionnaires and rating methods
    questionnaires: list with the names of the questionnaires to be included in the radar chart
    rating_methods: list with the names of the rating methods to be included in the radar chart

    Returns:
    -------
    None

    """
    # add last point to the list of questionnaires to close the polygon
    questionnaires = [*questionnaires, questionnaires[0]]
    # create empty list to store data of the rating methods
    data_rating_methods = []
    # get data of the rating methods
    for index_rm, _rating_method in enumerate(rating_methods):
        data_rating_method = list(data[data["rating_method"] == rating_methods[index_rm]]["response"])
        data_rating_method = [
            *data_rating_method,
            data_rating_method[0],
        ]  # add last point to the list of data to close the polygon
        data_rating_methods += [data_rating_method]
    # create figure
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=data_rating_methods[index_method],
                theta=questionnaires,
                # fill='toself',    # uncomment to fill the polygons  # noqa: ERA001
                name=rating_methods[index_method],
            )
            for index_method, rating_method in enumerate(rating_methods)
        ],
        layout=go.Layout(
            title=go.layout.Title(text="Questionnaire Scores across Rating Methods"),
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
        ),
    )

    # show figure
    fig.show()

    return fig


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    for phase in phases:
        # Load data
        data = pd.read_csv(datapath + phase + "/" + filename)

        # %% NORMALIZE DATA
        # create empty dataframe to fill with normalized mean values for all questionnaires
        normalized_data = pd.DataFrame({"questionnaire": [], "rating_method": [], "response": []})
        # loop over questionnaires
        for questionnaire_index, questionnaire in enumerate(questionnaires):
            # get data of that questionnaire
            questionnaire_data = data[data["questionnaire"] == questionnaire]["response"]
            # get original scale of the questionnaire
            questionnaire_scale = scales[questionnaire]
            for rating_method_index, rating_method in enumerate(rating_methods):
                # get data of that rating method
                rating_method_data = questionnaire_data[data["rating_method"] == rating_method]
                # calculate the mean of the questionnaire for that rating method
                questionnaire_mean_rating_method = rating_method_data.mean()
                # normalize the mean value
                normalized_mean_questionnaire_rating_method = normalize(
                    questionnaire_mean_rating_method, questionnaire_scale
                )

                # add rating method, questionnaire and normalized mean value to dataframe
                index = questionnaire_index * len(rating_methods) + rating_method_index
                normalized_data.loc[index, "questionnaire"] = questionnaire
                normalized_data.loc[index, "rating_method"] = rating_method
                normalized_data.loc[index, "response"] = normalized_mean_questionnaire_rating_method

        # %% PLOT AND SAVE RADAR CHART
        radar_chart = plot_radar(normalized_data, questionnaires, rating_methods)

        resultpath_phase_datavisualization = resultpath + phase + "/datavisualization/"
        # create path if it does not exist
        if not os.path.exists(resultpath_phase_datavisualization):
            os.makedirs(resultpath_phase_datavisualization)
        # save radar chart as .pdf file
        radar_chart.write_image(resultpath_phase_datavisualization + "radar_chart.pdf")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

# %%
