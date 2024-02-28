"""
Plotting radar chart for AVR questionnaire data.

The following steps are performed:
    - Normalization of data (so that all scales are the same)
    - Plotting of radar chart with all questionnaires combined
    - Saving of radar chart as .pdf file

Author: Lucy Roellecke
Created on: 27 February 2024
Last updated: 28 February 2024
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
questionnaire_labels = [
    ("Emotion Representation"),
    "Invasiveness",
    "Presence",
    "Satisfaction",
    "System Usability Scale",
]  # labels for the questionnaires
rating_methods = ["Grid", "Flubber", "Proprioceptive", "Baseline"]  # rating methods to be included in the radar chart
rm_colors = {
    "Grid": "#F8766D",
    "Flubber": "#00BA38",
    "Proprioceptive": "#619CFF",
    "Baseline": "#C77CFF",
}  # colors for the rating methods


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
                line=dict(color=rm_colors[rating_methods[index_method]]),  # set color of the line
                # fill='toself',    # uncomment to fill the polygons  # noqa: ERA001
                # opacity=0.5,    # uncomment to set the opacity of the polygons  # noqa: ERA001
                name=rating_methods[index_method],
                textfont_size=20,
            )
            for index_method, rating_method in enumerate(rating_methods)
        ],
        layout=go.Layout(
            title=go.layout.Title(text="Questionnaire Scores across Rating Methods", font=dict(size=20)),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    angle=-288,
                    tickangle=-270,  # so that ticks are not upside down
                    tickfont_size=10,
                    tickfont_color="grey",
                    tickvals=[0, 20, 40, 60, 80, 100],
                    tickmode="array",
                    ticktext=["", "20", "40", "60", "80", ""],
                    ticklen=0,
                    linewidth=0,
                ),
                angularaxis=dict(
                    tickvals=questionnaires, tickmode="array", ticktext=questionnaire_labels, tickfont_size=18
                ),
            ),
            template="none",  # white background and grey grid
            paper_bgcolor="#E3F0E9",  # set background color to fit other plots on the poster
            legend=dict(font=dict(size=16)),
            showlegend=True,  # show legend
        ),
    )

    # show figure
    # fig.show()  # noqa: ERA001

    return fig  # noqa: RET504


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
        if not os.path.exists(resultpath_phase_datavisualization):  # noqa: PTH110
            os.makedirs(resultpath_phase_datavisualization)  # noqa: PTH103
        # save radar chart as .pdf file
        radar_chart.write_image(resultpath_phase_datavisualization + "radar_chart.pdf")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

# %%
