#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create plots and HTML for overall pianist style distinctiveness"""

import os

import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from deep_pianist_identification import utils, plotting

CONCEPTS = ["Melody", "Harmony", "Rhythm", "Dynamics", "Overall"]
FACTORIZED_MODEL_RESULTS = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
PLOT_OUTPUT_DIR = os.path.join(utils.get_project_root(), "app/assets/plots/style")

WIDTH, HEIGHT = 500, 500
MARGIN = 60


def load_csv(csv_name: str = "class_accuracy.csv") -> pd.DataFrame:
    """Loads a .csv file created when validating the factorized model"""
    if not csv_name.endswith('.csv'):
        csv_name += '.csv'
    path = os.path.join(FACTORIZED_MODEL_RESULTS, csv_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {csv_name} inside {FACTORIZED_MODEL_RESULTS}. Have you run `validation.py`?"
        )
    return pd.read_csv(path)


def load_confusion_matrix(concept_name: str) -> pd.DataFrame:
    loaded = load_csv(f'confusion_matrices/confusion_matrix_{concept_name}.csv')
    if set(loaded.columns.tolist()) != set(loaded.index.tolist()):
        loaded.index = loaded.columns.tolist()
    return loaded


def create_html(pianist: str, template: str = os.path.join(utils.get_project_root(), 'app/pages/templates/style.html')):
    if not os.path.exists(template):
        raise FileNotFoundError(
            f'Could not find template at location {template}'
        )
    with open(template, 'r') as f:
        txt = (
            f.read()
            .replace('Abdullah Ibrahim', pianist)
            .replace('abdullah_ibrahim', pianist.lower().replace(' ', '_'))
        )
    out_path = os.path.join(utils.get_project_root(), f'app/pages/style/{pianist.replace(" ", "_").lower()}.html')
    with open(out_path, 'w') as f:
        f.write(txt)


class RadarPlotOverallStyle(plotting.BasePlot):
    def __init__(self, df: pd.DataFrame, pianist_name: str):
        super().__init__()
        self.pianist_name = pianist_name
        self.df = self._format_df(df)
        # Plotly expects lists as input, rather than a dataframe
        self.categories, self.values = self.df['concepts'].tolist(), self.df['class_acc'].tolist()
        self.fig = None

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["concepts"] = (
            df["concepts"]
            .apply(lambda x: "overall" if x == "melody+harmony+rhythm+dynamics" else x)
            .str
            .title()
        )
        return (
            df[
                (df['pianist'] == self.pianist_name) &
                (df['concepts'].isin(CONCEPTS))
                ].reset_index(drop=True)
            [["concepts", "class_acc"]]
        )

    def _create_plot(self):
        self.fig = go.Figure(
            data=go.Scatterpolar(
                r=self.values + [self.values[0]],
                theta=self.categories + [self.categories[0]],
                fill='toself',
                fillcolor='rgba(34, 123, 234, 0.5)',
                line=dict(
                    color='rgb(0, 0, 128)',
                    width=plotting.LINEWIDTH
                ),
                hoverinfo='none'
            )
        )

    def _format_fig(self) -> None:
        self.fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0., 1.],
                    showline=True,
                    linewidth=plotting.LINEWIDTH,
                    linecolor='lightgray',
                    showticklabels=False
                ),
                angularaxis=dict(
                    linewidth=plotting.LINEWIDTH,
                    linecolor='gray',
                    showline=True,
                    rotation=90,
                    direction="clockwise",
                )
            ),
            title=None,
            width=WIDTH,
            height=HEIGHT,
            margin=dict(
                l=MARGIN,
                r=MARGIN,
                t=MARGIN,
                b=MARGIN
            ),
            showlegend=False,
            dragmode=False,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font_family='"Open Sans", verdana, arial, sans-serif',
        )

    def save_fig(self):
        out_dir = os.path.join(PLOT_OUTPUT_DIR, "radarplots")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out = os.path.join(out_dir, f"radarplot_{self.pianist_name.replace(' ', '_').lower()}.html")
        self.fig.write_html(
            out,
            include_plotlyjs="cdn",
            config=dict(
                modeBarButtonsToRemove=['zoom', 'pan', 'select', 'lasso2d']
            )
        )


class BarPlotSimilarPianists(plotting.BasePlot):
    def __init__(self, df: pd.DataFrame, pianist_name: str, concept_name: str):
        super().__init__()
        self.pianist_name = pianist_name
        self.df = self._format_df(df)
        self.x, self.y = self.df.index.tolist(), self.df.tolist()
        self.concept_name = concept_name

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        small = df[self.pianist_name]
        # TODO: think a bit more about this. could express WRT accuracy of "actual" class predictions?
        #  also, do we want to take rows/columns here? which of these best contributes to "similarity"?
        #  rows = actually was Bill Evans, but model predicts someone else
        #  columns = actually was someone else, but model predicts Bill Evans
        return (
            small[small.index != self.pianist_name]
            .sort_values(ascending=False)
            .div(100.)
            .iloc[:5]
        )

    def _create_plot(self):
        self.fig = go.Figure(
            go.Bar(
                x=self.x,
                y=self.y,
                marker=dict(
                    color=plotting.CONCEPT_COLOR_MAPPING[self.concept_name][:-2],
                ),
                textposition='outside',
                hoverinfo='none'
            )
        )

    def _add_pianist_images(self):
        for x_val, (pianist, y_val) in enumerate(zip(self.x, self.y)):
            pianist = pianist.lower().replace(' ', '_')
            # TODO: probably won't work when hosting online
            img = Image.open(os.path.join(utils.get_project_root(), f"app/assets/images/pianists/{pianist}.png"))
            self.fig.add_layout_image(
                dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=x_val - 0.2,
                    y=y_val + 0.25,
                    sizex=0.5,
                    sizey=0.2,
                    opacity=1.,
                )
            )

    def _format_fig(self):
        self.fig.update_layout(
            xaxis=dict(
                # title=dict(
                #     text="Pianist",
                #     font=dict(size=14),
                # ),
                # title_standoff=0,
                # ticklabelposition="outside",
                tickangle=0,
                visible=True,
                showline=True,
                linewidth=plotting.LINEWIDTH,
                linecolor='lightgray',
                fixedrange=True,
            ),
            yaxis=dict(
                # title=dict(
                #     text="Similarity",
                # ),
                # title_standoff=0,
                # ticklabelposition="outside",
                tickangle=0,
                range=[0., 1.],
                visible=True,
                showline=True,
                linewidth=plotting.LINEWIDTH,
                linecolor='lightgray',
                fixedrange=True,
                showticklabels=False
            ),
            font_family='"Open Sans", verdana, arial, sans-serif',
            bargap=0.2,
            title=None,
            width=WIDTH * 1.5,
            height=HEIGHT,
            margin=dict(
                l=50,
                r=50,
                t=50,
                b=100
            ),
            showlegend=False,
            dragmode=False,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=True,
        )
        self._add_pianist_images()

    def save_fig(self):
        out_dir = os.path.join(PLOT_OUTPUT_DIR, "barplots")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        fn = f"barplot_{self.pianist_name.replace(' ', '_').lower()}_{self.concept_name.lower()}.html"
        self.fig.write_html(
            os.path.join(out_dir, fn),
            include_plotlyjs="cdn",
            config=dict(
                responsive=True,
                modeBarButtonsToRemove=[
                    'zoom', 'pan', 'select', 'lasso2d', "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
                ]
            )
        )


def main():
    # Load in the dataframe of class accuracy scores
    class_acc_df = load_csv(csv_name="class_accuracy.csv")
    # Iterate over all pianists and their respective data
    for pianist_name, pianist_data in class_acc_df.groupby('pianist'):
        # Create the radar plot
        rp = RadarPlotOverallStyle(pianist_data, pianist_name)
        rp.create_plot()
        rp.save_fig()
        # Iterate over all the concepts (melody, harmony), apart from our last one ("overall", i.e. using all concepts)
        for concept in CONCEPTS[:-1]:
            # Load in the confusion matrix for predictions made with this concept
            confusion_matrix = load_confusion_matrix(concept)
            # Create the bar plot for this pianist and concept
            bp = BarPlotSimilarPianists(confusion_matrix, pianist_name, concept)
            bp.create_plot()
            bp.save_fig()
        # Create the HTML by copying the template and updating all of the file paths correctly
        create_html(pianist_name)


if __name__ == "__main__":
    main()
