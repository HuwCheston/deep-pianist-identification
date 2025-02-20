#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes, functions, and variables."""

import json
import os
from datetime import timedelta
from urllib.error import HTTPError, URLError

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pretty_midi import note_number_to_name
from scipy.cluster.hierarchy import dendrogram
from skimage import io
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

from deep_pianist_identification import utils
from deep_pianist_identification.explainability.cav_utils import (
    create_linkage_matrix, correlation_based_clustering, get_sign_count_significance
)
from deep_pianist_identification.preprocessing import m21_utils as m21_utils

# Define constants
WIDTH = 18.8  # This is a full page width: half page plots will need to use 18.8 / 2
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
WHITE = '#FFFFFF'

RED = '#FF0000'
GREEN = '#008000'
BLUE = '#0000FF'
YELLOW = '#FFFF00'
RGB = [RED, GREEN, BLUE]
CONCEPT_COLOR_MAPPING = {"Melody": "#ff0000ff", "Harmony": "#00ff00ff", "Rhythm": "#0000ffff", "Dynamics": "#ff9900ff"}

LINEWIDTH = 2
LINESTYLE = '-'
TICKWIDTH = 3
MARKERSCALE = 1.6
MARKERS = ['o', 's', 'D']
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

DOTTED = 'dotted'
DASHED = 'dashed'

# Remote URL where images of all pianists are stored
IMG_LOC = "https://raw.githubusercontent.com/HuwCheston/Jazz-Trio-Database/refs/heads/main/references/images/musicians"
# Used for saving PNG images with the correct background
SAVE_KWS = dict(facecolor=WHITE, dpi=300)
SAVE_EXTS = ['svg', 'png', 'pdf']
# Keyword arguments to use when applying a grid to a plot
GRID_KWS = dict(color=BLACK, alpha=ALPHA, lw=LINEWIDTH / 2, ls=LINESTYLE)
# Used when adding a legend to an axis
LEGEND_KWS = dict(frameon=True, framealpha=1, edgecolor=BLACK)

N_BOOT = 10000
N_BINS = 50
PLOT_AFTER_N_EPOCHS = 5


def fmt_heatmap_axis(heatmap_ax: plt.Axes):
    """Applies consistent formatting to a heatmap plot"""
    # Iterating through both the heatmap and any attached axis (colourbars, e.g.)
    for a in [heatmap_ax, *heatmap_ax.figure.axes]:
        # Setting aesthetics for each individual spine (t, b, l, r)
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_color(BLACK)
            spine.set_linewidth(LINEWIDTH)
        # Setting aesthetics for ticks
        plt.setp(a.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        a.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
    # Flip the y-axis of the whole plot
    heatmap_ax.invert_yaxis()


def save_fig_all_exts(fpath: str, fig: plt.Figure) -> None:
    """Saves a figure `fig` with filepath `fpath` using all extensions in `SAVE_EXTS`"""
    # Check that the directory we want to save the figure in exists
    root_dir = os.path.dirname(fpath)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f'path {root_dir} does not exist')
    # For backwards compatibility: if we've added the extension in already, remove it
    if fpath.endswith(".png"):
        fpath = fpath.replace(".png", "")
    # Iterate through all the filetypes we want to use and save the plot
    for ext in SAVE_EXTS:
        fig.savefig(fpath + f'.{ext}', format=ext, **SAVE_KWS)
    # Close all figures at the end
    plt.close('all')


class BasePlot:
    """Base plotting class from which all others inherit"""
    mpl.rcParams.update(mpl.rcParamsDefault)

    # These variables should all be overridden at will in child classes
    df = None
    fig, ax = None, None
    g = None

    def __init__(self, **kwargs):
        # Set fontsize
        plt.rcParams.update({'font.size': FONTSIZE})
        self.figure_title = kwargs.get('figure_title', 'baseplot')

    def _format_df(self, df: pd.DataFrame):
        return df

    def create_plot(self) -> tuple:
        """Calls plot creation, axis formatting, and figure formatting classes, then saves in the decorator"""
        self._create_plot()
        self._format_ax()
        self._format_fig()
        return self.fig, self.ax

    def _create_plot(self) -> None:
        """This function should contain the code for plotting the graph"""
        return

    def _format_ax(self) -> None:
        """This function should contain the code for formatting the `self.ax` objects"""
        return

    def _format_fig(self) -> None:
        """This function should contain the code for formatting the `self.fig` objects"""
        return

    def close(self):
        """Alias for `plt.close()`"""
        plt.close(self.fig)


class HeatmapConfusionMatrix(BasePlot):
    def __init__(self, confusion_mat: np.ndarray, pianist_mapping: dict, title: str = '', **kwargs):
        super().__init__(**kwargs)
        self.mat = confusion_mat
        self.title = title
        self.pianist_mapping = pianist_mapping
        self.num_classes = len(self.pianist_mapping.keys())
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))

    def _create_plot(self) -> None:
        return sns.heatmap(
            data=self.mat, ax=self.ax, cmap="Reds", linecolor=WHITE, square=True, annot=False,
            fmt='.0f', linewidths=LINEWIDTH // 2, vmin=0, vmax=100,
            cbar_kws=dict(
                label='Probability (%)', location="right", shrink=0.75,
                ticks=[0, 25, 50, 75, 100],
            ))

    def _format_ax(self):
        self.ax.set(
            xlabel="Predicted pianist", ylabel="Actual pianist", title=self.title,
            xticks=range(self.num_classes), yticks=range(self.num_classes)
        )
        # Set axis ticks correctly
        self.ax.set_xticks([i + 0.5 for i in self.ax.get_xticks()], self.pianist_mapping.values(), rotation=90)
        self.ax.set_yticks([i + 0.5 for i in self.ax.get_yticks()], self.pianist_mapping.values(), rotation=0)
        # Make all spines visible on both axis and colorbar
        for ax in [self.ax, self.ax.figure.axes[-1]]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(BLACK)
                spine.set_linewidth(LINEWIDTH)
        # Set axis and tick thickness
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)

    def _format_fig(self):
        self.ax.invert_yaxis()
        self.fig.tight_layout()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations/confusion_matrices")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        outpath = os.path.join(fold, f"heatmap_confusion_matrix_{self.title}")
        save_fig_all_exts(outpath, self.fig)


class BarPlotMaskedConceptsAccuracy(BasePlot):
    def __init__(self, df, xvar: str = "track_acc", xlabel: str = "Accuracy (track-level)", **kwargs):
        super().__init__(**kwargs)
        self.xvar = xvar
        self.xlabel = xlabel
        self.df = self._format_df(df.dropna(subset="concepts"))
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 3))

    def _format_df(self, df):
        pd.options.mode.chained_assignment = None
        counter = lambda x: abs(x.count('+') - 3)
        df["n_concepts"] = df['concepts'].apply(counter)
        full_acc = df[df['concepts'] == 'melody+harmony+rhythm+dynamics']['track_acc'].iloc[0]
        sort_and_title = lambda x: ', '.join(sorted([i.title() for i in x.split('+')]))
        df["concepts"] = df["concepts"].apply(sort_and_title)
        df['accuracy_loss'] = (df['track_acc'] - full_acc).abs()
        pd.options.mode.chained_assignment = "warn"
        return (
            df.sort_values(by=["n_concepts", "track_acc"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def _create_plot(self) -> None:
        return sns.barplot(
            self.df, y="concepts", x=self.xvar, hue="n_concepts",
            palette="tab10", legend=True,
            edgecolor=BLACK, linewidth=LINEWIDTH,
            linestyle=LINESTYLE, ax=self.ax, zorder=10
        )

    def _add_bar_labels(self, x_mult: float = 0.01):
        for idx, row in self.df.iterrows():
            if idx == 0:
                continue
            self.ax.text(
                row['track_acc'] + x_mult, idx,
                '{:.3f}'.format(round(row['accuracy_loss'], 3)),
                va='center', ha='left'
            )

    def _format_ax(self):
        # self._add_bar_labels()
        sns.move_legend(self.ax, loc="lower right", title='$N$ masks', **LEGEND_KWS)
        self.ax.set(ylabel="Musical dimensions", xlabel=self.xlabel, xlim=(0, 1))
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
        self.ax.grid(axis="x", zorder=0, **GRID_KWS)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self, title: str = 'barplot_masked_concept'):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        save_fig_all_exts(os.path.join(fold, f"{title}_{self.xvar}"), self.fig)


class BarPlotSingleMaskAccuracy(BarPlotMaskedConceptsAccuracy):
    def __init__(self, df, **kwargs):
        super().__init__(df, xvar="accuracy_loss", xlabel="Accuracy loss", **kwargs)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH // 2, WIDTH // 3))

    def _format_df(self, df):
        def format_concept(row):
            row_concepts = row['concepts'].split(', ')
            masked_concept = [i for i in all_concepts if i not in row_concepts][0]
            row['concepts'] = masked_concept
            return row

        df = super()._format_df(df)
        all_concepts = df[df['n_concepts'] == 0]['concepts'].iloc[0].split(', ')
        single_concept = df[df['n_concepts'] == 1].apply(format_concept, axis=1)
        single_concept['cmap'] = single_concept['concepts'].map(CONCEPT_COLOR_MAPPING)
        return single_concept.sort_values(by='accuracy_loss', ascending=False)

    def _create_plot(self):
        sns.barplot(
            self.df, y='concepts', x=self.xvar, hue='concepts', legend=True,
            edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE,
            ax=self.ax, zorder=10, palette=self.df['cmap'].tolist()
        )

    def _format_ax(self):
        super()._format_ax()
        self.ax.get_legend().remove()
        self.ax.set(ylabel='Musical dimensions', xlim=(0, self.df[self.xvar].max() * 1.1))

    def save_fig(self, title: str = 'barplot_single_mask'):
        super().save_fig(title)


class BarPlotSingleConceptAccuracy(BarPlotMaskedConceptsAccuracy):
    VLINE_KWS = dict(ymin=0, ymax=1, zorder=1000, linewidth=LINEWIDTH, linestyle=DASHED)

    def __init__(self, df, **kwargs):
        super().__init__(df, xvar="track_acc", xlabel='Accuracy', **kwargs)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH // 2, WIDTH // 3))
        self.baseline_accuracy = kwargs.get("baseline_accuracy", None)
        self.full_accuracy = kwargs.get("full_accuracy", None)

    def _format_df(self, df):
        df = super()._format_df(df)
        single_concept = df[df['n_concepts'] == 3]
        single_concept['cmap'] = single_concept['concepts'].map(CONCEPT_COLOR_MAPPING)
        return single_concept

    def _create_plot(self):
        sns.barplot(
            self.df, y='concepts', x=self.xvar, hue='concepts', legend=True,
            edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE,
            ax=self.ax, zorder=10, palette=self.df['cmap'].tolist()
        )

    def _format_ax(self):
        super()._format_ax()
        self.ax.get_legend().remove()
        self.ax.set_ylabel('Musical dimensions')
        if self.full_accuracy is not None:
            self.ax.set_xlim(0, self.full_accuracy * 1.1)
        else:
            self.ax.set_xlim(0, self.df[self.xvar].max() * 1.1)
        for acc, acc_name in zip([self.baseline_accuracy, self.full_accuracy], ["Baseline", "Full"]):
            if acc is None:
                continue
            self.ax.axvline(acc, color=BLACK, **self.VLINE_KWS)
            self.ax.text(acc, -0.75, f'${acc_name}$', ha='center', va='bottom')

    def save_fig(self, title: str = 'barplot_single_concept'):
        super().save_fig(title)


class _StripplotTopKFeatures(BasePlot):
    """Parent class for strip plots showing the top/bottom-K features for a given performer"""

    ERROR_KWS = dict(lw=LINEWIDTH, color=BLACK, capsize=8, zorder=0, elinewidth=LINEWIDTH, ls='none')
    SCATTER_KWS = dict(legend=False, s=200, edgecolor=BLACK, zorder=10)

    def __init__(
            self,
            concept_dict: dict,
            pianist_name: str = "",
    ):
        super().__init__()
        self.concept_dict = concept_dict
        self.pianist_name = pianist_name
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 2.25))

    def _create_music21(self, _: str):
        return

    def _add_notation(
            self,
            ngram: str,
            y: float,
            x: float = 1.025,
            zoom: float = 0.25,
            right_crop: float = 0.92,
            png_dim: tuple = (125, 350, 1000, 650),
            dpi: int = 300
    ) -> None:
        """Adds a given feature/ngram as notation onto the axis with a given y coordinate"""
        # Create the music21 object as a Score
        notes = self._create_music21(ngram)
        # Convert this score into a numpy array
        cropped_image = m21_utils.score_to_image_array(notes, right_crop=right_crop, png_dim=png_dim, dpi=dpi)
        # Convert the numpy array into a matplotlib object
        imagebox = OffsetImage(cropped_image, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), xycoords='axes fraction', frameon=False, box_alignment=(0, 0.5))
        self.ax.add_artist(ab)

    def add_notation(self, direction: str, ngram_names: list[str]):
        pass

    def _create_plot(self):
        pass

    def _add_performer_image(self, x: float = 0.95, y: float = 0.95):
        """Adds an image of the performer into the plot with the desired x- and y-coordinates"""
        gh = f"{IMG_LOC}/{self.pianist_name.lower().replace(' ', '_')}.png"
        try:
            image = io.imread(gh)
        except (HTTPError, URLError):
            logger.warning(f"Couldn't find performer image for {self.pianist_name} at {gh}!")
            return
        imagebox = OffsetImage(image, zoom=1.0)
        ab = AnnotationBbox(imagebox, (x, y), xycoords='axes fraction', frameon=False, box_alignment=(1., 1.))
        self.ax.add_artist(ab)

    def _format_ax(self):
        """Setting plot aesthetics on an axis-level basis"""
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH)
        self.ax.tick_params(axis='both', width=TICKWIDTH)

    def _format_fig(self):
        """Setting plot aeshetics on a figure-level basis"""
        self.fig.subplots_adjust(left=0.14, top=0.95, bottom=0.11, right=0.8)

    def _save_fig(self, output_dir: str, concept_name: str):
        fp = os.path.join(
            output_dir,
            f'topk_stripplot_{self.pianist_name.lower().replace(" ", "_")}_{concept_name}'
        )
        save_fig_all_exts(fp, self.fig)


class StripplotTopKMelodyFeatures(_StripplotTopKFeatures):
    """ Creates a strip plot showing the top-/bottom-k melody features associated with a given performer

    Examples:
    >>> t = dict(
    >>>     top_ors=np.array([1.2, 1.15, 1.10, 1.05, 1.01]),    # The top K odds ratios for this performer
    >>>     bottom_ors=np.array([0.8, 0.85, 0.9, 0.95, 0.975]),    # The bottom K odds ratios
    >>>     top_std=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),    # The corresponding top K standard deviations
    >>>     bottom_std=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),    # Bottom k standard deviations
    >>>     # These arrays contain the names of the n-grams, each of them should be unique
    >>>     top_names=np.array(["M_[1, 2, 1]", "M_[1, 1, 3]", "M_[1, 2, 1]", "M_[1, 5, 1]", "M_[2, 1, 2]"]),
    >>>     bottom_names=np.array(["M_[1, 2, 3]", "M_[1, -4, 1]", "M_[-1, -1, -5]", "M_[1, -2, 5]", "M_[-1, -1, 3]"])
    >>> )
    >>> plo = StripplotTopKMelodyFeatures(t, "Testy McTestFace")
    >>> plo.create_plot()
    """

    def __init__(
            self,
            concept_dict: dict,
            pianist_name: str = ""
    ):
        super().__init__(concept_dict, pianist_name)

    def _create_music21(self, feature: str):
        # Evaluate the n-gram as a list of integers
        eval_feature = m21_utils.intervals_to_pitches(feature)
        # Center the feature so that the mean pitch height is G4
        g4 = utils.MIDDLE_C + (utils.OCTAVE // 2)
        centered_feature = m21_utils.center_feature(eval_feature, threshold=g4)
        # Convert the feature to a music21 measure
        measure = m21_utils.melody_feature_to_m21_measure(centered_feature)
        # Add the measure to the part and the part to the score
        part = m21_utils.measure_to_part(measure)
        score = m21_utils.part_to_score(part)
        # Estimate the spelling
        estimated_spelling = m21_utils.estimate_m21_score_spelling(score)
        m21_utils.adjust_m21_melody_to_partitura(score, estimated_spelling)  # works in-place
        return score

    def _create_plot(self):
        # Hard-coding our y-axis positions is maybe not the best solution for scalability, but works for now
        notation_y_positions = [[0.95, 0.85, 0.75, 0.65, 0.55], [0.45, 0.35, 0.25, 0.15, 0.05]]
        for direction, all_positions in zip(['bottom', 'top'], notation_y_positions):
            sorters = np.array(range(len(self.concept_dict[f"{direction}_ors"])))
            if direction == 'top':
                sorters = sorters[::-1]
            # Format the name by removing the first two characters
            names = np.array([i[2:] for i in self.concept_dict[f"{direction}_names"]])[sorters]
            # Creating the errorbar, showing standard deviation
            self.ax.errorbar(
                self.concept_dict[f"{direction}_ors"][sorters], names,
                xerr=self.concept_dict[f"{direction}_std"][sorters], **self.ERROR_KWS
            )
            # Creating the scatter plot
            sns.scatterplot(
                x=self.concept_dict[f"{direction}_ors"][sorters], y=names,
                facecolor=GREEN if direction == 'top' else RED,
                **self.SCATTER_KWS, ax=self.ax, label=f"{direction.title()}-5",
            )
            # Adds all features as notation onto the axis at a given position
            for y, ng in zip(all_positions, names):
                self._add_notation(ng, y)

    def _format_ax(self):
        self._add_performer_image()
        self.ax.tick_params(right=True)
        # self.ax.set_title(f'{self.pianist_name}, melody features')
        self.ax.set(xlabel='Weights', ylabel='Feature')
        fmt_text = [str(m21_utils.intervals_to_pitches(yl.get_text())) for yl in self.ax.get_yticklabels()]
        self.ax.set_yticks(self.ax.get_yticks(), fmt_text)
        self.ax.grid(axis='x', zorder=0, **GRID_KWS)
        self.ax.axhline(4.5, 0, 1, linewidth=LINEWIDTH, color=BLACK, alpha=ALPHA, ls=DASHED)
        self.ax.axvline(0, 0, 1, linewidth=LINEWIDTH, color=BLACK)
        self.ax.legend(**LEGEND_KWS, loc='lower left')
        super()._format_ax()

    def save_fig(self, output_dir: str):
        super()._save_fig(output_dir, "melody")


class StripplotTopKHarmonyFeatures(_StripplotTopKFeatures):
    """ Creates a strip plot showing the top-/bottom-k harmony features associated with a given performer

    Examples:
        >>> t = dict(
        >>>     top_ors = np.array([1.2, 1.15, 1.10, 1.05, 1.01]),  # The top K odds ratios for this performer
        >>>     bottom_ors = np.array([0.8, 0.85, 0.9, 0.95, 0.975]),  # The bottom K odds ratios
        >>>     top_std = np.array([0.01, 0.01, 0.01, 0.01, 0.01]),  # The corresponding top K standard deviations
        >>>     bottom_std = np.array([0.01, 0.01, 0.01, 0.01, 0.01]),  # Bottom k standard deviations
        >>>      # These arrays contain the names of the features, each of them should be unique
        >>>     top_names = np.array(["H_[1, 2, 1]", "H_[1, 1, 3]", "H_[1, 7, 1]", "H_[1, 5, 1]", "H_[2, 1, 2]"]),
        >>>     bottom_names = np.array(["H_[1, 2, 3]", "H_[1, 4, 1]", "H_[-1, 1, -5]", "H_[1, 2, 5]", "H_[-1, -1, 3]"])
        >>> )
        >>> plo = StripplotTopKHarmonyFeatures(t, "Testy McTestFace")
        >>> plo.create_plot()
    """

    def __init__(
            self,
            concept_dict: dict,
            pianist_name: str = "",
    ):
        super().__init__(concept_dict, pianist_name)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 1.5))

    def _create_plot(self):
        # Maybe not a good idea to hardcode these positions but whatever
        notation_x_positions = [[0.015, 0.115, 0.215, 0.315, 0.415], [0.515, 0.615, 0.715, 0.815, 0.915]]
        for direction, all_positions in zip(['bottom', 'top'], notation_x_positions):
            sorters = np.array(range(len(self.concept_dict[f"{direction}_ors"])))
            if direction == 'top':
                sorters = sorters[::-1]
            # Remove the first two characters from the name of each feature
            names = np.array([i[2:] for i in self.concept_dict[f"{direction}_names"]])[sorters]
            # Creating the errorbar, showing standard deviation
            self.ax.errorbar(
                names, self.concept_dict[f"{direction}_ors"][sorters],
                yerr=self.concept_dict[f"{direction}_std"][sorters], **self.ERROR_KWS
            )
            # Creating the scatter plot
            sns.scatterplot(
                y=self.concept_dict[f"{direction}_ors"][sorters], x=names,
                facecolor=GREEN if direction == 'top' else RED,
                **self.SCATTER_KWS, ax=self.ax, label=f"{direction.title()}-5",
            )
            # Add notation for all features
            for x, ng in zip(all_positions, names):
                self._add_notation(ng, x=x, y=1.125, zoom=0.25, png_dim=(100, 350, 500, 850))

    @staticmethod
    def format_feature(feature: str) -> tuple:
        """Formats the feature from a string to a tuple"""
        return tuple(sorted({0, *eval(feature)}))

    def _create_music21(self, feature: str):
        # Evaluate the feature as a list of integers
        eval_feature = self.format_feature(feature)
        # Center the feature around middle C
        centered_feature = m21_utils.center_feature(eval_feature)
        # Convert the feature into note classes with separate right and left hand
        rh, lh = m21_utils.feature_to_separate_hands(centered_feature)
        # Convert the notes into chords
        rh_chord, lh_chord = m21_utils.measure_to_chord(rh), m21_utils.measure_to_chord(lh)
        # Convert the chords into parts
        rh_part, lh_part = m21_utils.measure_to_part(rh_chord), m21_utils.measure_to_part(lh_chord)
        # Convert the parts to a score
        score = m21_utils.part_to_score([rh_part, lh_part])
        # Estimate the correct enharmonic spelling
        estimated_spelling = m21_utils.estimate_m21_score_spelling(score)
        m21_utils.adjust_m21_hands_to_partitura(score, estimated_spelling)
        return score

    def _format_ax(self):
        self._add_performer_image(x=0.125)
        self.ax.tick_params(top=True)
        # self.ax.set_title(f'{self.pianist_name}, harmony features', y=1.175)
        self.ax.set(ylabel='Weights', xlabel='Feature')
        fmt_text = [str(self.format_feature(yl.get_text())) for yl in self.ax.get_xticklabels()]
        self.ax.set_xticks(self.ax.get_xticks(), fmt_text, rotation=90, va='top', ha='right')
        self.ax.grid(axis='y', zorder=0, **GRID_KWS)
        self.ax.axvline(4.5, 0, 1, linewidth=LINEWIDTH, color=BLACK, alpha=ALPHA, ls=DASHED)
        self.ax.axhline(0, 0, 1, linewidth=LINEWIDTH, color=BLACK)
        self.ax.legend(**LEGEND_KWS, loc='lower right')
        super()._format_ax()

    def _format_fig(self):
        """Setting plot aeshetics on a figure-level basis"""
        self.fig.subplots_adjust(left=0.075, top=0.84, bottom=0.2, right=0.95)

    def save_fig(self, output_dir: str):
        super()._save_fig(output_dir, "harmony")


class HeatmapWhiteboxFeaturePianoRoll(BasePlot):
    def __init__(
            self,
            feature_excerpt,
            performer_name: str,
            feature_name: str,
            clip_name: str
    ):
        super().__init__()
        self.feature_excerpt = feature_excerpt
        self.performer_name = performer_name
        self.feature_name = feature_name
        self.clip_name = clip_name
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 3))

    def _create_plot(self):
        feature_roll = self.feature_excerpt.get_piano_roll(fs=utils.FPS)
        feature_roll[feature_roll == 0] = np.nan
        sns.heatmap(
            feature_roll,
            ax=self.ax,
            alpha=1.0,
            cbar_kws={'label': 'Velocity', 'pad': 0.01},
            cmap="rocket",
            vmin=0,
            vmax=utils.MAX_VELOCITY
        )

    def _format_ax(self):
        self.ax.invert_yaxis()
        self.ax.set(
            ylim=(utils.MIDI_OFFSET, utils.PIANO_KEYS + utils.MIDI_OFFSET),
            title=self.clip_name,
            xlabel="",
            ylabel="",
            yticks=range(0, utils.PIANO_KEYS, 12),
            yticklabels=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, 12)],
        )
        fmt_heatmap_axis(self.ax)

    def _format_fig(self):
        # Figure aesthetics
        self.fig.suptitle(f"{self.performer_name} feature: {self.feature_name}")
        self.fig.supxlabel("Time (seconds)")
        self.fig.supylabel("Note")
        self.fig.tight_layout()

    def save_fig(self, output_loc):
        save_fig_all_exts(output_loc, self.fig)


class BarPlotWhiteboxDatabaseCoefficients(BasePlot):
    """Bar plot showing the performer correlations obtained from both source databases for a whitebox model"""
    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10)
    ERROR_KWS = dict(lw=LINEWIDTH, color=BLACK, capsize=8, zorder=100, elinewidth=LINEWIDTH, ls='none')
    ASTERISK_KWS = dict(ha='center', va='center', zorder=100000)
    X_PADDING = 0.025
    Y_PADDING = 0.1

    def __init__(self, coef_df):
        super().__init__()
        self.df = self._format_df(coef_df)
        self.fig, self.ax = plt.subplots(1, 2, figsize=(WIDTH, WIDTH // 2), sharex=True, sharey=False)

    def _format_df(self, df):
        df['feature'] = df['feature'].str.title()
        mel_df = (
            df[df['feature'] == 'Melody']
            .reset_index(drop=True)
            .sort_values(by='corr', ascending=False)
        )
        sorters = {k: num for num, k in enumerate(mel_df['pianist'].tolist())}
        har_df = (
            df[df['feature'] == 'Harmony']
            .reset_index(drop=True)
            .sort_values(by='pianist', key=lambda x: x.map(sorters))
            .reset_index(drop=True)
        )
        return pd.concat([mel_df, har_df], axis=0).reset_index(drop=True)

    def _create_plot(self):
        for concept, ax in zip(['Melody', 'Harmony'], self.ax.flatten()):
            sub = self.df[self.df['feature'] == concept].reset_index(drop=True)
            sns.barplot(data=sub, x='corr', y='pianist', color=CONCEPT_COLOR_MAPPING[concept], ax=ax, **self.BAR_KWS)
            for y, row in sub.iterrows():
                x = row['corr'] + self.X_PADDING if row['corr'] > 0 else row['corr'] - self.X_PADDING
                y += self.Y_PADDING
                ax.text(x, y, row['sig'], **self.ASTERISK_KWS)

    def _format_ax(self):
        xmin = min([ax.get_xlim()[0] for ax in self.ax.flatten()]) - self.X_PADDING
        xmax = max([ax.get_xlim()[1] for ax in self.ax.flatten()]) + self.X_PADDING
        for concept, ax in zip(['Melody', 'Harmony'], self.ax.flatten()):
            ax.axvline(0, 0, 1, linewidth=LINEWIDTH, color=BLACK)
            ax.set(
                ylabel='Pianist', xlabel='Coefficient ($r$)', yticks=self.ax[0].get_yticks(),
                yticklabels=self.ax[0].get_yticklabels(), title=concept, xlim=(xmin, xmax)
            )
            ax.grid(axis='x', zorder=-1, **GRID_KWS)
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH)
            ax.tick_params(axis='both', width=TICKWIDTH)

    def _format_fig(self):
        # self.fig.supxlabel('Coefficient $r$')
        # self.fig.supylabel('Pianist')
        self.fig.tight_layout()

    def save_fig(self, outpath: str):
        save_fig_all_exts(outpath, self.fig)


class HeatmapCAVSensitivity(BasePlot):
    HEATMAP_KWS = dict(
        square=True, center=0.5, linecolor=WHITE, linewidth=LINEWIDTH // 2, cmap=cmc.bam, vmin=0, vmax=1, alpha=0.7
    )
    CLUSTER_KWS = dict(n_clusters=None, distance_threshold=0, metric='precomputed', linkage='average')
    DENDRO_KWS = dict(truncate_mode=None, no_labels=False, color_threshold=0, above_threshold_color=BLACK)

    def __init__(
            self,
            sign_counts: np.array,
            random_sign_counts: np.array,
            class_names: np.array,
            cav_names: np.array,
    ):
        super().__init__()
        self.class_names = class_names
        self.sensitivity_type = "Mean Sign Count"
        self.cav_names = cav_names
        self.fig = plt.figure(figsize=(WIDTH, WIDTH))
        gs0 = GridSpec(3, 3, width_ratios=[20, 4, 1], height_ratios=[1, 4, 20])
        self.ax = self.fig.add_subplot(gs0[6])
        self.dax1 = self.fig.add_subplot(gs0[3])
        self.dax2 = self.fig.add_subplot(gs0[7])
        self.cax = self.fig.add_subplot(gs0[8])
        # Create arrays
        self.mean_sign_counts = sign_counts.mean(axis=2).T  # (n_performers, n_cavs)
        self.ast_sign_counts = get_sign_count_significance(sign_counts, random_sign_counts).T  # (n_performers, n_cavs)
        # Flatten to (n_performers * n_experiments, n_cavs)
        self.flat_cav_scs = sign_counts.reshape(sign_counts.shape[0], -1).T
        # Flatten to (n_cavs * n_experiments, n_cavs)
        self.flat_perf_scs = np.swapaxes(sign_counts, 0, 1).reshape(sign_counts.shape[1], -1).T
        # Convert the mean sign count matrix to a dataframe for easier plotting
        self.df = self._format_df(self.mean_sign_counts)
        self.ast_df = self._format_df(self.ast_sign_counts)

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        res_df = pd.DataFrame(df, columns=self.cav_names)
        res_df['performer'] = self.class_names
        return res_df.set_index('performer')

    def _create_plot(self):
        self._create_dendrogram()
        self._create_heatmap()

    def _create_dendrogram(self):
        for dax, data, orientation in zip(
                [self.dax1, self.dax2],
                [self.flat_cav_scs, self.flat_perf_scs],
                ['top', 'right']
        ):
            aggc = correlation_based_clustering(data, **self.CLUSTER_KWS)
            linkage = create_linkage_matrix(aggc)
            # Plot the corresponding dendrogram
            with plt.rc_context({'lines.linewidth': LINEWIDTH, 'lines.linestyle': LINESTYLE}):
                dendrogram(linkage, ax=dax, orientation=orientation, **self.DENDRO_KWS)

    def _sort_df_by_dendrogram(self, df):
        # Get the indices of our concepts in the order used by the dendrogram
        xticks = [eval(x.get_text()) for x in self.dax1.get_xticklabels()]
        # Create the mapping for indices and CAVs
        xtick_map = {i: k for i, k in enumerate(self.cav_names)}
        # Map the indices onto the name of the CAV
        xtick_labels = [xtick_map[x] for x in xticks]
        # Do the same for our performer names
        yticks = [eval(y.get_text()) for y in self.dax2.get_yticklabels()]
        ytick_map = {i: k for i, k in enumerate(self.class_names)}
        ytick_labels = [ytick_map[y] for y in yticks]
        # Reindex the dataframe
        return df[xtick_labels].reindex(ytick_labels)

    def _create_heatmap(self):
        self.df = self._sort_df_by_dendrogram(self.df)
        self.ast_df = self._sort_df_by_dendrogram(self.ast_df)
        _ = sns.heatmap(
            self.df, ax=self.ax, fmt="", **self.HEATMAP_KWS, cbar_ax=self.cax, annot=self.ast_df,
            cbar_kws=dict(
                label=self.sensitivity_type, shrink=0.75, ticks=[0., 0.25, 0.5, 0.75, 1.], location="right",
            )
        )

    def _format_ax(self):
        fmt_heatmap_axis(self.ax)
        # Set axis and tick thickness
        # There seems to occasionally be an encoding issue with the arrows here
        self.ax.set(
            xlabel="Harmonic Concept",
            ylabel="Pianist"
        )
        self.ax.invert_yaxis()
        # Format TOP dendrogram
        plt.setp(self.dax1.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.dax1.set(xticklabels=[], yticks=[])
        self.dax1.tick_params(axis='x', width=TICKWIDTH, color=BLACK, bottom="on", )
        for sp in ["left", "top", "right"]:
            self.dax1.spines[sp].set_visible(False)
        # Format RIGHT dendrogram
        plt.setp(self.dax2.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.dax2.set(xticks=[], yticklabels=[])
        self.dax2.tick_params(axis='y', width=TICKWIDTH, color=BLACK, left='on')
        for sp in ["bottom", "top", "right"]:
            self.dax2.spines[sp].set_visible(False)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"cav_plots/{self.sensitivity_type.replace(' ', '_').lower()}_heatmap")
        save_fig_all_exts(fp, self.fig)
        # Dump the csv file as well
        fp = os.path.join(fold, f"cav_plots/{self.sensitivity_type.replace(' ', '_').lower()}.csv")
        self.df.to_csv(fp)


class HeatmapCAVPairwiseCorrelation(BasePlot):
    HEATMAP_KWS = dict(
        cmap="vlag", linecolor=WHITE, square=True, annot=False, fmt='.0f', linewidths=LINEWIDTH // 2, vmin=-1., vmax=1.,
        cbar_kws=dict(label='Correlation ($r$)', location="right", shrink=0.75, ticks=[-1., -0.5, 0., 0.5, 1.]),
        center=0.
    )

    def __init__(
            self,
            matrix: np.array,
            columns: list[str],
            ax_label: str = "Harmonic Concept"
    ):
        super().__init__()
        self.df = pd.DataFrame(matrix, columns=columns).corr()
        self.ax_label = ax_label
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))

    def _create_plot(self):
        return sns.heatmap(self.df, ax=self.ax, **self.HEATMAP_KWS)

    def _format_ax(self):
        fmt_heatmap_axis(self.ax)
        self.ax.set(xlabel=self.ax_label, ylabel=self.ax_label)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fname = self.ax_label.lower().replace(' ', '_')
        fp = os.path.join(fold, f"cav_plots/{fname}_pairwise_correlation_heatmap")
        save_fig_all_exts(fp, self.fig)
        # Dump the csv file as well
        fp = os.path.join(fold, f"cav_plots/{fname}_pairwise_correlation.csv")
        self.df.to_csv(fp)


class BarPlotDatasetDurationCount(BasePlot):
    """Creates figures relating to dataset duration and count. Accepts input in form from utils.get_track_metadata()"""

    X_LABELS = ['Number of recordings', 'Duration of recordings (minutes)', 'Recording duration (minutes)']
    Y_LABELS = ['Pianist', 'Pianist', 'Number of recordings']
    BAR_KWS = dict(stacked=True, edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, zorder=10)
    HIST_KWS = dict(multiple='stack', alpha=ALPHA, edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, bins=100)
    VLINE_KWS = dict(ymin=0, ymax=1, zorder=100, linewidth=LINEWIDTH * 2, linestyle=DASHED)
    MIN_MINUTES = 80
    _split_colors = ["#ffc0cb", "#1f77b4", "#ff0000"]
    SPLIT_PALETTE = sns.color_palette(_split_colors, as_cmap=True)

    def __init__(self, track_metadata: pd.DataFrame):
        super().__init__()
        self.duration_df, self.count_df, self.hist_df, self.split_df = self._format_df(track_metadata)
        self.fig, self.bar_ax, self.hist_ax, self.split_ax = self._get_fig_ax()

    @staticmethod
    def _get_fig_ax():
        fig = plt.figure(figsize=(WIDTH, WIDTH))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax4 = fig.add_subplot(gs[2, :])
        return fig, [ax1, ax2], ax3, ax4

    def _format_df(self, df: pd.DataFrame):
        df['dataset'] = df['dataset'].map({'jtd': 'JTD', 'pijama': 'PiJAMA'})
        total_duration = (
            df.groupby(['pianist', 'dataset'])
            ['duration']
            .sum()
            .reset_index(drop=False)
            .pivot_table(index='pianist', columns=['dataset'], values='duration', margins=True)
            .sort_values('All', ascending=True)
            .drop('All', axis=1)
            .sort_values('All', ascending=True, axis=1)
            .drop('All')
            .div(60)
        )  # pandas sucks
        total_count = (
            df.groupby(['pianist', 'dataset'])
            ['duration']
            .count()
            .reset_index(drop=False)
            .pivot_table(index='pianist', columns=['dataset'], values='duration', margins=False)
            .reindex(total_duration.index)
        )
        included_pianists = total_duration[total_duration.sum(axis=1) > self.MIN_MINUTES].index
        histdf = df[df['pianist'].isin(included_pianists)]
        histdf['duration'] /= 60
        # split dataframe
        splitdf = (
            df.groupby(['pianist', 'split'], sort=False)
            ['n_clips']
            .sum()
            .reset_index(drop=False)
            .pivot_table(index='pianist', columns=['split'], values='n_clips')
            .fillna(10)
            .assign(sum=lambda x: x.sum(axis=1))
            .sort_values(by="sum", ascending=False)
            .drop(columns=["sum"])
            .rename(columns={"test": "Validation", "validation": "Test", "train": "Train"})
            [["Train", "Validation", "Test"]]
        )
        return total_duration, total_count, histdf, splitdf

    def _create_bar_plots(self):
        for a, dat, xlab, ylab in zip(self.bar_ax, [self.count_df, self.duration_df], self.X_LABELS, self.Y_LABELS):
            dat.plot.barh(ax=a, width=0.75, **self.BAR_KWS)
            a.set(ylabel=ylab, xlabel=xlab)

    def _format_bar_ax(self):
        for a in self.bar_ax:
            sns.move_legend(a, loc="lower right", title="", **LEGEND_KWS)
            a.axhspan(4.5, 25.5, 0, 1, color=BLACK, alpha=ALPHA)
            a.text(0.6, 0.225, 'Included', color=WHITE, transform=a.transAxes, fontsize=FONTSIZE * 2)

    def _create_hist_plot(self):
        g = sns.histplot(data=self.hist_df, x='duration', hue='dataset', ax=self.hist_ax, **self.HIST_KWS)
        for (_, dataset), col_idx in zip(self.hist_df.groupby('dataset'), range(2)):
            g.axvline(dataset['duration'].median(), color=plt.get_cmap('tab10')(col_idx), **self.VLINE_KWS)

    def _create_split_plot(self):
        self.split_df.plot(kind='bar', ax=self.split_ax, width=0.5, color=self._split_colors, **self.BAR_KWS)

    def _format_split_ax(self):
        self.split_ax.set(xlabel="Pianist", ylabel="Number of 30-second clips")
        sns.move_legend(self.split_ax, loc="upper right", title="", **LEGEND_KWS)

    def _format_hist_ax(self):
        # handles, labels = self.hist_ax.get_legend_handles_labels()
        # self.hist_ax.get_legend().remove()
        # custom_dashed_line = lines.Line2D([0], [0], color=BLACK, linestyle=DASHED, lw=LINEWIDTH, label='')
        # self.hist_ax.legend(handles=[*handles, custom_dashed_line], labels=[*labels, 'Median'])
        sns.move_legend(self.hist_ax, loc="upper right", title='', **LEGEND_KWS)
        self.hist_ax.set(xlabel=self.X_LABELS[-1], ylabel=self.Y_LABELS[-1], xticks=[0, 5, 10, 15, 20], xlim=(0, 20))

    def _format_ax(self):
        self._format_bar_ax()
        self._format_hist_ax()
        self._format_split_ax()
        for a, griddy in zip([*self.bar_ax, self.hist_ax, self.split_ax], ['x', 'x', 'y', 'y']):
            plt.setp(a.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            a.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
            a.grid(axis=griddy, zorder=0, **GRID_KWS)

    def _format_fig(self):
        self.fig.tight_layout()
        for ax in [self.hist_ax, self.split_ax]:
            pos1 = ax.get_position()  # get the original position
            pos2 = [pos1.x0 - 0.075, pos1.y0, pos1.width + 0.075, pos1.height]
            ax.set_position(pos2)  # set a new position

    def _create_plot(self):
        self._create_bar_plots()
        self._create_hist_plot()
        self._create_split_plot()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/dataset_figures")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"dataset_recording_durations_count")
        save_fig_all_exts(fp, self.fig)


class HeatmapConfusionMatrices(BasePlot):
    HM_KWS = dict(
        cmap="Reds", linecolor=WHITE, square=True, annot=False,
        fmt='.0f', linewidths=LINEWIDTH // 2, vmin=0, vmax=100,
        cbar_kws=dict(label='Percentage of Instances', ticks=[0, 25, 50, 75, 100])
    )

    def __init__(self, confusion_matrices: list[np.array], matrix_names: list[str], pianist_mapping: dict, **kwargs):
        super().__init__(**kwargs)
        self.mat = confusion_matrices
        self.matrix_names = matrix_names
        self.pianist_mapping = pianist_mapping
        self.num_classes = len(self.pianist_mapping.keys())
        self.fig = plt.figure(figsize=(WIDTH, WIDTH))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[20, 20, 1])
        self.hm_axs = [self.fig.add_subplot(gs[x, y]) for x, y in zip([0, 0, 1, 1], [0, 1, 1, 0])]
        self.cbar_ax = self.fig.add_subplot(gs[:, 2])

    def _create_plot(self) -> None:
        for matrix, ax, name in zip(self.mat, self.hm_axs, self.matrix_names):
            sns.heatmap(data=matrix, ax=ax, cbar_ax=self.cbar_ax, **self.HM_KWS)
            ax.set(title=name.title())

    def _format_ax(self):
        for num, ax in enumerate(self.hm_axs):
            ax.set(xticks=range(self.num_classes), yticks=range(self.num_classes))
            # Set axis ticks correctly
            if num == 2:
                ax.set_xticks([i + 0.5 for i in ax.get_xticks()], self.pianist_mapping.values(), rotation=90)
            elif num == 1:
                pass
            elif num == 3:
                ax.set_xticks([i + 0.5 for i in ax.get_xticks()], self.pianist_mapping.values(), rotation=90)
                ax.set_yticks([i + 0.5 for i in ax.get_yticks()], self.pianist_mapping.values(), rotation=0)
            else:
                ax.set_yticks([i + 0.5 for i in ax.get_yticks()], self.pianist_mapping.values(), rotation=0)

            # Set axis and tick thickness
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
            # Invert y-axis for symmetry between x and y-axis ticks
            ax.invert_yaxis()
        # Make all spines visible on both axis and colorbar
        for ax in [*self.hm_axs, self.cbar_ax]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(BLACK)
                spine.set_linewidth(LINEWIDTH)

    def _format_fig(self):
        self.fig.supxlabel("Predicted pianist")
        self.fig.supylabel("Actual pianist")
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.000001)
        pos1 = self.cbar_ax.get_position()  # get the original position
        pos2 = [pos1.x0, pos1.y0 + 0.025, pos1.width, pos1.height * 0.94]
        self.cbar_ax.set_position(pos2)  # set a new position

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"heatmap_single_concept_prediction_accuracy")
        save_fig_all_exts(fp, self.fig)


class HeatmapCAVKernelSensitivity(BasePlot):
    def __init__(
            self,
            clip_path: str,
            sensitivity_array: np.array,
            clip_roll: np.array,
            cav_name: str = ""
    ):
        super().__init__()
        self.clip_name = os.path.sep.join(clip_path.split(os.path.sep)[-2:])
        self.clip_path = clip_path
        self.sensitivity_array = sensitivity_array
        self.clip_roll = clip_roll
        self.cav_name = cav_name
        # Matplotlib figure and axis
        self.clip_start, self.clip_end = self.parse_clip_start_stop(clip_path)
        self.fig = plt.figure(figsize=(WIDTH, WIDTH // 3))
        gs0 = GridSpec(1, 3, width_ratios=[1, 29, 1], hspace=0.01)
        self.cax1 = self.fig.add_subplot(gs0[0])
        self.ax = self.fig.add_subplot(gs0[1])
        self.cax2 = self.fig.add_subplot(gs0[2])

    @staticmethod
    def seconds_to_timedelta(seconds: int):
        # Format in %M:%S format
        td = timedelta(seconds=seconds)
        return str(td)[2:]

    @staticmethod
    def parse_clip_start_stop(clip_path: str):
        """Parse clip start and stop time from clip number using filepath"""
        clip_num = int(clip_path.split(os.path.sep)[-1].replace('clip_', '').replace('.mid', ''))
        clip_start = utils.CLIP_LENGTH * clip_num
        clip_end = clip_start + utils.CLIP_LENGTH
        return clip_start, clip_end

    def parse_human_readable_trackname(self):
        # Parse the start and end of the clip to a human-readable timestamp
        clip_start_fmt = self.seconds_to_timedelta(self.clip_start)
        clip_end_fmt = self.seconds_to_timedelta(self.clip_end)
        # Getting metadata json location from clip path
        fmt_fpath = os.path.sep.join(self.clip_path.replace("clips", "raw").split(os.path.sep)[:-1])
        json_path = f'{fmt_fpath}/metadata.json'
        # If we can't load the metadata for whatever reason, just return an empty string
        try:
            loaded = json.load(open(json_path, 'r'))
        except FileNotFoundError:
            return ""
        # Otherwise, return the formatted timestamp using the same format as our LIME plots
        else:
            return (
                f'{loaded["bandleader"]} — "{loaded["track_name"]}" from "{loaded["album_name"]}", '
                f'{loaded["recording_year"]}. CAV: {self.cav_name.title()}'
            )

    def _create_plot(self):
        # Plot the original piano roll
        self.clip_roll[self.clip_roll == 0] = np.nan  # set 0 values to NaN to make transparent
        sns.heatmap(
            self.clip_roll,
            ax=self.ax,
            cbar_ax=self.cax1,
            alpha=1.0,
            cbar_kws=dict(label='Velocity', pad=0.6),
            cmap="rocket",
            vmin=0,
            vmax=utils.MAX_VELOCITY
        )
        # Create the background for the heatmap by resizing to the same size as the piano roll
        background = resize(self.sensitivity_array, (utils.PIANO_KEYS, utils.CLIP_LENGTH * utils.FPS))
        sns.heatmap(
            background,
            ax=self.ax,
            cbar_ax=self.cax2,
            alpha=0.4,
            cmap=cmc.cork,
            center=0.,
            cbar_kws=dict(label='Score', pad=0.6)
        )

    def _format_fig(self):
        # Figure aesthetics
        self.fig.subplots_adjust(right=0.93, left=0.07, top=0.9, bottom=0.15)

    def _format_ax(self):
        self.cax1.yaxis.tick_left()
        self.cax1.yaxis.set_label_position("left")
        # Axis aesthetics
        self.ax.set(
            xticks=range(0, (utils.CLIP_LENGTH * utils.FPS + 1), utils.FPS * 5),
            xticklabels=[self.seconds_to_timedelta(i) for i in range(self.clip_start, self.clip_end + 1, 5)],
            yticks=range(0, utils.PIANO_KEYS, utils.OCTAVE),
            yticklabels=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, 12)],
            title=self.parse_human_readable_trackname(),
            xlabel="Time (seconds)",
            ylabel="Note",
        )
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=0)
        fmt_heatmap_axis(self.ax)

    def save_fig(self):
        # Get the directory to save the heatmap in
        cn = self.cav_name.lower().replace(" ", "_")
        di = os.path.join(
            utils.get_project_root(),
            "reports/figures/ablated_representations/cav_plots/clip_heatmaps/",
            cn
        )

        if not os.path.isdir(di):
            os.makedirs(di)
        # Format track and cav name
        tn = self.clip_name.lower().replace(os.path.sep, '_').replace('.mid', '')
        fp = os.path.join(di, f'{tn}_{cn}')
        # Save the figure
        save_fig_all_exts(fp, self.fig)


class LollipopPlotMaskedConceptsAccuracy(BarPlotMaskedConceptsAccuracy):
    LPOP_KWS = dict(width=0.025 * 0.9, height=1 * 0.9, zorder=100, edgecolor=BLACK, linewidth=LINEWIDTH)
    VLINE_KWS = dict(ymin=0, ymax=1, zorder=1000, linewidth=LINEWIDTH, linestyle=DASHED)

    def __init__(self, df, xvar: str = "track_acc", xlabel: str = "Accuracy (track-level)", **kwargs):
        super().__init__(df, xvar, xlabel, **kwargs)
        self.baseline_accuracy = kwargs.get('baseline_accuracy', None)

    def _create_plot(self) -> None:
        sns.barplot(
            self.df, y="concepts", x=self.xvar, width=0, edgecolor=BLACK,
            linewidth=LINEWIDTH ** 2, linestyle=LINESTYLE, ax=self.ax, zorder=10
        )

    def _add_lollipop(self, x: float = 0.015, lpop_space: float = 30):
        for idx, row in self.df.iterrows():
            concepts = sorted(row['concepts'].split(', '))
            for n_concept, concept in enumerate(concepts):
                color = CONCEPT_COLOR_MAPPING[concept]
                circ = mpatches.Ellipse(xy=(x + (n_concept / lpop_space), idx), facecolor=color, **self.LPOP_KWS)
                self.ax.add_patch(circ)

    def _add_legend(self):
        hands = []
        for color in CONCEPT_COLOR_MAPPING.values():
            patch = mpatches.Ellipse(xy=(0, 0), facecolor=color, **self.LPOP_KWS)
            hands.append(patch)
        labs = list(CONCEPT_COLOR_MAPPING.keys())
        self.ax.legend(hands, labs, loc="lower right", title='', **LEGEND_KWS)

    def _format_ax(self):
        self._add_lollipop()
        self._add_legend()
        # self._add_bar_labels(x_mult=0.02)
        ymin, ymax = self.ax.get_ylim()
        self.ax.set(
            ylabel="Musical dimension", xlabel=self.xlabel, xlim=(0., 1.), ylim=(ymin + 0.1, ymax - 0.1)
        )
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
        self.ax.grid(axis="x", zorder=0, **GRID_KWS)
        if self.baseline_accuracy is not None:
            self.ax.axvline(self.baseline_accuracy, **self.VLINE_KWS)
            self.ax.text(self.baseline_accuracy, -0.75, '$Baseline$', ha='center', va='bottom')

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"lollipop_masked_concept_{self.xvar}")
        save_fig_all_exts(fp, self.fig)


class HistPlotDatabaseNullDistributionCoefficients(BasePlot):
    KDE_KWS = dict(bw_adjust=.5, clip_on=False, fill=True, alpha=ALPHA, linewidth=1.5)
    VLINE_KWS = dict(ymin=0, ymax=1, zorder=100, linewidth=LINEWIDTH, linestyle=DASHED)

    def __init__(
            self,
            mel_null_dists: np.array,
            har_null_dists: np.array,
            mel_coefs: np.array,
            har_coefs: np.array,
            mel_ps: np.array,
            har_ps: np.array,
            class_mapping
    ):
        super().__init__()
        self.dists = [mel_null_dists, har_null_dists]
        self.coefs = [mel_coefs, har_coefs]
        self.ps = [mel_ps, har_ps]
        self.class_mapping = np.array(list(class_mapping.values()))
        self.fig, self.ax = plt.subplots(
            len(class_mapping.values()), 2,
            figsize=(WIDTH, WIDTH),
            sharex=True, sharey=True
        )

    def _create_plot(self):
        sorters = np.argsort(self.coefs[0])[::-1]
        class_mapping = self.class_mapping[sorters]
        for feature_num, (dists, coefs, ps, color) in enumerate(zip(self.dists, self.coefs, self.ps, RGB)):
            dists = dists[:, sorters].T
            coefs = coefs[sorters]
            ps = ps[sorters]
            feature_axs = self.ax[:, feature_num]
            for ax, dist, coef, p, cls in zip(feature_axs.flatten(), dists, coefs, ps, class_mapping):
                sns.kdeplot(dist, ax=ax, color=color, **self.KDE_KWS)
                ax.axvline(coef, color=color, **self.VLINE_KWS)
                if p == 1.:
                    p_txt = rf'$p$ = 1.'
                elif p >= 0.001:
                    p_txt = rf'$p$ = {str(round(p, 3))[1:]}'
                else:
                    p_txt = rf'$p$ < .001'
                p_txt = f'{cls} {p_txt}'
                ax.text(
                    1., 0.5, p_txt, transform=ax.transAxes, fontsize=FONTSIZE / 1.5,
                    ha='right', va='center', bbox=dict(
                        facecolor='wheat', boxstyle='round', edgecolor=BLACK,
                        linewidth=LINEWIDTH / 1.5
                    )
                )

    def _format_ax(self):
        for num, title in enumerate(["Melody", "Harmony"]):
            self.ax[0, num].set_title(title)
            self.ax[-1, num].set_xlabel('Coefficient ($r$)')
            self.ax[10, num].text(-0.05, 0.5, "Count", ha='center', va='center', transform=self.ax[10, num].transAxes,
                                  rotation=90)

        for ax in self.ax.flatten():
            for spine in ["left", "right", "top"]:
                ax.spines[spine].set_visible(False)
            ax.set(yticks=[], ylabel="")
            ax.axvline(0, 0, 1, color=BLACK, linewidth=LINEWIDTH)
            # Set axis and tick thickness
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)

    def _format_fig(self):
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.05)

    def save_fig(self, outpath: str):
        save_fig_all_exts(outpath, self.fig)


class BarPlotWhiteboxFeatureCounts(BasePlot):
    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, zorder=10)

    def __init__(
            self,
            feature_counts: np.array,
            mel_features_names: np.array,
            har_features_names: np.array,
            k: int = 20
    ):
        super().__init__()
        # Number of features to get for each set
        self.k = k
        # Total appearances of each feature across all recordings
        self.feature_counts_summed = feature_counts.sum(axis=0)
        # Arrays of names for all features
        self.mel_names = mel_features_names
        self.har_names = har_features_names
        assert len(self.mel_names) + len(self.har_names) == len(self.feature_counts_summed)
        # Dataframe for plotting
        self.df = self._format_df(None)
        self.fig, self.ax = plt.subplots(1, 2, figsize=(WIDTH, WIDTH // 2), sharex=False, sharey=False)

    @staticmethod
    def map_ngram_to_pitches(intervals: str) -> str:
        pitch_set = [0]
        current_pitch = 0
        for interval in eval(intervals):
            current_pitch += interval
            pitch_set.append(current_pitch)
        return str(tuple(pitch_set))

    def _format_df(self, _):
        def format_feature(idxs, names) -> tuple[np.ndarray, np.ndarray]:
            features_counts = self.feature_counts_summed[idxs]
            topk_idxs = np.argsort(features_counts)[::-1][:self.k]
            topk_counts = features_counts[topk_idxs]
            topk_names = np.array(names)[topk_idxs]
            return topk_counts, topk_names

        # Arrays of idxs for each feature set
        mel_idxs = np.arange(0, len(self.mel_names))
        har_idxs = np.arange(len(mel_idxs), len(mel_idxs) + len(self.har_names))
        # Should be the same as the number of features we have counts for
        assert len(mel_idxs) + len(har_idxs) == len(self.feature_counts_summed)
        # Formatting for each feature set
        mel_topk_counts, mel_topk_names = format_feature(mel_idxs, self.mel_names)
        har_topk_counts, har_topk_names = format_feature(har_idxs, self.har_names)
        # Additional formatting
        mel_topk_names = [self.map_ngram_to_pitches(ng) for ng in mel_topk_names]
        har_topk_names = [str((0, *eval(ch))) for ch in har_topk_names]
        # Creating the dataframe with given column names
        return pd.DataFrame({
            'Melody_name': mel_topk_names,
            'Melody_count': mel_topk_counts,
            'Harmony_name': har_topk_names,
            'Harmony_count': har_topk_counts
        })

    def _create_plot(self):
        for ax, feature in zip(self.ax.flatten(), ["Melody", "Harmony"]):
            sns.barplot(
                data=self.df, x=f"{feature}_name", y=f"{feature}_count",
                ax=ax, color=CONCEPT_COLOR_MAPPING[feature], **self.BAR_KWS
            )
            ax.set(
                title=feature, xlabel="Feature", ylabel="Count",
                ylim=self.ax[0].get_ylim(), yticks=self.ax[0].get_yticks()
            )

    def _format_ax(self):
        for ax in self.ax.flatten():
            # Set axis and tick thickness
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
            ax.grid(axis='y', zorder=0, **GRID_KWS)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        # Get the directory to save the heatmap in
        di = os.path.join(
            utils.get_project_root(),
            "reports/figures/whitebox",
        )
        if not os.path.isdir(di):
            os.makedirs(di)
        fp = os.path.join(di, f'barplot_feature_counts')
        # Save the figure
        save_fig_all_exts(fp, self.fig)


class BarPlotConceptClassAccuracy(BasePlot):
    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, zorder=10)

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(4, 1, figsize=(WIDTH, WIDTH), sharex=True, sharey=True)

    def _format_df(self, df):
        df['concepts'] = df['concepts'].str.title()
        df['n_concepts'] = df['concepts'].apply(lambda x: len(x.split('+')))
        return df[df['n_concepts'] == 1]

    def _create_plot(self):
        for ax, (concept, color) in zip(self.ax.flatten(), CONCEPT_COLOR_MAPPING.items()):
            sns.barplot(
                data=self.df[self.df["concepts"] == concept],
                ax=ax,
                x='pianist',
                y='class_acc',
                color=color,
                **self.BAR_KWS
            )
            ax.set_title(concept)

    def _format_ax(self):
        for ax in self.ax.flatten():
            # Set axis and tick thickness
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
            ax.grid(axis='y', zorder=0, **GRID_KWS)
            ax.set(xlabel='', ylabel='', ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1.0],
                   yticklabels=[0, 25, 50, 75, 100])
        self.ax[-1].set_xticks(self.ax[-1].get_xticks(), self.ax[-1].get_xticklabels(), rotation=90)

    def _format_fig(self):
        self.fig.supxlabel('Pianist')
        self.fig.supylabel('Recordings Classified Correctly (%)')
        self.fig.tight_layout()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"barplot_concept_class_accuracy")
        save_fig_all_exts(fp, self.fig)


class BarPlotWhiteboxDomainImportance(BasePlot):
    RESULTS_LOC = os.path.join(utils.get_project_root(), 'reports/figures/whitebox/permutation')
    PALETTE = [CONCEPT_COLOR_MAPPING[con] for con in ["Harmony", "Melody"]]
    BAR_KWS = dict(legend=True, zorder=10, palette=PALETTE, edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE)
    ERROR_KWS = dict(
        x=[-0.2, 0.2, 0.8, 1.2, 1.8, 2.2], lw=LINEWIDTH, color=BLACK,
        capsize=8, zorder=10000, elinewidth=LINEWIDTH, ls='none'
    )

    def __init__(self):
        super().__init__()
        self.df = self._format_df(None)
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=2, figsize=(WIDTH, WIDTH // 3),
            sharex=True, sharey=False
        )

    def _format_df(self, _):
        all_dfs = []
        for md in ["lr", "svm", "rf"]:
            csv_loc = os.path.join(self.RESULTS_LOC, f'domain_importance_{md}.csv')
            if not os.path.exists(csv_loc):
                fake = [{"feature": val} | {col: np.nan for col in ["mean", "std", "high", "low"]} for val in
                        ["Harmony", "Harmony (Bootstrapped)", "Melody", "Melody (Bootstrapped)"]]
                read = pd.DataFrame(fake)
            else:
                read = pd.read_csv(csv_loc, index_col=0)
            read['md_type'] = md.upper()
            all_dfs.append(read)
        df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
        df['bootstrapped'] = df['feature'].str.contains("Boot")
        df['feature'] = df['feature'].str.replace(" (Bootstrapped)", "")
        return df

    def _create_plot(self) -> None:
        for a, is_boot in zip(self.ax.flatten(), [False, True]):
            subset = self.df[self.df['bootstrapped'] == is_boot]
            sns.barplot(subset, ax=a, x='md_type', y='mean', hue='feature', **self.BAR_KWS)
            a.errorbar(y=subset['mean'], yerr=subset['std'], **self.ERROR_KWS)

    def _format_ax(self):
        for ax, tit in zip(self.ax.flatten(), ['All features', "Bootstrapped samples of features"]):
            ax.set(
                ylabel='Mean feature importance', xlabel='Model type', title=tit, ylim=(0, ax.get_ylim()[1] * 1.5)
            )
            ax.grid(axis='y', zorder=-1, **GRID_KWS)
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH)
            ax.tick_params(axis='both', width=TICKWIDTH)
            sns.move_legend(ax, loc="upper right", title="Feature", **LEGEND_KWS)

    def _format_fig(self) -> None:
        self.fig.tight_layout()

    def save_fig(self):
        fp = os.path.join(self.RESULTS_LOC, 'barplot_domain_importance')
        save_fig_all_exts(fp, self.fig)


class HeatmapMelodyExtraction(BasePlot):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots(2, 1, figsize=(WIDTH, WIDTH // 3), sharex=True, sharey=True)
        self.pianorolls = self._get_pianorolls()

    @staticmethod
    def _get_pianorolls():
        from pretty_midi import PrettyMIDI

        names = ['example_polyphony', 'example_polyphony_skylined']
        for name in names:
            path = os.path.join(utils.get_project_root(), 'references', name + '.mid')
            poly = PrettyMIDI(path)
            poly_roll = np.flipud(poly.get_piano_roll(100))
            poly_roll[poly_roll != 0] = 255
            poly_roll[poly_roll == 0] = np.nan
            yield poly_roll

    def _create_plot(self):
        for roll, ax in zip(self.pianorolls, self.ax.flatten()):
            ax.imshow(roll, aspect='auto')

    def _format_ax(self):
        for ax_, tit in zip(self.ax.flatten(), ['Original', "Melody (Skyline)"]):
            ax_.set(
                xlabel='Time', ylabel='Pitch', title=tit, yticklabels=[], xticklabels=[],
                xticks=[], yticks=[], ylim=(utils.MIDI_OFFSET + utils.PIANO_KEYS, utils.MIDI_OFFSET),
            )
            plt.setp(ax_.spines.values(), linewidth=LINEWIDTH)

    def _format_fig(self) -> None:
        self.fig.tight_layout()

    def save_fig(self):
        out = os.path.join(utils.get_project_root(), 'reports/figures/melody_extraction_demo')
        save_fig_all_exts(out, self.fig)


class LinePlotWhiteboxAccuracyN(BasePlot):
    ROOT = os.path.join(utils.get_project_root(), 'references/whitebox/optimization_results_at_n')
    MAX_N_RESULTS = ['2', '23', '234', '2345', '23456', '234567']
    MIN_N_RESULTS = ['234567', '34567', '4567', '567', '67', '7']
    LINE_KWS = dict(linewidth=LINEWIDTH * 2, linestyle=LINESTYLE, color=BLACK)

    def __init__(self, _=None):
        super().__init__()
        self.df = self._format_df(None)
        self.fig, self.ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(WIDTH, WIDTH // 2))

    def _format_df(self, _):
        def load_max_acc(ex):
            path = os.path.join(self.ROOT, f'20class_80min_lr_harmony+melody_{ex}.csv')
            assert os.path.exists(path)
            return pd.read_csv(path)['accuracy'].max()

        res = []
        for ext in self.MAX_N_RESULTS:
            res.append({
                'n': '\n'.join([str(eval(i) + 1) for i in ext]),
                'acc': load_max_acc(ext),
                'experiment': 'max_n'
            })
        for ext in self.MIN_N_RESULTS:
            res.append({
                'n': '\n'.join([str(eval(i) + 1) for i in ext]),
                'acc': load_max_acc(ext),
                'experiment': 'min_n'
            })

        res = pd.DataFrame(res).sort_values(by=['experiment', 'n']).reset_index(drop=True)
        return res

    def _create_plot(self):
        for ax, exp in zip(self.ax.flatten(), ['max_n', 'min_n']):
            sub = self.df[self.df['experiment'] == exp].reset_index(drop=True)
            ax.plot(sub['n'], sub['acc'], **self.LINE_KWS)

    def _format_ax(self):
        for ax in self.ax.flatten():
            ax.set(xlabel='$n\in\{x\}$', ylabel='Optimized validation accuracy (LR)', )
            # ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels(), rotation=90)
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
            ax.grid(axis='y', zorder=0, **GRID_KWS)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        out = os.path.join(utils.get_project_root(), 'reports/figures/whitebox/optimization_results_at_n')
        save_fig_all_exts(out, self.fig)


class BigramplotPCAFeatureCount(BasePlot):
    TEXT_KWS = dict(ha='center', va='center')
    CIRCLE_KWS = dict(color=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, fill=False, alpha=ALPHA)
    LINE_KWS = dict(color=BLACK, linewidth=LINEWIDTH, linestyle=DOTTED, alpha=ALPHA)
    N_SLICES = 8
    N_FEATURES_PER_SLICE = 5
    N_PERFORMERS_PER_SLICE = 20

    def __init__(self, df: pd.DataFrame, n_components_to_plot: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_components_to_plot = n_components_to_plot
        self.df = self._format_df(df)
        self.fig, self.ax = plt.subplots(
            1, n_components_to_plot // 2, figsize=(WIDTH, WIDTH // 2), sharex=True, sharey=True
        )

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df['component'] < self.n_components_to_plot].reset_index(drop=True)

    @staticmethod
    def _scale_loading(loading: pd.Series) -> pd.Series:
        mm = MinMaxScaler(feature_range=(-1., 1.))
        return mm.fit_transform(loading.to_numpy().reshape(-1, 1)).flatten()

    def get_region(self, x_coords, y_coords):
        if isinstance(x_coords, pd.Series):
            x_coords = x_coords.to_numpy()
        if isinstance(y_coords, pd.Series):
            y_coords = y_coords.to_numpy()
        # get angle
        angle = np.arctan2(x_coords, y_coords)
        # normalize
        angle = np.mod(angle, 2 * np.pi)
        # express as a region
        return (angle // (2 * np.pi / self.N_SLICES)).astype(int)

    def get_topk_idxs(self, subset_df: pd.DataFrame, n_to_plot: int) -> list[int]:
        subset_df['region'] = self.get_region(subset_df['loading_x'], subset_df['loading_y'])
        subset_df['magnitude'] = subset_df['loading_x'].abs() + subset_df['loading_y'].abs()
        all_idxs = []
        for idx, sub in subset_df.groupby('region'):
            all_idxs.extend(sub.sort_values(by='magnitude', ascending=False).index[:n_to_plot].tolist())
        return all_idxs

    def add_performer_text(self, row: pd.Series, ax: plt.Axes) -> plt.Text:
        cls = row['label'].split(' ')[0][0] + row['label'].split(' ')[1][0]
        return ax.text(row['loading_x'], row['loading_y'], cls, fontsize=FONTSIZE * 1.5, **self.TEXT_KWS)

    def add_feature_text(self, row: pd.Series, ax: plt.Axes) -> plt.Text:
        # We need to add some objects to the plot or else `adjust_text` doesn't seem to work
        ax.arrow(0, 0, row['loading_x'], row['loading_y'], alpha=0.)
        txt = m21_utils.intervals_to_pitches(row['label'])
        return ax.text(row['loading_x'], row['loading_y'], txt, fontsize=FONTSIZE / 1.5, **self.TEXT_KWS)

    def _create_plot(self):
        for ax, comp1_idx, comp2_idx in zip(
                self.ax.flatten(),
                range(0, self.n_components_to_plot, 2),  # 0, 2, 4, 6, ...
                range(1, self.n_components_to_plot + 1, 2)  # 1, 3, 5, 7, ...
        ):
            texts = []
            for typ, n_to_plot, txt_func in zip(
                    ['performer', 'feature'],
                    [self.N_PERFORMERS_PER_SLICE, self.N_FEATURES_PER_SLICE],
                    [self.add_performer_text, self.add_feature_text]
            ):
                comp1 = (
                    self.df[(self.df['component'] == comp1_idx) & (self.df['type'] == typ)]
                    .reset_index(drop=True)
                    .sort_values(by='label')
                )
                comp2 = (
                    self.df[(self.df['component'] == comp2_idx) & (self.df['type'] == typ)]
                    .reset_index(drop=True)
                    .sort_values(by='label')
                )
                comp1['loading_y'] = self._scale_loading(comp2['loading'])
                comp1['loading_x'] = self._scale_loading(comp1['loading'])
                to_plot = comp1[comp1.index.isin(self.get_topk_idxs(comp1, n_to_plot))]
                for idx, row in to_plot.iterrows():
                    txt = txt_func(row, ax)
                    texts.append(txt)
            adjust_text(texts, ax=ax, expand=(1.3, 1.3), arrowprops=dict(arrowstyle='->', color=RED))

    def _format_ax(self):
        for num, ax in zip(range(1, self.n_components_to_plot, 2), self.ax.flatten()):
            # We're plotting even-numbered PCAs on x-axis, odd-numbered on y
            ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel=f'PCA {num}', ylabel=f'PCA {num + 1}')
            plt.setp(ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
            ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
            # ax.grid(axis='both', zorder=0, **GRID_KWS)
            circle1 = plt.Circle((0, 0), 1., **self.CIRCLE_KWS)
            ax.add_patch(circle1)
            ax.axhline(0, 0, 1, **self.LINE_KWS)
            ax.axvline(0, 0, 1, **self.LINE_KWS)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self, out_path: str):
        save_fig_all_exts(out_path, self.fig)


if __name__ == "__main__":
    # This just generates some plots that we can't easily create otherwise
    blp = LinePlotWhiteboxAccuracyN()
    blp.create_plot()
    blp.save_fig()
    met = utils.get_track_metadata(utils.get_pianist_names())
    bp = BarPlotDatasetDurationCount(met)
    bp.create_plot()
    bp.save_fig()
    bp2 = BarPlotWhiteboxDomainImportance()
    bp2.create_plot()
    bp2.save_fig()
    hm = HeatmapMelodyExtraction()
    hm.create_plot()
    hm.save_fig()
