#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes, functions, and variables."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from music21.chord import Chord
from music21.note import Note
from music21.stream import Stream
from pretty_midi import note_name_to_number, note_number_to_name

from deep_pianist_identification import utils

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

LINEWIDTH = 2
LINESTYLE = '-'
TICKWIDTH = 3
MARKERSCALE = 1.6
MARKERS = ['o', 's', 'D']
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

SAVE_KWS = dict(format='png', facecolor=WHITE)

# Keyword arguments to use when applying a grid to a plot
GRID_KWS = dict(color=BLACK, alpha=ALPHA, lw=LINEWIDTH / 2, ls=LINESTYLE)

N_BOOT = 10000
N_BINS = 50
PLOT_AFTER_N_EPOCHS = 5


def fmt_heatmap_axis(heatmap_ax: plt.Axes):
    for a in [heatmap_ax, *heatmap_ax.figure.axes]:
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_color(BLACK)
            spine.set_linewidth(LINEWIDTH)
        plt.setp(a.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        a.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
    heatmap_ax.invert_yaxis()


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
    def __init__(self, confusion_mat: np.ndarray, pianist_mapping: dict, **kwargs):
        super().__init__(**kwargs)
        self.mat = confusion_mat
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
            xlabel="Predicted pianist", ylabel="Actual pianist",
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


class BarPlotMaskedConceptsAccuracy(BasePlot):
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = self._format_df(df.dropna(subset="concepts"))
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 2))

    def _format_df(self, df):
        pd.options.mode.chained_assignment = None
        counter = lambda x: x.count('+') + 1 if '+' in x else 0
        df["n_concepts"] = df['concepts'].apply(counter)
        sort_and_title = lambda x: ', '.join(sorted([i.title() for i in x.split('+')]))
        df["concepts"] = df["concepts"].apply(sort_and_title)
        pd.options.mode.chained_assignment = "warn"
        return (
            df.sort_values(by=["n_concepts", "track_acc"], ascending=False)
            .reset_index(drop=True)
        )

    def _create_plot(self) -> None:
        return sns.barplot(
            self.df, y="concepts", x="track_acc", hue="n_concepts", palette="tab10",
            edgecolor=BLACK, linewidth=LINEWIDTH,
            linestyle=LINESTYLE, ax=self.ax, legend=False, zorder=10
        )

    def _format_ax(self):
        self.ax.set(ylabel="Concepts", xlabel="Validation accuracy (track)")
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
        self.ax.grid(axis="x", zorder=0, **GRID_KWS)

    def _format_fig(self):
        self.fig.tight_layout()


class StripplotTopKFeatures(BasePlot):
    """Creates a strip/scatter plot showing the top/bottom-K features for a given performer

    Examples:
        >>> t = dict(
        >>>     top_ors=np.array([1.2, 1.15, 1.10, 1.05, 1.01]),    # The top K odds ratios for this performer
        >>>     bottom_ors=np.array([0.8, 0.85, 0.9, 0.95, 0.975]),    # The bottom K odds ratios
        >>>     top_std=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),    # The corresponding top K standard deviations
        >>>     bottom_std=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),    # Bottom k standard deviations
        >>>     # These arrays contain the names of the n-grams
        >>>     top_names=np.array(["M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]"]),
        >>>     bottom_names=np.array(["M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]"])
        >>> )
        >>> plo = StripplotTopKFeatures(t, "Testy McTestFace")
        >>> plo.create_plot()
        >>> plo.save_fig()

    """

    ERROR_KWS = dict(lw=LINEWIDTH, color=BLACK, capsize=8, zorder=0, elinewidth=LINEWIDTH, ls='none')
    SCATTER_KWS = dict(legend=False, s=200, edgecolor=BLACK, zorder=10)

    def __init__(
            self,
            concept_dict: dict,
            pianist_name: str = "",
            concept_name: str = ""
    ):
        super().__init__()
        self.concept_dict = concept_dict
        self.pianist_name = pianist_name
        self.concept_name = concept_name
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 2))

    def _add_notation(self, ngram: str, y: float) -> None:
        """Adds a given feature/ngram as notation onto the axis with a given y coordinate"""
        # This just holds all the note instances we'll create
        notes = Stream()
        # We'll express all notation with relation to middle C
        notes.append(Note(note_number_to_name(utils.MIDDLE_C)))
        # Evaluate the ngram as a list of integers then iterate through all notes
        for n in eval(ngram):
            # Get the previous note as a name ("C5") and convert to a MIDI number
            prev_note = str(notes[-1].pitch)
            prev_number = note_name_to_number(prev_note)
            # Add the current interval class to the previous note number
            next_number = prev_number + n
            # If we're plotting chords, we want to express all chords on the same octave
            if self.concept_name == "harmony":
                # So keep subtracting an octave from the note until it's within the same octave as middle C
                while next_number >= utils.MIDDLE_C + utils.OCTAVE:
                    next_number -= utils.OCTAVE
            # Music21 expects note names, so convert to a name and append to the list
            next_note = Note(note_number_to_name(next_number))
            notes.append(next_note)
        # Converting the stream to a chord and then back will remove all rhythmic information
        if self.concept_name == "harmony":
            notes = Stream(Chord(notes))

        # Export the notation as an image
        try:
            notes.write("musicxml.png", "tmp.png")
        except OSError:
            # When running in a screen instance, this line may fail with a cryptic QT error
            # Something like `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`
            # The solution is to set the environment variable `QT_QPA_PLATFORM=offscreen` and retry
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            notes.write("musicxml.png", "tmp.png")

        # Read the image in to matplotlib and add to the axes
        im = plt.imread("tmp-1.png")
        imagebox = OffsetImage(im, zoom=0.4)
        ab = AnnotationBbox(imagebox, (1.2, y), xycoords='axes fraction', frameon=False)
        self.ax.add_artist(ab)
        # Remove the temporary files we've created
        os.remove("tmp-1.png")
        os.remove("tmp.musicxml")

    def add_notation(self, direction: str, ngram_names: list[str]):
        """Adds all features as notation onto the axis at a given position"""
        # Hard-coding our y-axis positions is maybe not the best solution for scalability, but works for now
        if direction == 'bottom':
            ys = [0.95, 0.855, 0.765, 0.675, 0.585]
        else:
            ys = [0.41, 0.32, 0.23, 0.14, 0.05]
        for y, ng in zip(ys, ngram_names):
            self._add_notation(ng, y)

    def _create_plot(self):
        for direction in ['bottom', 'top']:
            sorters = np.array(range(len(self.concept_dict[f"{direction}_ors"])))
            if direction == 'top':
                sorters = sorters[::-1]
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
                **self.SCATTER_KWS, ax=self.ax
            )
            self.add_notation(direction, names)
            # All this line does is add an extra empty "value" to separate the top and bottom k features better visually
            if direction == 'bottom':
                sns.scatterplot(x=np.array([1.]), s=0, y=np.array(['']), ax=self.ax)

    def _format_ax(self):
        """Setting plot aesthetics on an axis-level basis"""
        self.ax.tick_params(right=True)
        self.ax.set(title=f'{self.pianist_name}, {self.concept_name} features', xlabel='Odds ratio', ylabel='Feature')
        self.ax.grid(axis='x', zorder=0, **GRID_KWS)
        self.ax.axhline(5, 0, 1, linewidth=LINEWIDTH, color=BLACK, alpha=ALPHA, ls='dashed')
        self.ax.axvline(1, 0, 1, linewidth=LINEWIDTH, color=BLACK)
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH)
        self.ax.tick_params(axis='both', width=TICKWIDTH)

    def _format_fig(self):
        """Setting plot aeshetics on a figure-level basis"""
        self.fig.subplots_adjust(left=0.1, top=0.925, bottom=0.1, right=0.75)

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), 'reports/figures', 'whitebox/performer_strip_plots')
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(
            fold, f'topk_stripplot_{self.pianist_name.lower().replace(" ", "_")}_{self.concept_name}.png'
        )
        self.fig.savefig(fp, **SAVE_KWS)
        plt.close('all')


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
        self.fig.savefig(output_loc, **SAVE_KWS)
        plt.close('all')


if __name__ == "__main__":
    # Test the StripplotTopKFeatures plotting class
    tmp = dict(
        top_ors=np.array([1.2, 1.15, 1.10, 1.05, 1.01]),
        bottom_ors=np.array([0.8, 0.85, 0.9, 0.95, 0.975]),
        top_std=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        bottom_std=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        top_names=np.array(["M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]"]),
        bottom_names=np.array(["M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]", "M_[1, 1, 1]"])
    )
    sp = StripplotTopKFeatures(tmp, "Tempy McTempface", "melody")
    sp.create_plot()
