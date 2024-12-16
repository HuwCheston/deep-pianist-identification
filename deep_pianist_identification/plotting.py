#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes, functions, and variables."""

import json
import os
from datetime import timedelta
from urllib.error import HTTPError, URLError

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from music21.chord import Chord
from music21.clef import bestClef
from music21.meter import TimeSignature
from music21.note import Note
from music21.stream import Part, Measure, Score
from pretty_midi import note_name_to_number, note_number_to_name
from skimage import io
from skimage.transform import resize

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

DOTTED = 'dotted'
DASHED = 'dashed'

# Remove URL where images of all pianists are stored
IMG_LOC = "https://raw.githubusercontent.com/HuwCheston/Jazz-Trio-Database/refs/heads/main/references/images/musicians"
# Used for saving PNG images with the correct background
SAVE_KWS = dict(format='png', facecolor=WHITE)
# Keyword arguments to use when applying a grid to a plot
GRID_KWS = dict(color=BLACK, alpha=ALPHA, lw=LINEWIDTH / 2, ls=LINESTYLE)
# Used when adding a legend to an axis
LEGEND_KWS = dict(frameon=True, framealpha=1, edgecolor=BLACK)

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
        fp = os.path.join(fold, f"heatmap_confusion_matrix_{self.title}.png")
        self.fig.savefig(fp, **SAVE_KWS)


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

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"barplot_masked_concept_{self.xvar}.png")
        self.fig.savefig(fp, **SAVE_KWS)


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
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 2.25))

    def ic_to_music21_note(self, ic: int, prev_note):
        # Get the previous note as a name ("C5") and convert to a MIDI number
        prev_number = note_name_to_number(str(prev_note))
        # Add the current interval class to the previous note number
        next_number = prev_number + ic
        # If we're plotting chords, we want to express all chords on the same octave
        if self.concept_name == "harmony":
            # So keep subtracting an octave from the note until it's within the same octave as middle C
            while next_number >= utils.MIDDLE_C + utils.OCTAVE:
                next_number -= utils.OCTAVE
        # Music21 expects note names, so convert to a name and append to the list
        return Note(note_number_to_name(next_number), quarterLength=1)

    def _create_music21(self, ngram: str) -> Score:
        # These hold all the note instances we'll create
        score, part, measure = Score(), Part(), Measure()
        # Evaluate the n-gram as a list of integers
        ngram_list = eval(ngram)
        ts_numerator = len(ngram_list) if self.concept_name == "melody" else 1
        # Time signature with same number of quarter notes as ICs in n-gram
        ts = TimeSignature(f"{ts_numerator}/4")
        ts.style.hideObjectOnPrint = True  # hides the time signature
        # Add the time signature to the measure
        measure.append(ts)
        # We'll express all notation with relation to middle C
        measure.append(Note(note_number_to_name(utils.MIDDLE_C), quarterLength=1))
        # Iterate over all the ICs in the n-gram
        for ic in ngram_list:
            # Add the note to the measure
            measure.append(self.ic_to_music21_note(ic, measure[-1].pitch))
        # Converting the measure to a chord and then back will remove all rhythmic information
        if self.concept_name == "harmony":
            measure = Measure(Chord([i for i in measure if not isinstance(i, TimeSignature)]))
        # Add the measure to the part and the part to the score
        part.append(measure)
        score.append(part)
        # Get the best clef for the part recursively
        cl = bestClef(part, recurse=True)
        part.append(cl)
        # Remove the final bar line from the part
        part[-1].rightBarline = None
        return score

    def _add_notation(self, ngram: str, y: float) -> None:
        """Adds a given feature/ngram as notation onto the axis with a given y coordinate"""
        notes = self._create_music21(ngram)
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
        image = plt.imread("tmp-1.png")
        imagebox = OffsetImage(image, zoom=0.25)
        ab = AnnotationBbox(imagebox, (1.025, y), xycoords='axes fraction', frameon=False, box_alignment=(0, 0.5))
        self.ax.add_artist(ab)
        # Remove the temporary files we've created
        os.remove("tmp-1.png")
        os.remove("tmp.musicxml")

    def add_notation(self, direction: str, ngram_names: list[str]):
        """Adds all features as notation onto the axis at a given position"""
        # Hard-coding our y-axis positions is maybe not the best solution for scalability, but works for now
        if direction == 'bottom':
            ys = [0.95, 0.85, 0.75, 0.65, 0.55]
        else:
            ys = [0.45, 0.35, 0.25, 0.15, 0.05]
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
                **self.SCATTER_KWS, ax=self.ax, label=f"{direction.title()}-5",
            )
            self.add_notation(direction, names)
        self.ax.legend(**LEGEND_KWS, loc='lower left')

    def _add_performer_image(self):
        gh = f"{IMG_LOC}/{self.pianist_name.lower().replace(' ', '_')}.png"
        try:
            image = io.imread(gh)
        except (HTTPError, URLError):
            logger.warning(f"Couldn't find performer image for {self.pianist_name} at {gh}!")
            return
        imagebox = OffsetImage(image, zoom=1.0)
        ab = AnnotationBbox(imagebox, (0.95, 0.95), xycoords='axes fraction', frameon=False, box_alignment=(1., 1.))
        self.ax.add_artist(ab)

    def _format_ax(self):
        """Setting plot aesthetics on an axis-level basis"""
        self._add_performer_image()
        self.ax.tick_params(right=True)
        self.ax.set(title=f'{self.pianist_name}, {self.concept_name} features', xlabel='Odds ratio', ylabel='Feature')
        self.ax.grid(axis='x', zorder=0, **GRID_KWS)
        self.ax.axhline(4.5, 0, 1, linewidth=LINEWIDTH, color=BLACK, alpha=ALPHA, ls=DASHED)
        self.ax.axvline(1, 0, 1, linewidth=LINEWIDTH, color=BLACK)
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH)
        self.ax.tick_params(axis='both', width=TICKWIDTH)

    def _format_fig(self):
        """Setting plot aeshetics on a figure-level basis"""
        self.fig.subplots_adjust(left=0.11, top=0.935, bottom=0.11, right=0.775)

    def save_fig(self, output_dir: str):
        fp = os.path.join(
            output_dir,
            f'topk_stripplot_{self.pianist_name.lower().replace(" ", "_")}_{self.concept_name}.png'
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


class BarPlotWhiteboxDatabaseCoefficients(BasePlot):
    """Bar plot showing the performer correlations obtained from both source databases for a whitebox model"""
    BAR_KWS = dict(palette="tab10", edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, legend=False, zorder=10)
    ERROR_KWS = dict(lw=LINEWIDTH, color=BLACK, capsize=8, zorder=100, elinewidth=LINEWIDTH, ls='none')
    X_PADDING = 0.01

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
            sns.barplot(data=sub, x='corr', y='pianist', hue='pianist', ax=ax, **self.BAR_KWS)
            for idx, row in sub.iterrows():
                x = row['corr'] + self.X_PADDING if row['corr'] > 0 else row['corr'] - self.X_PADDING
                ax.text(x, idx, row['sig'], ha='center')

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
        self.fig.savefig(outpath, **SAVE_KWS)


class HeatmapCAVSensitivity(BasePlot):
    HEATMAP_KWS = dict(square=True, cmap="vlag", center=0.5, linecolor=WHITE, linewidth=LINEWIDTH // 2, vmin=0, vmax=1)

    def __init__(
            self,
            sensitivity_matrix: np.array,
            class_mapping: dict,
            cav_names: np.array,
            performer_birth_years: np.array,
            significance_asterisks: np.array,
            sensitivity_type: str = 'TCAV Sign Count',
    ):
        super().__init__()
        self.class_mapping = class_mapping
        self.sensitivity_type = sensitivity_type
        self.cav_names = cav_names
        self.performer_birth_years = performer_birth_years
        self.sorters = np.argsort(self.performer_birth_years)[::-1]
        # TODO: check that this is sorting significance asterisks properly
        self.significance_asterisks = significance_asterisks[self.sorters, :]
        self.df = self._format_df(sensitivity_matrix)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))

    def _format_df(self, sensitivity_matrix: np.array):
        # Allow for passing in a sensitivity matrix with fewer columns than CAVs
        cols = self.cav_names
        if sensitivity_matrix.shape[1] < len(cols):
            cols = cols[:sensitivity_matrix.shape[1]]
        # Create the dataframe
        res_df = pd.DataFrame(sensitivity_matrix, columns=cols)
        # Add performer birth year into the column
        res_df['performer'] = [f'{p}, ({b})' for p, b in zip(self.class_mapping.values(), self.performer_birth_years)]
        # Sort by birth year and set performer to index
        return res_df.reindex(self.sorters).set_index('performer')

    def _create_plot(self):
        _ = sns.heatmap(
            self.df, ax=self.ax, annot=self.significance_asterisks, fmt="", **self.HEATMAP_KWS,
            cbar_kws=dict(
                label=self.sensitivity_type, shrink=0.75, ticks=[0., 0.25, 0.5, 0.75, 1.], location="right",
            )
        )

    def _format_ax(self):
        fmt_heatmap_axis(self.ax)
        # Set axis and tick thickness
        self.ax.set(
            xlabel="Harmony CAV (→→→ $increasing$ $complexity$ →→→)",
            ylabel="Pianist (→→→ $increasing$ $birth$ $year$ →→→)"
        )
        self.ax.invert_yaxis()

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"cav_plots/{self.sensitivity_type.replace(' ', '_').lower()}_heatmap.png")
        self.fig.savefig(fp, **SAVE_KWS)


class HeatmapCAVPairwiseCorrelation(BasePlot):
    HEATMAP_KWS = dict(
        cmap="vlag", linecolor=WHITE, square=True, annot=False, fmt='.0f', linewidths=LINEWIDTH // 2, vmin=-1., vmax=1.,
        cbar_kws=dict(label='Correlation ($r$)', location="right", shrink=0.75, ticks=[-1., -0.5, 0., 0.5, 1.]),
        center=0.
    )

    def __init__(
            self,
            cav_matrix: pd.DataFrame,
            cav_mapping: list
    ):
        super().__init__()
        self.df = pd.DataFrame(cav_matrix, columns=cav_mapping).corr()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))

    def _create_plot(self):
        return sns.heatmap(self.df, ax=self.ax, **self.HEATMAP_KWS)

    def _format_ax(self):
        fmt_heatmap_axis(self.ax)

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"cav_plots/cav_sensitivity_pairwise_correlation_heatmap.png")
        self.fig.savefig(fp, **SAVE_KWS)


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
        fp = os.path.join(fold, f"dataset_recording_durations_count.png")
        self.fig.savefig(fp, **SAVE_KWS)


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
        fp = os.path.join(fold, f"heatmap_single_concept_prediction_accuracy.png")
        self.fig.savefig(fp, **SAVE_KWS)


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
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 3))

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
        tmp = os.path.sep.join(self.clip_path.split(os.path.sep)[-3:-1])
        clip_start_fmt = self.seconds_to_timedelta(self.clip_start)
        clip_end_fmt = self.seconds_to_timedelta(self.clip_end)
        json_path = os.path.join(utils.get_project_root(), 'data/raw', tmp, 'metadata.json')
        loaded = json.load(open(json_path, 'r'))
        return (f'{loaded["bandleader"]} — "{loaded["track_name"]}" ({clip_start_fmt}—{clip_end_fmt}) '
                f'from ${loaded["album_name"]}$, {loaded["recording_year"]} (CAV: {self.cav_name})')

    def _create_plot(self):
        # Create the background for the heatmap by resizing to the same size as the piano roll
        background = resize(self.sensitivity_array, (utils.PIANO_KEYS, utils.CLIP_LENGTH * utils.FPS))
        sns.heatmap(
            background,
            ax=self.ax,
            alpha=0.7,
            cmap="mako",
            cbar_kws={'label': 'Sensitivity (1 = original)', 'pad': -0.07}
        )
        # Plot the original piano roll
        self.clip_roll[self.clip_roll == 0] = np.nan  # set 0 values to NaN to make transparent
        sns.heatmap(
            self.clip_roll,
            ax=self.ax,
            alpha=1.0,
            cbar_kws={'label': 'Velocity', 'pad': 0.01},
            cmap="rocket",
            vmin=0,
            vmax=utils.MAX_VELOCITY
        )

    def _format_fig(self):
        # Figure aesthetics
        self.fig.subplots_adjust(right=1.07, left=0.05, top=0.93, bottom=0.20)

    def _format_ax(self):
        # Axis aesthetics
        self.ax.set(
            xticks=range(0, utils.CLIP_LENGTH * utils.FPS, utils.FPS),
            xticklabels=[self.seconds_to_timedelta(i) for i in range(self.clip_start, self.clip_end, 1)],
            yticks=range(0, utils.PIANO_KEYS, 12),
            title=self.parse_human_readable_trackname(),
            xlabel="Time (seconds)",
            ylabel="Note",
            yticklabels=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, 12)],
        )
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
        fp = os.path.join(di, f'{tn}_{cn}.png')
        # Save the figure
        self.fig.savefig(fp, **SAVE_KWS)
        # Close the figure
        plt.close(self.fig)


class BarPlotCAVMixedLMCoefficients(BasePlot):
    BAR_KWS = dict(edgecolor=BLACK, linewidth=LINEWIDTH, linestyle=LINESTYLE, zorder=10, width=0.75, legend=False)
    ERROR_KWS = dict(lw=LINEWIDTH, color=BLACK, capsize=8, zorder=100, elinewidth=LINEWIDTH, ls='none')

    def __init__(
            self,
            mlm_coefs: np.array,
            mlm_cis: np.array,
            cav_names: list[str]
    ):
        super().__init__()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH // 2))
        df_tmp = pd.DataFrame([mlm_coefs, mlm_cis[:, 0], mlm_cis[:, 1], cav_names[:len(mlm_coefs)]])
        self.df = self._format_df(df_tmp)

    def _format_df(self, df: pd.DataFrame):
        mlm_df = df.T
        mlm_df.columns = ['coef', 'low', 'high', 'cav']
        mlm_df['sig'] = ~((mlm_df['low'] <= 0) & (0 <= mlm_df['high']))
        mlm_df['low'] = mlm_df['coef'] - mlm_df['low']
        mlm_df['high'] = mlm_df['high'] - mlm_df['coef']
        return mlm_df

    def _create_plot(self):
        sns.barplot(data=self.df, x='cav', y='coef', hue='sig', ax=self.ax, **self.BAR_KWS)
        self.ax.errorbar(x=self.df['cav'], y=self.df['coef'], yerr=[self.df['low'], self.df['high']], **self.ERROR_KWS)

    def _format_ax(self):
        self.ax.axhline(0, 0, 1, linewidth=LINEWIDTH, color=BLACK, linestyle=LINESTYLE)
        plt.setp(self.ax.spines.values(), linewidth=LINEWIDTH, color=BLACK)
        self.ax.tick_params(axis='both', width=TICKWIDTH, color=BLACK)
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=45, ha='right')
        self.ax.grid(axis='y', zorder=0, **GRID_KWS)
        self.ax.set(xlabel='Concept', ylabel=r'Coefficient $\beta$')

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        # Get the directory to save the heatmap in
        di = os.path.join(
            utils.get_project_root(),
            "reports/figures/ablated_representations/cav_plots",
        )
        if not os.path.isdir(di):
            os.makedirs(di)
        fp = os.path.join(di, f'barplot_mixedlm_coefficients.png')
        # Save the figure
        self.fig.savefig(fp, **SAVE_KWS)
        # Close the figure
        plt.close(self.fig)


class LollipopPlotMaskedConceptsAccuracy(BarPlotMaskedConceptsAccuracy):
    COLOR_MAPPING = {"Melody": "#ff0000ff", "Harmony": "#00ff00ff", "Rhythm": "#0000ffff", "Dynamics": "#ff9900ff"}
    LPOP_KWS = dict(width=0.025 * 0.9, height=1 * 0.9, zorder=10000, edgecolor=BLACK, linewidth=LINEWIDTH)

    def __init__(self, df, xvar: str = "track_acc", xlabel: str = "Accuracy (track-level)", **kwargs):
        super().__init__(df, xvar, xlabel, **kwargs)

    def _create_plot(self) -> None:
        sns.barplot(
            self.df, y="concepts", x=self.xvar, width=0, edgecolor=BLACK,
            linewidth=LINEWIDTH ** 2, linestyle=LINESTYLE, ax=self.ax, zorder=10
        )

    def _add_lollipop(self, x: float = 0.015, lpop_space: float = 30):
        for idx, row in self.df.iterrows():
            concepts = sorted(row['concepts'].split(', '))
            for n_concept, concept in enumerate(concepts):
                color = self.COLOR_MAPPING[concept]
                circ = mpatches.Ellipse(xy=(x + (n_concept / lpop_space), idx), facecolor=color, **self.LPOP_KWS)
                self.ax.add_patch(circ)

    def _add_legend(self):
        hands = []
        for color in self.COLOR_MAPPING.values():
            patch = mpatches.Ellipse(xy=(0, 0), facecolor=color, **self.LPOP_KWS)
            hands.append(patch)
        labs = list(self.COLOR_MAPPING.keys())
        self.ax.legend(hands, labs, loc="lower right", title='Dimensions', **LEGEND_KWS)

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

    def save_fig(self):
        fold = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        fp = os.path.join(fold, f"lollipop_masked_concept_{self.xvar}.png")
        self.fig.savefig(fp, **SAVE_KWS)


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
        self.fig.savefig(outpath, **SAVE_KWS)


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
        # Creating the dataframe with given column names
        return pd.DataFrame({
            'Melody_name': mel_topk_names,
            'Melody_count': mel_topk_counts,
            'Harmony_name': har_topk_names,
            'Harmony_count': har_topk_counts
        })

    def _create_plot(self):
        for ax, feature, color in zip(self.ax.flatten(), ["Melody", "Harmony"], RGB):
            sns.barplot(data=self.df, x=f"{feature}_name", y=f"{feature}_count", ax=ax, color=color, **self.BAR_KWS)
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
        fp = os.path.join(di, f'barplot_feature_counts.png')
        # Save the figure
        self.fig.savefig(fp, **SAVE_KWS)
        # Close the figure
        plt.close(self.fig)


if __name__ == "__main__":
    met = utils.get_track_metadata(utils.get_pianist_names())
    bp = BarPlotDatasetDurationCount(met)
    bp.create_plot()
    bp.save_fig()
