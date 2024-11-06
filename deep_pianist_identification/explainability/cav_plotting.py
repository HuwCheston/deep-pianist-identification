#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules for plotting CAVs, including interactive plots"""

import os
from copy import deepcopy
from itertools import product

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
from loguru import logger
from pretty_midi import PrettyMIDI, note_number_to_name
from skimage.transform import resize
from tqdm import tqdm

import deep_pianist_identification.plotting as plotting
from deep_pianist_identification import utils
from deep_pianist_identification.extractors import get_piano_roll, ExtractorError

__all__ = ["BIRTH_YEARS", "HeatmapPianistCAV", "HeatmapCAVKernelSensitivity", "HeatmapCAVKernelSensitivityInteractive"]

# Array of pianist birth years and indexes needed to sort these in order
BIRTH_YEARS = np.genfromtxt(
    os.path.join(utils.get_project_root(), 'references/data_splits/20class_80min/birth_years.txt'), dtype=int
)


class HeatmapPianistCAV(plotting.BasePlot):
    def __init__(self, corr_df: pd.DataFrame, cav_type: str = "Voicings"):
        super().__init__()
        self.df = self._format_df(corr_df)
        self.cav_type = cav_type
        self.fig, self.ax = plt.subplots(1, 1, figsize=(plotting.WIDTH, plotting.WIDTH))

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        sorters = np.argsort(BIRTH_YEARS)
        corr_df = df.reindex(df.index[sorters])
        corr_df.index = [f"{n} ({y})" for n, y in zip(corr_df.index, BIRTH_YEARS[sorters])]
        return corr_df

    def _create_plot(self) -> None:
        return sns.heatmap(
            self.df,
            square=True,
            cmap="vlag",
            linecolor=plotting.WHITE,
            linewidth=plotting.LINEWIDTH // 2,
            ax=self.ax,
            center=0.,
            cbar_kws=dict(
                label="Coefficient $r$",
                shrink=0.75
            )
        )

    def _format_ax(self):
        # Set aesthetics for both colorbar and main axis
        plotting.fmt_heatmap_axis(self.ax)
        self.ax.set(
            xlabel=f"{self.cav_type} CAV (→→→ $increasing$ $complexity$ →→→)",
            ylabel="Pianist (→→→ $increasing$ $birth$ $year$ →→→)"
        )

    def _format_fig(self):
        self.fig.tight_layout()

    def save_fig(self):
        fp = os.path.join(
            utils.get_project_root(),
            "reports/figures",
            f"cav_plots/{self.cav_type.lower()}",
            "correlation_heatmap"
        )
        for fmt in ["png", "svg"]:
            self.fig.savefig(fp + "." + fmt, format=fmt, facecolor=plotting.WHITE)


class HeatmapCAVKernelSensitivity(plotting.BasePlot):
    CAV_DIM = 512

    def __init__(
            self,
            clip_name: str,
            cav: np.array,
            encoder,
            extractor_cls,
            cav_name: str = "",
            heatmap_size: tuple = None,
            kernel_size: tuple = None,
            draw_rectangle: bool = False,
            cav_type: str = "Voicings"
    ):
        super().__init__()
        # We use this model to generate embeddings for the piano roll
        self.encoder = encoder
        # We use this class to extract the "concept" piano rolls from the MIDI
        self.extractor_cls = extractor_cls
        # Path to the midi file for the clip
        self.clip_name = clip_name
        self.clip_path = os.path.join(utils.get_project_root(), 'data/clips', clip_name)
        assert os.path.isfile(self.clip_path), f"Could not find input MIDI {clip_name}"
        # Convert the input MIDI to PrettyMIDI and piano roll format
        self.clip_midi = PrettyMIDI(self.clip_path)
        self.clip_roll = get_piano_roll(
            [(i.start, i.end, i.pitch, i.velocity) for i in self.clip_midi.instruments[0].notes]
        )
        # Array of coefficients obtained for the CAV
        self.cav = cav
        assert len(self.cav) == self.CAV_DIM, f"Input CAV did not have expected shape of {self.CAV_DIM} dims"
        self.cav_name = cav_name  # Defaults to empty string
        # Size of the heatmap that will be generated (H, W)
        self.heatmap_size = heatmap_size if heatmap_size is not None else (65, 26)
        # Size of the kernel that will be slid over the MIDI (H, W) = (semitones, seconds)
        self.kernel_size = kernel_size if kernel_size is not None else (24, 5)
        self.kernels = self.get_kernels()
        # Get the original sensitivity using the entire piano roll
        self.original_sensitivity = self.compute_sensitivity(self.clip_midi)
        # Variables that will hold the biggest change in sensitivity values and kernel location
        self.largest_sensitivity_change = 0
        self.largest_sensitivity_kernel = (0, 0)
        self.draw_rectangle = draw_rectangle
        # Will hold the array of kernel sensitivities
        self.sensitivity_array = None
        # Matplotlib figure and axis
        self.fig, self.ax = plt.subplots(1, 1, figsize=(plotting.WIDTH, plotting.WIDTH // 3))
        self.cav_type = cav_type

    def get_kernels(self):
        # Unpack the kernel and heatmap size tuples for readability
        k_h, k_w = self.kernel_size
        heat_h, heat_w = self.heatmap_size
        # Get array of heights
        heights = np.linspace(0, utils.PIANO_KEYS - k_h, heat_h)
        assert all([int(i) == i for i in heights]), "All height values should be ints"
        # Get array of widths (can possibly be floats?)
        widths = np.linspace(0, utils.CLIP_LENGTH - k_w, heat_w)
        # Get the combinations of both arrays and return
        return list(product(heights, widths))

    def compute_sensitivity(self, midi: PrettyMIDI) -> float:
        try:
            # Extract the required piano roll concept
            extractor = self.extractor_cls(midi).roll
            # Set note velocities (channel dimension) to 1 for notes that are played
            extractor[extractor != 0] = 1.
        except ExtractorError as e:
            logger.warning(f"Failed to compute kernel sensitivity with error {e}")
            return np.nan
        else:
            # Convert to tensor
            tens = torch.tensor(extractor).to(utils.DEVICE)
            # Add the missing dimensions back in
            x_roll = tens.unsqueeze(0).unsqueeze(0)
            # Forwards through the disentangled model
            with torch.no_grad():
                x_embed = self.encoder(x_roll).squeeze(1).cpu().numpy()
            # Return CAV sensitivity
            return x_embed[0] @ self.cav

    def _sensitivity_for_one_kernel(self, h_low: float, w_low: float) -> float:
        # Unpack kernel size for readability
        k_h, k_w = self.kernel_size
        # Get upper boundaries of the kernel
        h_hi = h_low + k_h
        w_hi = w_low + k_w
        # Deep copy so we don't modify the underlying data (maybe not needed?)
        temp = deepcopy(self.clip_midi)
        # Remove notes that are contained within the kernel
        temp.instruments[0].notes = [
            n for n in temp.instruments[0].notes
            if not (h_low <= n.pitch - utils.MIDI_OFFSET <= h_hi)
               or not any([(w_low <= n.start <= w_hi), (w_low <= n.end <= w_hi)])
        ]
        # Compute the sensitivity with this masked MIDI object
        sensitivity = self.compute_sensitivity(temp)
        # Is this the largest change in CAV sensitivity we've seen?
        change = self.original_sensitivity - sensitivity
        # If so, update the variables that are keeping track of this
        if change >= self.largest_sensitivity_change:
            self.largest_sensitivity_change = change
            self.largest_sensitivity_kernel = (h_low, w_low)
        return sensitivity

    def compute_kernel_sensitivities(self) -> np.array:
        all_sensitivities = []
        for h_low, w_low in tqdm(self.kernels, desc='Computing kernel sensitivities: '):
            kernel_sen = self._sensitivity_for_one_kernel(h_low, w_low)
            all_sensitivities.append(kernel_sen)
        self.sensitivity_array = np.array(all_sensitivities).reshape(self.heatmap_size)
        return self.sensitivity_array

    def create_plot(self):
        # If we haven't yet created the sensitivity array, do this now
        if self.sensitivity_array is None:
            self.compute_kernel_sensitivities()
        # Create the plot and format the axis and figure objects
        self._create_plot()
        self._format_ax()
        self._format_fig()
        return self.fig, self.ax

    def _create_plot(self):
        # Create the background for the heatmap by resizing to the same size as the piano roll
        background = resize(self.sensitivity_array, (utils.PIANO_KEYS, utils.CLIP_LENGTH * utils.FPS))
        sns.heatmap(
            background,
            ax=self.ax,
            alpha=0.7,
            cmap="mako",
            cbar_kws={'label': 'Sensitivity', 'pad': -0.07}
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
            vmax=127
        )
        # Draw a rectangle around the kernel that leads to the greatest loss in CAV sensitivity
        if self.draw_rectangle:
            self._draw_most_sensitive_kernel()

    def _format_fig(self):
        # Figure aesthetics
        self.fig.suptitle(f"{self.cav_type} CAV: {self.cav_name}")
        self.fig.supxlabel("Time (seconds)")
        self.fig.supylabel("Note")
        self.fig.tight_layout()

    def _draw_most_sensitive_kernel(self):
        # Unpack tuples
        k_h, k_w = self.kernel_size
        ls_h, ls_w = self.largest_sensitivity_kernel
        # Set the time dimension correctly (seconds to frames)
        ls_w *= utils.FPS
        k_w *= utils.FPS
        # Create the patch and add onto the axis
        rect = patches.Rectangle(
            (ls_w, ls_h), k_w, k_h,
            linewidth=plotting.LINEWIDTH,
            edgecolor=plotting.RED,
            facecolor='none'
        )
        self.ax.add_patch(rect)

    def _format_ax(self):
        # Axis aesthetics
        self.ax.set(
            xticks=range(0, utils.CLIP_LENGTH * utils.FPS, utils.FPS),
            xticklabels=range(0, utils.CLIP_LENGTH, 1),
            yticks=range(0, utils.PIANO_KEYS, 12),
            title=self.clip_name,
            xlabel="",
            ylabel="",
            yticklabels=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, 12)],
        )
        self.ax.text(
            2950, 86, f"Original Sensitivity = {round(self.original_sensitivity, 3)})",
            rotation=0,
            va='top',
            ha='right',
            fontsize=plotting.FONTSIZE,
            bbox=dict(
                facecolor=plotting.WHITE,
                alpha=1.0,
                linewidth=plotting.LINEWIDTH,
                edgecolor=plotting.BLACK
            ),
        )
        plotting.fmt_heatmap_axis(self.ax)

    def save_fig(self):
        # Get the directory to save the heatmap in
        di = os.path.join(
            utils.get_project_root(),
            "reports/figures",
            f"cav_plots/{self.cav_type.lower()}/clip_heatmaps"
        )
        if not os.path.isdir(di):
            os.makedirs(di)
        # Format track and cav name
        tn = self.clip_name.lower().replace(os.path.sep, '_')
        cn = self.cav_name.lower().replace(" ", "_")
        fp = os.path.join(di, f'{tn}_{cn}.png')
        # Save the figure
        self.fig.savefig(fp, format="png", facecolor=plotting.WHITE)


class HeatmapCAVKernelSensitivityInteractive(HeatmapCAVKernelSensitivity):
    CAV_DIM = 512

    def __init__(
            self,
            clip_name: str,
            cav: np.array,
            encoder: torch.nn.Module,
            extractor_cls,
            cav_name: str = "",
            heatmap_size: tuple = None,
            kernel_size: tuple = None,
            cav_idx: int = 1,
            cav_type: str = "Voicings"
    ):
        super().__init__(
            clip_name,
            cav,
            encoder,
            extractor_cls,
            cav_name,
            heatmap_size,
            kernel_size,
            draw_rectangle=False,  # won't do anything so just set it to false
            cav_type=cav_type
        )
        self.fig, self.ax = None, None
        plt.close()
        self.cav_idx = cav_idx

    def get_hovertext(self):
        # Create hover text, including indices for non-NaN values
        mask = np.isnan(self.clip_roll)
        hover_text = np.full(self.clip_roll.shape, '', dtype=object)  # Initialize with empty strings
        for i in range(self.clip_roll.shape[0]):
            for j in range(self.clip_roll.shape[1]):
                if not mask[i, j]:
                    hover_text[i, j] = f'Note: {note_number_to_name(i)}<br>Time: {(j / utils.FPS):.2f}'
        return hover_text

    @staticmethod
    def create_plotly_colorscale_from_seaborn(seaborn_cmap: str, n_colors: int = 256) -> list:
        palette = sns.color_palette(seaborn_cmap, n_colors=n_colors)
        colors = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1)' for r, g, b in palette]
        return list(zip(np.linspace(0, 1, len(colors)), colors))

    def _create_plot(self):
        background = resize(self.sensitivity_array, (utils.PIANO_KEYS, utils.CLIP_LENGTH * utils.FPS))
        self.clip_roll[self.clip_roll == 0] = np.nan
        f1 = go.Heatmap(
            z=self.clip_roll,
            showscale=True,
            hoverinfo='text',
            text=self.get_hovertext(),
            colorscale=self.create_plotly_colorscale_from_seaborn("rocket"),
            colorbar=dict(title='Velocity', x=1.05),
            zmin=0,
            zmax=127
        )
        f2 = go.Heatmap(
            z=background,
            showscale=True,
            hoverinfo='skip',
            opacity=1.0,
            colorscale=self.create_plotly_colorscale_from_seaborn("mako"),
            colorbar=dict(title='Sensitivity', x=1.15),
            reversescale=False
        )
        self.fig = go.Figure(data=[f2, f1])

    def _format_ax(self):
        self.fig.update_layout(
            xaxis=dict(
                tickmode='array',
                ticktext=list(range(0, utils.CLIP_LENGTH, 5)),
                tickvals=list(range(0, utils.CLIP_LENGTH * utils.FPS, 500)),
                title='Time (Seconds)',
                showgrid=False,
                fixedrange=True,
                showline=True,
                linewidth=plotting.LINEWIDTH,
                linecolor=plotting.BLACK,
                mirror=True
            ),
            yaxis=dict(
                tickmode='array',
                ticktext=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, 12)],
                tickvals=list(range(0, utils.PIANO_KEYS, 12)),
                title='Note',
                showgrid=False,
                # gridwidth=plotting.LINEWIDTH // 2,
                # gridcolor=plotting.BLACK,
                fixedrange=True,
                showline=True,
                linewidth=plotting.LINEWIDTH,
                linecolor=plotting.BLACK,
                mirror=True
            ),
            height=400,
            width=1000,
            margin=dict(l=30, r=30, t=30, b=30),
            plot_bgcolor=plotting.WHITE
        )

    def _format_fig(self):
        # Everything is done in self._format_ax so no need for this function here
        pass

    def save_fig(self):
        # This just saves the plot JSON information so that it can be loaded in JavaScript for our web application
        # create a folder for this clip
        fname = self.clip_name.replace(os.path.sep, '_').split('.')[0]
        fp = os.path.join(
            utils.get_project_root(),
            f"reports/figures/cav_plots/{self.cav_type.lower()}/clip_heatmaps_interactive",
            fname
        )
        if not os.path.isdir(fp):
            os.makedirs(fp)
        # Write the HTML object
        self.fig.write_html(os.path.join(fp, "plot.html"), include_plotlyjs="cdn")
        # Write the MIDI object if it doesn't already exist
        midi_f = os.path.join(fp, 'midi.mid')
        if not os.path.isfile(midi_f):
            # Truncate notes outside the range
            notes = [n for n in self.clip_midi.instruments[0].notes if n.start <= utils.CLIP_LENGTH]
            self.clip_midi.instruments[0].notes = notes
            self.clip_midi.write(midi_f)
        # Save the plot layout to JSON
        layout_f = os.path.join(fp, 'layout.json')
        if not os.path.isfile(layout_f):
            with open(layout_f, 'w') as f:
                f.write(self.fig.layout.to_json() + '\n')
        # Save the piano roll
        roll_f = os.path.join(fp, 'roll.json')
        if not os.path.isfile(roll_f):
            with open(roll_f, 'w') as f:
                f.write(self.fig.data[1].to_json() + '\n')
        # Save the CAV heatmap json
        cav_name = f'cav_{str(self.cav_idx).zfill(3)}.json'
        cav_f = os.path.join(fp, cav_name)
        # We do a little hacking to add the overall sensitivity value to the plotly json
        # It should still read ok when rendered with our javascript stuff, however
        j = self.fig.data[0].to_json()
        cav_js = j[:1] + f'"overall": {self.original_sensitivity:.2f},' + j[1:]
        with open(cav_f, 'w') as f:
            f.write(cav_js + '\n')
