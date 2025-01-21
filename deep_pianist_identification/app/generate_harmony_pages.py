#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create plots and HTML for pianist harmonic progression usage"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from pretty_midi import PrettyMIDI, note_number_to_name
from seaborn import color_palette
from skimage.transform import resize
from tqdm import tqdm

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.explainability.cav_utils import CAV_MAPPING, CAVKernelSlider, CAV
from deep_pianist_identification.extractors import ExtractorError

CAV_NAME_TO_IDX = {m: n for n, m in enumerate(CAV_MAPPING, 1)}  # A mapping of CAV names to chapter numbers
INPUT_DIR = os.path.join(utils.get_project_root(), 'reports/figures/ablated_representations/cav_plots')
N_CONCEPTS_PER_PERFORMER = 5  # we'll only show the 5 most sensitive concepts for every performer
N_CLIPS_PER_CONCEPT = 10  # we'll show up to this number of clips for each concept
# TODO: we should probably just use whichever clip has the maximum value for the 1st experiment
AGG_FUNC = np.mean  # used to aggregate the results from all experiments to a single value
HEATMAP_KERNEL = (24., 2.5)  # (semitones, seconds)
HEATMAP_STRIDE = (2., 2.)  # (semitones, seconds)


class HeatmapCAVKernelSensitivityInteractive(plotting.HeatmapCAVKernelSensitivity):
    """Creates an interactive heatmap using Plotly, combining masked concept sensitivity and a MIDI piano roll"""

    def __init__(
            self,
            clip_path: str,
            sensitivity_array: np.array,
            clip_roll: np.array,
            cav_name: str
    ):
        super().__init__(clip_path, sensitivity_array, clip_roll, cav_name)
        self.clip_name = os.path.sep.join(clip_path.split(os.path.sep)[-2:])
        plt.close('all')  # if we don't close the figure we end up with warnings in the command line
        self.fig, self.ax = None, None
        self.cav_idx = CAV_NAME_TO_IDX[cav_name]

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
        palette = color_palette(seaborn_cmap, n_colors=n_colors)
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
            zmax=utils.MAX_VELOCITY
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
                ticktext=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, utils.OCTAVE)],
                tickvals=list(range(0, utils.PIANO_KEYS, utils.OCTAVE)),
                title='Note',
                showgrid=False,
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
        # Everything is taken care of for this plot in _format_ax already
        # However, we still need to override the base method so we don't call matplotlib functions on plotly objs
        pass

    def save_fig(self):
        def _saver(filepath: str, obj) -> None:
            if not os.path.isfile(filepath):
                with open(filepath, 'w') as f:
                    f.write(obj.to_json() + '\n')

        fname = "_".join(self.clip_path.split(os.path.sep)[-3:]).replace('.mid', '')
        fp = os.path.join(utils.get_project_root(), "app/assets/plots/cav_sensitivity")
        # Save the plot layout
        _saver(os.path.join(fp, 'layout', fname + '_layout.json'), self.fig.layout)
        # Save the piano roll
        _saver(os.path.join(fp, 'roll', fname + '_roll.json'), self.fig.data[1])
        # Save the CAV heatmap json
        _saver(os.path.join(fp, 'sensitivity', fname + f'_cav_{str(self.cav_idx).zfill(3)}.json'), self.fig.data[0])


def load_cavs(cav_path: str = None) -> list[CAV]:
    """Load CAVs from hard disk and raise an error if they've not been created yet"""
    # Use the default file path if not provided
    if cav_path is None:
        cav_path = os.path.join(utils.get_project_root(), 'reports/figures/ablated_representations/cav_plots/cavs.p')
    try:
        cavs = pickle.load(open(cav_path, 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not load CAVs at {cav_path}. Create them by running `create_harmony_cavs.py`.')
    else:
        # Should have exactly 20 CAVs
        assert len(cavs) == len(CAV_MAPPING)
        return cavs


def parse_input(input_fpath: str, extension: str = None) -> str:
    """Parses an input filepath with given extension and checks that it exists"""
    # Add the extension to the filepath if we've provided it
    if extension is not None:
        if not input_fpath.endswith(extension):
            input_fpath += f'.{extension}'
    path = os.path.join(INPUT_DIR, input_fpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {input_fpath} inside {INPUT_DIR}. Have you run `create_harmony_cavs.py`?"
        )
    return path


def get_most_sensitive_concepts(n_concepts: int = N_CONCEPTS_PER_PERFORMER) -> dict:
    """Get the name of the N concepts that each pianist is most sensitive to"""
    sign_counts = pd.read_csv(parse_input("tcav_sign_count.csv", ), index_col=0)
    # Remove the birth years from our index
    sign_counts.index = sign_counts.index.str.split(',').str[0]
    # This creates a dictionary of Pianist: [Concept1, Concept2...] mappings
    return {
        str(idx).split(',')[0]: row.sort_values(ascending=True).index.to_list()[-n_concepts:][::-1]
        for idx, row in sign_counts.sort_index().iterrows()
    }


def get_clip_paths(pianist_sensitive_concepts: dict, n_clips: int = N_CLIPS_PER_CONCEPT):
    """Get paths to the X most sensitive clips for each pianist's most sensitive N concepts"""

    def proc(pian: str, sensitivities: dict) -> dict:
        # This requires some string processing to get the performer name from the filepath
        pianist_sensitivities_for_concept = {
            i: v for i, v in sensitivities.items()
            if i.split("/")[1].split('-')[0][:-1] == pian.split(' ')[1].lower()
        }
        # Sort the clip filepaths by aggregating the corresponding sensitivity scores
        # TODO: we should probably just use whichever clip has the maximum value from the 1st experiment
        return {
            k: AGG_FUNC(v) for k, v in
            sorted(
                pianist_sensitivities_for_concept.items(),
                key=lambda item: AGG_FUNC(item[1]),
                reverse=True
            )[:n_clips]
        }

    # Load in the JSON
    with open(parse_input("clip_sensitivities.json"), 'r') as f:
        clip_sensitivities = json.load(f)
    results_dict = {}

    # This gives us pianist name, [concept1, concept2 ...] in order of sensitivity
    for pianist, concepts in pianist_sensitive_concepts.items():
        results_dict[pianist] = {}
        # Iterating through individual concepts
        for concept in concepts:
            # Get paths to the X most sensitive clips by the current pianist to the current concept
            results_dict[pianist][concept] = proc(pianist, clip_sensitivities[concept])
    return results_dict


def generate_clip_metadata(clip_name: str, sensitivity: float, concept: str) -> dict:
    """Generates a JSON of metadata for a pianist to be saved in ./app/assets/metadata/cav_sensitivity"""
    # Get the path for the original track metadata.json inside ./data/raw and make sure it exists
    metadata_path = os.path.join(
        utils.get_project_root(),
        'data/raw',
        os.path.sep.join(clip_name.split('/')[:-1]),
        'metadata.json'
    )
    assert os.path.exists(metadata_path)
    # Load in the metadata as a dictionary
    with open(metadata_path, 'r') as f:
        metadata_json = json.load(f)
    # This is the name under which we'll store all assets related to this clip
    asset_path = f"{clip_name.replace('/', '_').replace('.mid', '')}_cav_{str(CAV_NAME_TO_IDX[concept]).zfill(3)}"
    # Create the dictionary and append to the list for this concept
    return dict(
        track_name=metadata_json["track_name"],
        album_name=metadata_json["album_name"],
        recording_year=metadata_json["recording_year"],
        asset_path=asset_path,
        original_sensitivity=sensitivity,  # maybe this isn't necessary?
    )


def dump_metadata_json(pianist: str, metadata_json: dict) -> None:
    """Dumps a metadata JSON to the required location"""
    # This is the path where we'll save the metadata JSON for this pianist
    outpath = os.path.join(
        utils.get_project_root(),
        'app/assets/metadata/cav_sensitivity',
        f'{pianist.replace(" ", "_").lower()}.json'
    )
    # Dump the JSON in the required location
    with open(outpath, 'w') as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=4)


def copy_midi_to_assets_folder(clip_path: str):
    """Copies MIDI clip from ./data/clips to ./app/assets/midi/cav_sensitivity"""
    file_path = clip_path.replace('/', '_').replace('.mid', '_midi.mid')  # using a flat structure here
    # Defining input and output files
    in_path = os.path.join(utils.get_project_root(), 'data/clips', clip_path)
    out_path = os.path.join(utils.get_project_root(), 'app/assets/midi/cav_sensitivity', file_path)
    # Skip writing anything if we've already done so
    if os.path.exists(out_path):
        return
    # Load the MIDI as a PrettyMIDI object
    pm = PrettyMIDI(in_path)
    # Truncate the notes to keep only those within the boundaries of the clip
    pm.instruments[0].notes = [i for i in pm.instruments[0].notes if i.end <= utils.CLIP_LENGTH]
    # Write the PrettyMIDI object to the assets folder
    pm.write(out_path)


def heatmap_exists(clip_path: str, cav_name: str) -> bool:
    """Returns `True` if all outputs for a given clip/CAV heatmap have been created, `False` if otherwise"""
    # This is the name we use to store outputs for this clip/CAV combination
    outpath = "_".join(clip_path.split(os.path.sep)[-3:]).replace('.mid', '')
    # This is where our output plots lie
    root = os.path.join(utils.get_project_root(), "app/assets/plots/cav_sensitivity")
    # These are all the outputs we're expecting to see for the current input
    fpaths = [
        os.path.join(root, 'layout', outpath + '_layout.json'),
        os.path.join(root, 'roll', outpath + '_roll.json'),
        os.path.join(root, 'sensitivity', outpath + f'_cav_{str(CAV_NAME_TO_IDX[cav_name]).zfill(3)}.json'),
    ]
    # True if all paths exist on the local file system, false if otherwise
    return all([os.path.isfile(fp) for fp in fpaths])


def generate_clip_heatmap(clip_path: str, cav_object: CAV, force: bool = False) -> None:
    """Generates interactive heatmap for this clip and CAV"""
    full_clip_path = os.path.join(utils.get_project_root(), 'data/clips', clip_path)
    # Skip creating the heatmap from scratch if it already exists on the local file structure
    if heatmap_exists(clip_path, cav_object.cav_name) and not force:
        return
    # Create the slider instance: slides a kernel over transcription and computes change in sensitivity to CAV
    slider = CAVKernelSlider(
        clip_path=full_clip_path,
        cav=cav_object,
        class_mapping=utils.get_class_mapping("20class_80min"),
        kernel_size=HEATMAP_KERNEL,
        stride=HEATMAP_STRIDE
    )
    try:
        kernel_sensitivities = slider.compute_kernel_sensitivities()
    except ExtractorError as e:
        print(f'Error for {clip_path}, continuing... {e}')
        return
    else:
        # Create the plot, which takes in the attributes computed from the CAVKernelSlider class
        plotter = HeatmapCAVKernelSensitivityInteractive(
            clip_path=full_clip_path,
            sensitivity_array=kernel_sensitivities,
            clip_roll=slider.clip_roll,
            cav_name=cav_object.cav_name,  # we can just access the name of the CAV directly from the class
        )
        plotter.create_plot()
        plotter.save_fig()


def create_html(pianist: str, metadata_dict: dict, tmp_pianist: str = "Abdullah Ibrahim") -> None:
    """Creates HTML by updating template and stores in app/pages/harmony/<pianist>.html"""
    # The concepts this pianist is associated with
    concepts = list(metadata_dict.keys())
    # Load in the template and get the concept gallery, the element we need to modify
    template = os.path.join(utils.get_project_root(), 'app/pages/templates/harmony.html')
    soup = BeautifulSoup(open(template, 'r'))
    gallery = soup.find_all('div', {"class": "concept-gallery"})[0]
    # Iterating through each "row" in the concept gallery
    for tag, button, anchor, concept in zip(
            gallery.find_all('strong'),
            gallery.find_all('button'),
            gallery.find_all('a'),
            concepts
    ):
        # Get the index of the current CAV
        cav_idx = CAV_NAME_TO_IDX[concept]
        # Updating all the headings with the correct concept name
        tag.string.replace_with(concept)
        # Updating all the buttons to use the correct argument
        button['onclick'] = f"playExampleMidi({cav_idx})"
        # Updating all the anchor tags
        anchor['alt'] = concept
        concept_fmt = concept.replace("'", r"\'").replace('"', r'\"')
        anchor['onclick'] = f"""popSelect('{concept_fmt}', '{pianist}')"""
        # We can also grab the image from each anchor tag
        img = anchor.find('img')
        img['alt'] = concept
        img['src'] = f"../../assets/images/concepts/cav_notation_{str(cav_idx).zfill(3)}.png"
    # Create the output path that we'll save the HTML in
    out_path = os.path.join(utils.get_project_root(), 'app/pages/harmony', pianist.lower().replace(' ', '_') + '.html')
    # Write HTML and replace any reference to our placeholder pianist (Abdullah Ibrahim, default) with the actual one
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(str(soup).replace(tmp_pianist, pianist))


def main():
    """Creates all assets and HTML files needed for the harmony CAV explorer section of the web app"""
    # Get the concepts that each pianist is most associated with
    sensitive_concepts = get_most_sensitive_concepts(N_CONCEPTS_PER_PERFORMER)
    # Get the clips that are most associated with each concept
    clip_paths = get_clip_paths(sensitive_concepts)
    # Load the CAVs from the hard disk
    loaded_cavs = load_cavs()
    # Iterate over the pianist and associated concepts
    for pianist, concept_tracks in clip_paths.items():
        pianist_dict = {}
        # Iterate through concept name, concept tracks
        for concept, tracks in concept_tracks.items():
            # Get the current CAV from our list
            current_cav = loaded_cavs[CAV_NAME_TO_IDX[concept] - 1]
            assert current_cav.cav_name == concept  # name of CAV should be what we expect
            # We'll store the metadata here for this pianist
            pianist_dict[concept] = []
            # Iterate through clip name, clip sensitivity
            for clip, sensitivity in tqdm(tracks.items(), desc=f'Processing pianist {pianist}, concept {concept}...'):
                # Generate metadata for this clip
                pianist_dict[concept].append(generate_clip_metadata(clip, sensitivity, concept))
                # Move the MIDI to the assets folder
                copy_midi_to_assets_folder(clip)
                # Generate the interactive heatmap
                generate_clip_heatmap(clip, current_cav)
        # Dump the metadata for this pianist
        dump_metadata_json(pianist, pianist_dict)
        # Create the HTML for this pianist
        create_html(pianist, pianist_dict)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)  # can't hurt, can it?
    main()
