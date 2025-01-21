#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create plots and HTML for pianist melodic progression usage"""

import json
import os

from pretty_midi import PrettyMIDI

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import MelodyExtractor, ExtractorError
from deep_pianist_identification.whitebox import wb_utils

DATASET = "20class_80min"
INPUT_DIR = os.path.join(utils.get_project_root(), 'reports/figures/whitebox/lr_weights')


def parse_input(input_fpath: str, extension: str = None) -> str:
    """Parses an input filepath with given extension and checks that it exists"""
    # Add the extension to the filepath if we've provided it
    if extension is not None:
        if not input_fpath.endswith(extension):
            input_fpath += f'.{extension}'
    path = os.path.join(INPUT_DIR, input_fpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {input_fpath} inside {INPUT_DIR}. Have you run `create_classifier.py`?"
        )
    return path


def extract_feature_timestamps(clip_midi: PrettyMIDI, desired_feature: list[int]) -> tuple[dict, int]:
    """Extract timestamps where a given feature appears in a clip"""
    # Sort the notes by onset start time and calculate the intervals
    melody = sorted(clip_midi.instruments[0].notes, key=lambda x: x.start)
    intervals = [i2.pitch - i1.pitch for i1, i2 in zip(melody[0:], melody[1:])]
    # Extract all the n-grams with the same n as our desired feature
    n = len(desired_feature)
    all_features = [(intervals[i: i + n]) for i in range(len(intervals) - n + 1)]
    # Subset to get only indices of desired n-grams
    desired_feature_idxs = [i for i, ngram in enumerate(all_features) if ngram == desired_feature]
    # Get starting and stopping timestamps for every appearance of this feature
    return [(melody[idx].start, melody[idx + n].end) for idx in desired_feature_idxs]


def get_melody(clip_path: str) -> PrettyMIDI | None:
    """Extract melody from a given clip and return a PrettyMIDI object"""
    # Load the clip in as a PrettyMIDI object
    loaded = PrettyMIDI(clip_path)
    # Try and create the piano roll, but return None if we end up with zero notes
    try:
        return MelodyExtractor(loaded, clip_start=0).output_midi
    except ExtractorError as e:
        print(f'Errored for {clip_path}, skipping! {e}')
        return None


def get_top_melody_features(pianists: list[str]) -> dict:
    with open(parse_input("lr_weights.json"), 'r') as js_loc:
        weights_json = json.load(js_loc)
    return {p: [eval(i.replace('M_', '')) for i in weights_json[p]['melody']['top_names']] for p in pianists}


def main():
    # Load in filenames, metadata etc. for all clips in the dataset regardless of their split (e.g., train, test)
    clips = [clip for data_split in wb_utils.get_all_clips(DATASET) for clip in data_split]
    # Load in the mapping of indexes: pianists
    pianist_mapping = utils.get_class_mapping(DATASET)
    # Get the names of the top N melody features associated with each pianist
    pianist_top_features = get_top_melody_features(pianist_mapping.values())
    # TODO: generate notation in music21 for each feature and store inside app/assets/images/features
    # TODO: get timestamps for feature usage within every clip
    # TODO: subset to get only the N (= 5?) clips that use each feature the most
    # TODO: subset MIDI to make feature appearances loud, all other notes quiet
    # TODO: generate a metadata.json with required filepaths, track names etc.
    # TODO: copy the template html and fill in required information then save inside app/assets/pages/melody


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)  # we shouldn't be doing anything random, but why not?
    main()
