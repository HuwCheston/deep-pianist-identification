#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pre-Render MIDI clips for quicker data loading during training."""

import os
from copy import deepcopy

import pandas as pd
from joblib import Parallel, delayed
from pretty_midi import PrettyMIDI

from deep_pianist_identification.utils import get_project_root, CLIP_LENGTH, CLIP_PADDING


# def truncate_clip(midi: PrettyMIDI, start: int, stop: int) -> PrettyMIDI:
#     """Truncates a PrettyMIDI object into a clip with given CLIP_LENGTH"""
#     temp = deepcopy(midi)  # Maybe unnecessary?
#     temp.adjust_times([start, stop], [0, CLIP_LENGTH + CLIP_PADDING])
#     return temp


# def get_clip_boundaries(midi: PrettyMIDI) -> list[tuple]:
#     track_start, track_end = 0, floor(midi.get_end_time())
#     for clip_start in range(track_start, track_end, HOP_SIZE):
#         clip_end = clip_start + CLIP_LENGTH
#         # Break out once we've reached the end of the track
#         # TODO: think more about all of this
#         if clip_start > track_end:
#             break
#         # Adding the hop size again here ensures that we have padding for our time dilation augmentation and jitter
#         yield clip_start, clip_end + CLIP_PADDING


def truncate_clip(clip_id, pm_obj) -> PrettyMIDI:
    """Truncates a PrettyMIDI object into a clip with given CLIP_LENGTH"""
    clip_start = clip_id * CLIP_LENGTH
    clip_end = clip_start + CLIP_LENGTH + CLIP_PADDING
    temp = deepcopy(pm_obj)  # Maybe unnecessary?
    temp.adjust_times([clip_start, clip_end], [0, CLIP_LENGTH + CLIP_PADDING])
    return temp


def load_clip(directory_path: str, midi_filename: str) -> list[str]:
    # Define the name of the clip path
    clip_folder = os.path.join(
        get_project_root(), 'data/clips', os.path.sep.join(directory_path.split(os.path.sep)[-2:])
    )
    if not os.path.exists(clip_folder):
        os.mkdir(clip_folder)
    # Load the raw MIDI object in
    midi_obj = PrettyMIDI(os.path.join(get_project_root(), 'data/raw', directory_path, midi_filename))
    # Remove all pedal changes for clarity, probably won't change results though
    midi_obj.instruments[0].control_changes = []
    # Iterate through the total number of clips we want
    for clip_idx in range(int(midi_obj.get_end_time() // CLIP_LENGTH)):
        # Create the clip by truncating the pretty_midi object
        clip = truncate_clip(clip_idx, midi_obj)
        # Write the clip into the clip folder
        clip.write(os.path.join(clip_folder, f'clip_{str(clip_idx).zfill(3)}.mid'))

    # # Get the boundaries for each clip in the track
    # clip_boundaries = list(get_clip_boundaries(midi_obj))
    # assert len(clip_boundaries) > 0
    # # Iterate through the total number of clips we want
    # for clip_idx, (clip_start, clip_end) in enumerate(clip_boundaries):
    #     # Create the clip by truncating the pretty_midi object
    #     clip = truncate_clip(midi_obj, clip_start, clip_end)
    #     # Write the clip into the clip folder
    #     clip.write(os.path.join(clip_folder, f'clip_{str(clip_idx).zfill(3)}.mid'))
    return clip_folder


def get_midi_filepaths(data_root: str, midi_name: str = "piano_midi.mid") -> str:
    # Load in the test and train splits
    test = pd.read_csv(os.path.join(get_project_root(), "references/data_splits/test_split.csv"), index_col=0)
    train = pd.read_csv(os.path.join(get_project_root(), "references/data_splits/train_split.csv"), index_col=0)
    # valid = pd.read_csv(os.path.join(get_project_root(), "references/data_splits/validation_split.csv"), index_col=0)
    # Join into a single dataframe and get the names of each track as a list
    # df = pd.concat([train, test, valid], axis=0).reset_index(drop=True)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    tracks = sorted(df['track'].tolist())
    # Iterate through all the tracks
    for track in tracks:
        # If the MIDI exists on the file system
        if os.path.exists(os.path.join(data_root, track, midi_name)):
            # Return the name of the track and the name of the MIDI file
            yield track, midi_name


if __name__ == "__main__":
    from loguru import logger

    root = os.path.join(get_project_root(), 'data/raw')
    midi_files = list(get_midi_filepaths(root))
    logger.info(f"Rendering {len(midi_files)} MIDI files from all data splits to clips!")
    with Parallel(n_jobs=-1, verbose=5) as par:
        procced = par(delayed(load_clip)(dp, filename) for dp, filename in midi_files)
