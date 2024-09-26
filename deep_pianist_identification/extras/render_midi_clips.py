#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pre-Render MIDI clips for quicker data loading during training"""

import os
from copy import deepcopy

from joblib import Parallel, delayed
from pretty_midi import PrettyMIDI

from deep_pianist_identification.utils import get_project_root, CLIP_LENGTH

# We need to add this overlap for our rhythm dilation augmentation
# We'll truncate to CLIP_LENGTH later
OVERLAP = CLIP_LENGTH // 2


def create_clip(clip_id, pm_obj) -> PrettyMIDI:
    """Truncates a PrettyMIDI object into a clip with given CLIP_LENGTH"""
    clip_start = clip_id * CLIP_LENGTH
    clip_end = clip_start + CLIP_LENGTH + OVERLAP
    temp = deepcopy(pm_obj)  # Maybe unnecessary?
    temp.adjust_times([clip_start, clip_end], [0, CLIP_LENGTH + OVERLAP])
    return temp


def load_clip(directory_path: str, midi_filename: str) -> list[str]:
    clip_folder = os.path.join(
        get_project_root(), 'data/clips', os.path.sep.join(directory_path.split(os.path.sep)[-2:])
    )
    if not os.path.exists(clip_folder):
        os.mkdir(clip_folder)
    # Load the raw MIDI object in
    midi_obj = PrettyMIDI(os.path.join(directory_path, midi_filename))
    # Remove all pedal changes for clarity, probably won't change results though
    midi_obj.instruments[0].control_changes = []
    # Iterate through the total number of clips we want
    for clip_idx in range(int(midi_obj.get_end_time() // CLIP_LENGTH)):
        # Create the clip by truncating the pretty_midi object
        clip = create_clip(clip_idx, midi_obj)
        # Write the clip into the clip folder
        clip.write(os.path.join(clip_folder, f'clip_{str(clip_idx).zfill(3)}.mid'))
    return clip_folder


def get_midi_filepaths(data_root: str) -> str:
    # Walk through all directories in the raw data folder
    for dirpath, dirnames, filenames in os.walk(data_root):
        for file in filenames:
            joined = os.path.join(dirpath, file)
            if file.endswith('.mid') and os.path.exists(joined):
                yield dirpath, file


if __name__ == "__main__":
    from loguru import logger

    root = os.path.join(get_project_root(), 'data/raw')
    midi_files = list(get_midi_filepaths(root))
    logger.info(f"Rendering {len(midi_files)} MIDI files to clips!")
    with Parallel(n_jobs=-1, verbose=5) as par:
        procced = par(delayed(load_clip)(dp, filename) for dp, filename in midi_files)
