#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pre-Render MIDI clips for quicker data loading during training"""

import os
from copy import deepcopy

from pretty_midi import PrettyMIDI
from tqdm import tqdm

from deep_pianist_identification.utils import get_project_root, CLIP_LENGTH


def create_clip(clip_id, pm_obj) -> PrettyMIDI:
    """Truncates a PrettyMIDI object into a clip with given CLIP_LENGTH"""
    clip_start = clip_id * CLIP_LENGTH
    clip_end = clip_start + CLIP_LENGTH
    temp = deepcopy(pm_obj)  # Maybe unnecessary?
    temp.adjust_times([clip_start, clip_end], [0, CLIP_LENGTH])
    return temp


if __name__ == "__main__":
    data_root = os.path.join(get_project_root(), 'data/raw')
    # Walk through all directories in the raw data folder
    for (dirpath, dirnames, filenames) in tqdm(os.walk(data_root), desc='Creating MIDI clips: '):
        for filename in filenames:
            # Skip non MIDI files
            if not filename.endswith('.mid'):
                continue
            # Get the path to the folder where we're storing our MIDI clips, creating it if it doesn't exist
            clip_folder = os.path.join(
                get_project_root(), 'data/clips', os.path.sep.join(dirpath.split(os.path.sep)[-2:])
            )
            if not os.path.exists(clip_folder):
                os.mkdir(clip_folder)
            # Load the raw MIDI object in
            midi_obj = PrettyMIDI(os.path.join(dirpath, filename))
            # Remove all pedal changes for clarity, probably won't change results though
            midi_obj.instruments[0].control_changes = []
            # Iterate through the total number of clips we want
            for clip_idx in range(int(midi_obj.get_end_time() // CLIP_LENGTH)):
                # Create the clip by truncating the pretty_midi object
                clip = create_clip(clip_idx, midi_obj)
                # Write the clip into the clip folder
                clip.write(os.path.join(clip_folder, f'clip_{str(clip_idx).zfill(3)}.mid'))
