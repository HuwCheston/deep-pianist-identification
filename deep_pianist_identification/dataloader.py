#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIDI DataLoader module"""

import os

import numpy as np
import pandas as pd
from numba import jit
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deep_pianist_identification.utils import get_project_root, PIANO_KEYS, FPS, MIDI_OFFSET, CLIP_LENGTH

__all__ = ["MIDILoader"]

MINIMUM_FRAMES = 1  # a note must last this number of frames to be used


@jit(nopython=True)
def normalize_array(track_array: np.ndarray) -> np.ndarray:
    # TODO: this means no notes are played at all!
    if np.max(track_array) == 0:
        return track_array
    else:
        return (track_array - np.min(track_array)) / (np.max(track_array) - np.min(track_array))


@jit(nopython=True, fastmath=True)
def get_piano_roll(clip_notes: list[tuple]) -> np.ndarray:
    clip_array = np.zeros((1, PIANO_KEYS, CLIP_LENGTH * FPS), dtype=np.float32)
    frame_iter = np.linspace(0, CLIP_LENGTH, (CLIP_LENGTH * FPS) + 1)
    for frame_idx, (frame_start, frame_end) in enumerate(zip(frame_iter, frame_iter[1:])):
        for note in clip_notes:
            note_start, note_end, note_pitch, note_velocity = note
            if note_start < frame_end and note_end >= frame_start:
                clip_array[0, note_pitch - MIDI_OFFSET, frame_idx] = note_velocity
    return clip_array


def get_midi_clip(midi_fpath, clip_idx) -> np.ndarray:
    pm = PrettyMIDI(os.path.join(midi_fpath, f'clip_{str(clip_idx).zfill(3)}.mid')).instruments[0]
    clip_notes = []
    for note in pm.notes:
        note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
        if (note_end - note_start) > (MINIMUM_FRAMES / FPS):
            # TODO: here is where we can randomize e.g. pitch, velocity, duration for masking particular attributes
            clip_notes.append((note_start, note_end, note_pitch, note_velocity))
    return get_piano_roll(clip_notes)


class MIDILoader(Dataset):
    FORCE_REGEN_CLIP = True

    def __init__(self, split: str, n_clips: int = None, normalize_velocity: bool = True):
        super().__init__()
        csv_path = os.path.join(get_project_root(), 'references/data_splits', f'{split}_split.csv')
        split_df = pd.read_csv(csv_path, delimiter=',', index_col=0)
        self.clips = []
        self.normalize_velocity = normalize_velocity
        for idx, track in tqdm(split_df.iterrows(), desc=f"Getting {split} data: "):
            track_clips = int(track['duration'] // CLIP_LENGTH)
            track_path = os.path.join(get_project_root(), 'data/clips', track['track'])
            for clip_idx in range(track_clips):
                self.clips.append((track_path, track['pianist'], clip_idx))
        self.clips = self.clips[:n_clips]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple:
        track_path, target_class, clip_idx = self.clips[idx]
        loaded_clip = get_midi_clip(track_path, clip_idx)
        if self.normalize_velocity:
            loaded_clip = normalize_array(loaded_clip)
        return loaded_clip, target_class


if __name__ == "__main__":
    from time import time
    from deep_pianist_identification.training import BATCH_SIZE

    loader = DataLoader(
        MIDILoader('train'),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    n_batches = 10
    times = []

    for i in tqdm(range(n_batches), desc='Loading batches: '):
        start = time()
        _, __ = next(iter(loader))
        times.append(time() - start)
    print(f'Took {np.mean(times)} (SD = {np.std(times)}) to create {n_batches} batches')
