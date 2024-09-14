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

from deep_pianist_identification.data_augmentation import augment_funcs
from deep_pianist_identification.utils import get_project_root, PIANO_KEYS, FPS, MIDI_OFFSET, CLIP_LENGTH

__all__ = ["MIDILoader"]

MINIMUM_FRAMES = 5  # a note must last this number of frames to be used
AUGMENTATION_PROB = 0.1


@jit(nopython=True)
def normalize_array(track_array: np.ndarray) -> np.ndarray:
    return (track_array - np.min(track_array)) / (np.max(track_array) - np.min(track_array))


@jit(nopython=True, fastmath=True)
def get_piano_roll(clip_notes: list[tuple]) -> np.ndarray:
    """Largely the same as PrettyMIDI.get_piano_roll, but with optimizations using Numba"""
    clip_array = np.zeros((1, PIANO_KEYS, CLIP_LENGTH * FPS), dtype=np.float32)
    frame_iter = np.linspace(0, CLIP_LENGTH, (CLIP_LENGTH * FPS) + 1)
    for frame_idx, (frame_start, frame_end) in enumerate(zip(frame_iter, frame_iter[1:])):
        for note in clip_notes:
            note_start, note_end, note_pitch, note_velocity = note
            if note_start < frame_end and note_end >= frame_start:
                clip_array[0, note_pitch - MIDI_OFFSET, frame_idx] = note_velocity
    return clip_array


def get_midi_clip(midi_fpath, clip_idx, data_augmentation: bool = False) -> np.ndarray:
    pm = PrettyMIDI(os.path.join(midi_fpath, f'clip_{str(clip_idx).zfill(3)}.mid'))
    # Apply data augmentation, if required
    if data_augmentation:
        for func in augment_funcs:
            if np.random.uniform(0., 1.) <= AUGMENTATION_PROB:
                pm = func(pm)

    piano_instrument = pm.instruments[0]
    clip_notes = []
    for note in piano_instrument.notes:
        note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
        if (note_end - note_start) > (MINIMUM_FRAMES / FPS):
            # TODO: here is where we can randomize e.g. pitch, velocity, duration for masking particular attributes
            clip_notes.append((note_start, note_end, note_pitch, note_velocity))
    return get_piano_roll(clip_notes)


class MIDILoader(Dataset):
    def __init__(
            self,
            split: str,
            n_clips: int = None,
            normalize_velocity: bool = True,
            data_augmentation: bool = False
    ):
        super().__init__()
        csv_path = os.path.join(get_project_root(), 'references/data_splits', f'{split}_split.csv')
        split_df = pd.read_csv(csv_path, delimiter=',', index_col=0)
        self.clips = []
        self.normalize_velocity = normalize_velocity
        self.data_augmentation = data_augmentation
        for idx, track in tqdm(split_df.iterrows(), desc=f"Getting {split} data: ", total=len(split_df)):
            track_path = os.path.join(get_project_root(), 'data/clips', track['track'])
            track_clips = len(os.listdir(track_path))
            for clip_idx in range(track_clips):
                self.clips.append((track_path, track['pianist'], clip_idx))
        if n_clips is not None:
            self.clips = self.clips[:n_clips]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple:
        track_path, target_class, clip_idx = self.clips[idx]
        # Load the MIDI clip in with data augmentation, if required
        loaded_clip = get_midi_clip(track_path, clip_idx, data_augmentation=self.data_augmentation)
        if self.normalize_velocity:
            loaded_clip = normalize_array(loaded_clip)
        return loaded_clip, target_class


if __name__ == "__main__":
    from time import time
    from loguru import logger

    batch_size = 12
    n_batches = 10
    times = []
    loader = DataLoader(
        MIDILoader('train', data_augmentation=True),
        batch_size=batch_size,
        shuffle=True,
    )

    for i in tqdm(range(n_batches), desc='Loading batches: '):
        start = time()
        _, __ = next(iter(loader))
        times.append(time() - start)
    logger.info(f'Took {np.mean(times):.2}s (SD = {np.std(times):.2f}) '
                f'to create {n_batches} batches of {batch_size} items.')
