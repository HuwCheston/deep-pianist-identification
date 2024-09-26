#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIDI DataLoader module"""

import os

import numpy as np
import pandas as pd
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

from deep_pianist_identification.data_augmentation import data_augmentation_transpose
from deep_pianist_identification.extractors import get_multichannel_piano_roll, get_singlechannel_piano_roll
from deep_pianist_identification.utils import get_project_root

__all__ = ["MIDILoader", "remove_bad_clips_from_batch"]

AUGMENTATION_PROB = 1.0


def apply_data_augmentation(pm_obj: PrettyMIDI) -> PrettyMIDI:
    """Applies a random augmentation to the MIDI object"""
    if np.random.uniform(0., 1.) <= AUGMENTATION_PROB:
        # Choose a random augmentation and apply it to the MIDI object
        # random_aug = np.random.choice(data_augmenta)
        new = data_augmentation_transpose(pm_obj)
        # If we haven't removed all the notes from the MIDI file
        if len(new.instruments[0].notes) > 0:
            return new
    # If we're not applying augmentation OR we've removed all the notes, return the initial MIDI object
    return pm_obj


class MIDILoader(Dataset):
    def __init__(
            self,
            split: str,
            n_clips: int = None,
            normalize_velocity: bool = True,
            data_augmentation: bool = False,
            multichannel: bool = False,
            use_concepts: str = None
    ):
        super().__init__()
        # Unpack provided arguments as attributes
        self.normalize_velocity = normalize_velocity
        self.data_augmentation = data_augmentation
        self.multichannel = multichannel
        self.use_concepts = use_concepts
        # Load in the CSV and get all the clips into a list
        csv_path = os.path.join(get_project_root(), 'references/data_splits', f'{split}_split.csv')
        self.clips = list(self.get_clips_for_split(csv_path))
        # Truncate the number of clips if we've passed in this argument
        if n_clips is not None:
            self.clips = self.clips[:n_clips]

    @staticmethod
    def get_clips_for_split(csv_path: str) -> tuple:
        """For a CSV file located at a given path, load all the clips (rows) it contains as tuples"""
        split_df = pd.read_csv(csv_path, delimiter=',', index_col=0)
        for idx, track in split_df.iterrows():
            track_path = os.path.join(get_project_root(), 'data/clips', track['track'])
            track_clips = len(os.listdir(track_path))
            for clip_idx in range(track_clips):
                yield track_path, track['pianist'], clip_idx

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple:
        track_path, target_class, clip_idx = self.clips[idx]
        # Load in the clip as a PrettyMIDI object
        pm = PrettyMIDI(os.path.join(track_path, f'clip_{str(clip_idx).zfill(3)}.mid'))
        # Apply data augmentation, if required
        if self.data_augmentation:
            pm = apply_data_augmentation(pm)
        # Load the piano roll as either single channel (all valid MIDI notes) or multichannel (split into concepts)
        # We normalize the piano roll directly in each function, as required
        if self.multichannel:
            try:
                piano_roll = get_multichannel_piano_roll(
                    pm, use_concepts=self.use_concepts, normalize=self.normalize_velocity
                )
            # If the clip somehow contains no chords or melody notes, we need to return None to skip using it
            except Exception as err:
                logger.warning(f'Failed for {track_path}, clip {clip_idx}, skipping! {err}')
                return None
        else:
            piano_roll = get_singlechannel_piano_roll(pm, normalize=self.normalize_velocity)
        return piano_roll, target_class


def remove_bad_clips_from_batch(returned_batch):
    """Removes clips from a batch where there was an issue creating one of the concepts"""
    filtered_batch = list(filter(lambda x: x is not None, returned_batch))
    return default_collate(filtered_batch)


if __name__ == "__main__":
    from time import time
    from loguru import logger

    batch_size = 12
    n_batches = 100
    times = []
    loader = DataLoader(
        MIDILoader(
            'train',
            data_augmentation=False,
            multichannel=True,
            normalize_velocity=True
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=remove_bad_clips_from_batch,
    )

    for i in tqdm(range(n_batches), desc=f'Loading {n_batches} batches of {batch_size} clips: '):
        start = time()
        try:
            # Shape: (batch, channels, pitch, time)
            batch, __ = next(iter(loader))
            assert len(batch.size()) == 4
        except Exception as e:
            logger.warning(f'Failed to create batch {i} with error: {e}')
        times.append(time() - start)

    logger.info(f'Took {np.sum(times):.2f}s to create {n_batches} batches of {batch_size} items. '
                f'(Mean = {np.mean(times):.2f}s, SD = {np.std(times):.2f}) ')
