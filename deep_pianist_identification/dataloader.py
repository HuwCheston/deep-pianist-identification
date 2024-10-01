#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIDI DataLoader module"""

import os

import numpy as np
import pandas as pd
from loguru import logger
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

from deep_pianist_identification.extractors import (
    ExtractorError, BaseExtractor, MelodyExtractor, HarmonyExtractor, RhythmExtractor, DynamicsExtractor,
    normalize_array,
)
from deep_pianist_identification.utils import get_project_root

__all__ = ["MIDILoader", "remove_bad_clips_from_batch"]


def remove_bad_clips_from_batch(returned_batch):
    """Removes clips from a batch where there was an issue creating one of the concepts"""
    filtered_batch = list(filter(lambda x: x is not None, returned_batch))
    return default_collate(filtered_batch)


class MIDILoader(Dataset):
    def __init__(
            self,
            split: str,
            n_clips: int = None,
            normalize_velocity: bool = True,
            multichannel: bool = False,
            use_concepts: str = None,
            extractor_cfg: dict = None,
    ):
        super().__init__()
        # Unpack provided arguments as attributes
        self.normalize_velocity = normalize_velocity
        self.multichannel = multichannel
        self.use_concepts = use_concepts
        self.extractor_cfg = extractor_cfg if extractor_cfg is not None else dict()
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

    def get_singlechannel_piano_roll(self, midi_obj: PrettyMIDI) -> np.ndarray:
        """Gets a single channel piano roll, including all data (i.e., without splitting into concepts)"""
        # We can use the base extractor, which will perform checks e.g., removing invalid notes
        # Our base extractor class doesn't accept any extra arguments, as it doesn't perform augmentations etc.
        extract = BaseExtractor(midi_obj)
        roll = extract.roll
        # Normalize if required
        if self.normalize_velocity:
            roll = normalize_array(roll)
        # Squeeze to create a new channel, for parity with our multichannel approach (i.e., batch, channel, pitch, time)
        return np.expand_dims(roll, axis=0)

    def get_multichannel_piano_roll(self, midi_obj: PrettyMIDI) -> np.ndarray:
        """For a single MIDI clip, make a four channel roll corresponding to Melody, Harmony, Rhythm, and Dynamics"""
        # Create all the extractors for our piano roll
        # Use the provided extractor configuration, or if this doesn't exist, fall back on default arguments in class
        mel = MelodyExtractor(midi_obj, **self.extractor_cfg.get("melody", dict()))
        harm = HarmonyExtractor(midi_obj, **self.extractor_cfg.get("harmony", dict()))
        rhy = RhythmExtractor(midi_obj, **self.extractor_cfg.get("rhythm", dict()))
        dyn = DynamicsExtractor(midi_obj, **self.extractor_cfg.get("dynamics", dict()))
        extractors = [mel, harm, rhy, dyn]
        # Normalize velocity dimension if required and stack arrays
        stacked = np.array([normalize_array(c.roll) if self.normalize_velocity else c.roll for c in extractors])
        # Using all arrays
        if self.use_concepts is None:
            return stacked
        # Remove any arrays we do not want to use
        else:
            return stacked[self.use_concepts, :, :]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple:
        track_path, target_class, clip_idx = self.clips[idx]
        # Load in the clip as a PrettyMIDI object
        pm = PrettyMIDI(os.path.join(track_path, f'clip_{str(clip_idx).zfill(3)}.mid'))
        # Load the piano roll as either single channel (all valid MIDI notes) or multichannel (split into concepts)
        if self.multichannel:
            creator_func = self.get_multichannel_piano_roll
        else:
            creator_func = self.get_singlechannel_piano_roll
        # If creating the piano roll fails for whatever reason, return None which will be caught in the collate_fn
        try:
            piano_roll = creator_func(pm)
        except ExtractorError as e:
            logger.warning(f'Failed for {track_path}, clip {clip_idx}, skipping! {e}')
            return None
        # Otherwise, return the roll, target class index, and the raw name of the track
        else:
            return piano_roll, target_class, track_path.split(os.path.sep)[-1]


if __name__ == "__main__":
    from time import time

    batch_size = 12
    n_batches = 100
    times = []
    loader = DataLoader(
        MIDILoader(
            'train',
            multichannel=True,
            normalize_velocity=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=remove_bad_clips_from_batch,
    )

    for i in tqdm(range(n_batches), desc=f'Loading {n_batches} batches of {batch_size} clips: '):
        start = time()
        # Shape: (batch, channels, pitch, time)
        batch, _, __ = next(iter(loader))
        times.append(time() - start)

    logger.info(f'Took {np.sum(times):.2f}s to create {n_batches} batches of {batch_size} items. '
                f'(Mean = {np.mean(times):.2f}s, SD = {np.std(times):.2f}) ')
