#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIDI DataLoader module"""

import os
import random

import numpy as np
import pandas as pd
from loguru import logger
from pretty_midi import PrettyMIDI
from tenacity import retry, retry_if_exception, stop_after_attempt
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

from deep_pianist_identification.extractors import (
    ExtractorError, RollExtractor, MelodyExtractor, HarmonyExtractor, RhythmExtractor, DynamicsExtractor,
    normalize_array, augment_midi
)
from deep_pianist_identification.utils import get_project_root, CLIP_LENGTH

__all__ = ["MIDILoader", "MIDITripletLoader", "MIDILoaderTargeted", "remove_bad_clips_from_batch"]

MAX_RETRIES = 100


def remove_bad_clips_from_batch(returned_batch):
    """Removes clips from a batch where there was an issue creating one of the concepts"""
    filtered_batch = list(filter(lambda x: x is not None, returned_batch))
    return default_collate(filtered_batch)


class MIDILoader(Dataset):
    def __init__(
            self,
            split: str,
            data_split_dir: str = "25class_0min",
            n_clips: int = None,
            data_augmentation: bool = True,
            augmentation_probability: float = 0.5,
            jitter_start: bool = True,
            normalize_velocity: bool = True,
            multichannel: bool = False,
            use_concepts: str = None,
            extractor_cfg: dict = None,
            classify_dataset: bool = False,
    ):
        super().__init__()
        # Unpack provided arguments as attributes
        self.split = split
        # If `True`, we will normalize the velocity dimension of each clip between 0 and 1
        self.normalize_velocity = normalize_velocity
        # If `True`, we will use data augmentation with specified probability `augmentation_probability
        self.data_augmentation = data_augmentation if split == "train" else False  # Never augment during testing
        self.augmentation_probability = augmentation_probability
        # If `True`, we will jitter the start time of each clip, between 0 and 15 seconds uniform
        self.jitter_start = jitter_start if split == "train" else False  # Never jitter during testing
        # If `True`, we will return multichannel piano rolls, corresponding the different musical dimensions
        self.multichannel = multichannel
        self.use_concepts = use_concepts  # For debugging: return only individual channels
        # Individual configuration dictionaries passed to piano roll extractors, only if `multichannel == True`
        self.extractor_cfg = extractor_cfg if extractor_cfg is not None else dict()
        # If `True`, we will train the model to classify which DATASET the clip comes from, not the performer!
        self.classify_dataset = classify_dataset
        # This is the function we'll use to create the piano roll from the raw MIDI input: single or multichannel
        self.creator_func = self.get_multichannel_piano_roll if self.multichannel else self.get_singlechannel_piano_roll
        # Load in the CSV and get all the clips into a list
        full_csv_path = os.path.join(get_project_root(), 'references/data_splits', data_split_dir, f'{split}_split.csv')
        if not os.path.exists(full_csv_path):
            raise FileNotFoundError(f"Can't find {split}_split.csv inside {data_split_dir}: have you created splits?")
        self.clips = list(self.get_clips_for_split(full_csv_path))
        # Truncate the number of clips if we've passed in this argument
        if n_clips is not None:
            self.clips = self.clips[:n_clips]

    def get_clips_for_split(self, csv_path: str) -> tuple:
        """For a CSV file located at a given path, load all the clips (rows) it contains as tuples"""
        split_df = pd.read_csv(csv_path, delimiter=',', index_col=0)
        for idx, track in split_df.iterrows():
            track_path = os.path.join(get_project_root(), 'data/clips', track['track'])
            track_clips = len(os.listdir(track_path))
            # Get the dataset from the track name
            dataset = track['track'].split('/')
            dataset_idx = 0 if dataset == 'jtd' else 1
            # Iterate through all clips for this track
            for clip_idx in range(track_clips):
                yield track_path, track['pianist'], dataset_idx, clip_idx

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        reraise=True,
        retry=retry_if_exception(ExtractorError),
    )
    def get_singlechannel_piano_roll(self, midi_obj: PrettyMIDI) -> np.ndarray:
        """Gets a single channel piano roll, including all data (i.e., without splitting into concepts)"""
        # Apply data augmentation to the MIDI clip if required
        if self.data_augmentation and self.split == "train":
            if np.random.uniform(0.0, 1.0) <= self.augmentation_probability:
                midi_obj = augment_midi(midi_obj)
        # If required, sample a random start time for the clip
        if self.jitter_start and self.split == "train":
            clip_start = np.random.uniform(0.0, CLIP_LENGTH // 2)
        # Otherwise, just use the start of the cli
        else:
            clip_start = 0.0
        # Create the basic extractor object
        roll = RollExtractor(midi_obj, clip_start=clip_start).roll
        # Normalize velocity dimension if required
        if self.normalize_velocity:
            roll = normalize_array(roll)
        # Squeeze to create a new channel, for parity with our multichannel approach
        return np.expand_dims(roll, axis=0)  # (batch, channel, pitch, time)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        reraise=True,
        retry=retry_if_exception(ExtractorError),
    )
    def get_multichannel_piano_roll(self, midi_obj: PrettyMIDI) -> np.ndarray:
        """For a single MIDI clip, make a four channel roll corresponding to Melody, Harmony, Rhythm, and Dynamics"""
        # Apply data augmentation to the MIDI clip if required
        if self.data_augmentation and self.split == "train":
            midi_obj = augment_midi(midi_obj)
        # Sample a random start time for the clip if required
        if self.jitter_start and self.split == "train":
            clip_start = np.random.uniform(0.0, CLIP_LENGTH // 2)
        # Otherwise, just use the start of the clip
        else:
            clip_start = 0.0
        # Create all the extractor objects
        # TODO: sort out how configs are passed through here
        mel = MelodyExtractor(midi_obj, clip_start=clip_start, **self.extractor_cfg.get("melody", dict()))
        harm = HarmonyExtractor(midi_obj, clip_start=clip_start, **self.extractor_cfg.get("harmony", dict()))
        rhy = RhythmExtractor(midi_obj, clip_start=clip_start, **self.extractor_cfg.get("rhythm", dict()))
        dyn = DynamicsExtractor(midi_obj, clip_start=clip_start, **self.extractor_cfg.get("dynamics", dict()))
        extractors = [mel, harm, rhy, dyn]
        # Normalize velocity dimension if required and stack arrays
        stacked = np.array([normalize_array(c.roll) if self.normalize_velocity else c.roll for c in extractors])
        # Using all arrays
        if self.use_concepts is None:
            return stacked  # (batch, channel == 4, pitch, time)
        # Remove any arrays we do not want to use
        else:
            return stacked[self.use_concepts, :, :]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple:
        # Unpack the clip to get the track path, the performer and dataset index, and the index of the clip
        track_path, target_class, target_dataset, clip_idx = self.clips[idx]
        # This is the target class index we'll be returning, depending on the classification problem
        target = target_dataset if self.classify_dataset else target_class
        # Load in the clip as a PrettyMIDI object
        pm = PrettyMIDI(os.path.join(track_path, f'clip_{str(clip_idx).zfill(3)}.mid'))
        # Load the piano roll as either single channel (all valid MIDI notes) or multichannel (split into concepts)
        try:
            piano_roll = self.creator_func(pm)
        # If creating the piano roll fails for whatever reason, return None which will be caught in the collate_fn
        # This will be retried MAX_RETRIES times in case it's just an issue with the random augmentation/jittering
        except ExtractorError as e:
            logger.error(f'Failed to create piano rolls for {track_path} (clip {clip_idx}) '
                         f'after retrying {MAX_RETRIES} times. Skipping this clip! '
                         f'Error message: {e}')
            return None
        # Otherwise, return the roll, target class (or dataset) index, and the raw name of the track
        else:
            return piano_roll, target, track_path.split(os.path.sep)[-1]


class MIDILoaderTargeted(MIDILoader):
    """Dataloader subclass that only returns clips from pianists from a particular target class (or classes)"""

    def __init__(
            self,
            split: str,
            target: int | list[int],
            data_split_dir: str = "25class_0min",
            n_clips: int = None,
            data_augmentation: bool = True,
            augmentation_probability: float = 0.5,
            jitter_start: bool = True,
            normalize_velocity: bool = True,
            multichannel: bool = False,
            use_concepts: str = None,
            extractor_cfg: dict = None,
            classify_dataset: bool = False,
    ):
        # We could probably implement this fairly simply but needs testing
        if classify_dataset is True:
            raise NotImplementedError("`classify_dataset` must be False when using `MIDILoaderTargeted`")
        self.target = target if isinstance(target, list) else [target]
        # Initialise the main dataloader with all passed argu
        super().__init__(
            split, data_split_dir, n_clips, data_augmentation, augmentation_probability, jitter_start,
            normalize_velocity, multichannel, use_concepts, extractor_cfg, classify_dataset
        )

    def get_clips_for_split(self, csv_path: str) -> tuple:
        """For a CSV file located at a given path, load all the clips (rows) it contains as tuples"""
        split_df = pd.read_csv(csv_path, delimiter=',', index_col=0)
        for idx, track in split_df.iterrows():
            # Skip over pianists who are not in our target list
            if track['pianist'] not in self.target:
                continue
            track_path = os.path.join(get_project_root(), 'data/clips', track['track'])
            track_clips = len(os.listdir(track_path))
            # Get the dataset from the track name
            dataset = track['track'].split('/')
            dataset_idx = 0 if dataset == 'jtd' else 1
            # Iterate through all clips for this track
            for clip_idx in range(track_clips):
                yield track_path, track['pianist'], dataset_idx, clip_idx


class MIDITripletLoader(MIDILoader):
    """A variant of the MIDILoader that returns both anchor and positive clips (i.e., two clips from one pianist)"""

    def __init__(
            self,
            split: str,
            data_split_dir: str = "25class_0min",
            n_clips: int = None,
            data_augmentation: bool = True,
            augmentation_probability: float = 0.5,
            jitter_start: bool = True,
            normalize_velocity: bool = True,
            multichannel: bool = False,
            use_concepts: str = None,
            extractor_cfg: dict = None,
            classify_dataset: bool = False,
    ):
        # We could probably implement this fairly simply but needs testing
        if classify_dataset is True:
            raise NotImplementedError("`classify_dataset` must be False when using `MIDITripletLoader`")
        # Initialise the main dataloader with all passed argu
        super().__init__(
            split, data_split_dir, n_clips, data_augmentation, augmentation_probability, jitter_start,
            normalize_velocity, multichannel, use_concepts, extractor_cfg, classify_dataset
        )

    def get_positive_sample(self, anchor_class: int, anchor_idx: int):
        # Create a list of indexes but skip over the index of the anchor clip
        idxs = [i for i in range(len(self)) if i != anchor_idx]
        # Randomly shuffle the idxs so that we don't always return the first clip for each pianist
        random.shuffle(idxs)
        # Get a positive sample: same target class, but a different clip
        for newidx in idxs:
            # Unpack the clip metadata
            positive_track, positive_target, _, pos_idx = self.clips[newidx]
            # If the pianist of the current clip is the same as our anchor pianist
            if positive_target == anchor_class:
                # Load the clip as a pretty MIDI object
                positive_midi = PrettyMIDI(os.path.join(positive_track, f'clip_{str(pos_idx).zfill(3)}.mid'))
                # Try to create the piano roll object
                try:
                    return self.creator_func(positive_midi)
                # If we fail to create the roll for whatever reason, silently proceed to the next clip
                except ExtractorError:
                    continue

    def __getitem__(self, idx: int) -> tuple:
        # Unpack the clip to get the track path, the performer and dataset index, and the index of the clip
        anchor_path, anchor_class, _, clip_idx = self.clips[idx]
        # Load in the anchor clip as a PrettyMIDI object
        pm = PrettyMIDI(os.path.join(anchor_path, f'clip_{str(clip_idx).zfill(3)}.mid'))
        # Load the piano roll as either single channel (all valid MIDI notes) or multichannel (split into concepts)
        try:
            anchor_roll = self.creator_func(pm)
        # If creating the piano roll fails for whatever reason, return None which will be caught in the collate_fn
        # This will be retried MAX_RETRIES times in case it's just an issue with the random augmentation/jittering
        except ExtractorError as e:
            logger.error(f'Failed to create piano rolls for {anchor_path} (clip {clip_idx}) '
                         f'after retrying {MAX_RETRIES} times. Skipping this clip! '
                         f'Error message: {e}')
            return None
        # Otherwise, return the roll, target class (or dataset) index, the raw name of the track, AND the positive MIDI
        else:
            positive_roll = self.get_positive_sample(anchor_class, idx)
            # If, somehow, we're not able to create another piano roll for this class
            if positive_roll is None:
                logger.error(f"Failed to create a positive piano roll for class {anchor_class}. This shouldn't happen!")
                return None
            return anchor_roll, anchor_class, anchor_path.split(os.path.sep)[-1], positive_roll


if __name__ == "__main__":
    from time import time
    from deep_pianist_identification.utils import seed_everything, SEED

    seed_everything(SEED)

    batch_size = 12
    times = []
    loader = DataLoader(
        MIDILoaderTargeted('train', target=[1, 2], data_augmentation=True, augmentation_probability=1.0,
                           jitter_start=True,
                           normalize_velocity=True, multichannel=True),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=remove_bad_clips_from_batch,
    )

    start = time()
    proc = 0
    for batch, _, __ in tqdm(loader, desc=f'Loading batches of {batch_size} clips: '):
        times.append(time() - start)
        start = time()
        proc += 1

    logger.info(f'Took {np.sum(times):.2f}s to create batches of {batch_size} items. '
                f'(Mean = {np.mean(times):.2f}s, SD = {np.std(times):.2f}) ')
