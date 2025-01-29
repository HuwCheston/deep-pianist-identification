#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader in deep_pianist_identification/dataloader.py."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import (
    MIDILoader, MIDITripletLoader, remove_bad_clips_from_batch, MIDILoaderTargeted
)
from deep_pianist_identification.utils import seed_everything, PIANO_KEYS, CLIP_LENGTH, FPS, SEED


class DataloaderTest(unittest.TestCase):
    batch_size = 10  # Should have at least one valid clip in a batch of this size
    n_batches = 1
    bad_clip, bad_idx = "jarrettk-pt-unaccompanied-xxxx-187f2exh", 15

    def test_singlechannel_output(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                normalize_velocity=True,
                multichannel=False,
                data_split_dir="25class_0min"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        batch = roll.shape[0]  # Batch size can vary if clips return None
        self.assertEqual((batch, 1, PIANO_KEYS, CLIP_LENGTH * FPS), roll.shape)
        # Array should contain values other than just zero
        self.assertTrue(roll.any())

    def test_return_clip_number(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                normalize_velocity=True,
                multichannel=True,
                data_split_dir="25class_0min",
                return_clip_path=False
            )
        )
        _, __, path = next(iter(loader))  # path will just be `track_name`
        self.assertTrue('clip_' not in path[0])
        loader = DataLoader(
            MIDILoader(
                'train',
                normalize_velocity=True,
                multichannel=True,
                data_split_dir="25class_0min",
                return_clip_path=True
            )
        )
        _, __, path = next(iter(loader))
        self.assertTrue('clip_' in path[0])  # path will be `track_name/clip_number`

    def test_multichannel_output(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                normalize_velocity=True,
                multichannel=True,
                data_split_dir="25class_0min"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        batch = roll.shape[0]  # Batch size can vary if clips return None
        self.assertEqual((batch, 4, PIANO_KEYS, CLIP_LENGTH * FPS), roll.shape)
        # Array should contain values other than just zero
        self.assertTrue(roll.any())

    def test_multichannel_invidualconcept_output(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                normalize_velocity=True,
                multichannel=True,
                use_concepts=[0, 1, 3],
                data_split_dir="25class_0min"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Shape should be (batch, 3, 88, 3000)
        batch = roll.shape[0]  # Batch size can vary if clips return None
        self.assertEqual((batch, 3, PIANO_KEYS, CLIP_LENGTH * FPS), roll.shape)
        # Array should contain values other than just zero
        self.assertTrue(roll.any())
        # Velocity should be moved into our 3rd channel, from the fourth
        self.assertTrue(torch.unique(roll[0, 2, :, :]).numel() > 2)
        self.assertFalse(torch.unique(roll[0, 0, :, :]).numel() > 2)

    def test_velocity_normalization(self):
        # No normalization
        loader = DataLoader(
            MIDILoader(
                'train',
                normalize_velocity=False,
                multichannel=True,
                data_split_dir="25class_0min"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Max velocity should be greater than 1
        self.assertTrue(roll[0, 0, :, :].max() >= 1)
        # With normalization
        loader = DataLoader(
            MIDILoader('train', normalize_velocity=True, multichannel=True),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Max velocity should equal 1
        self.assertTrue(roll[0, 0, :, :].max() == 1.0)

    def test_skip_bad_clip(self):
        loader = MIDILoader(
            'train',
            normalize_velocity=False,
            multichannel=True,
            data_split_dir="25class_0min"
        )
        # Keith Jarrett doesn't play anything in this clip
        bad_clip_idx = [idx for idx, clip in enumerate(loader.clips)
                        if self.bad_clip in clip[0] and clip[-1] == self.bad_idx]
        roll = loader.__getitem__(bad_clip_idx[0])  # This will also raise a message in Loguru
        self.assertIsNone(roll)
        # Keith Jarrett plays something in this clip
        good_clip_idx = [idx for idx, clip in enumerate(loader.clips)
                         if self.bad_clip in clip[0] and clip[-1] == self.bad_idx - 1]
        roll = loader.__getitem__(good_clip_idx[0])
        self.assertIsNotNone(roll)

    def test_binary_classification(self):
        loader = iter(MIDILoader(
            'train',
            normalize_velocity=False,
            multichannel=True,
            classify_dataset=True,
            data_split_dir="25class_0min"
        ))
        # Setting `classify_dataset == True` should result in a binary output for the target class
        for i in range(50):
            _, target, ___ = next(loader)
            self.assertTrue(target == 0 or target == 1)

    def test_bad_split(self):
        self.assertRaises(
            FileNotFoundError,
            lambda: MIDILoader(
                'train',
                normalize_velocity=False,
                multichannel=True,
                classify_dataset=True,
                data_split_dir="i_dont_exist"
            )
        )


class TripletLoaderTest(unittest.TestCase):
    batch_size = 10

    def test_anchor_positive_clips(self):
        loader = DataLoader(
            MIDITripletLoader(
                'train',
                normalize_velocity=True,
                multichannel=True,
                data_split_dir="25class_0min"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        # Get the first batch from our dataloader
        anch_roll, anch_class, anch_path, pos_roll = next(iter(loader))
        # The positive and anchor clips should have different content
        self.assertFalse(torch.equal(anch_roll, pos_roll))
        # But they should be the same size
        self.assertTrue(anch_roll.size() == pos_roll.size())

    def test_get_positive_samples(self):
        loader = MIDITripletLoader(
            'train',
            normalize_velocity=True,
            multichannel=True,
            data_split_dir="25class_0min"
        )
        anc_index, anc_class = 50, 5
        # Draw two random positive samples for the anchor index and class
        positive_one = loader.get_positive_sample(anc_class, anc_index)
        positive_two = loader.get_positive_sample(anc_class, anc_index)
        # The two positive samples should be different from each other
        self.assertFalse(np.array_equal(positive_one, positive_two))


class TargetedLoaderTest(unittest.TestCase):
    def test_single_target(self):
        expected = 1
        loader = DataLoader(
            MIDILoaderTargeted(
                'train',
                target=expected,
                normalize_velocity=True,
                multichannel=True,
                data_split_dir="25class_0min"
            ),
            batch_size=10,
            shuffle=True,
        )
        loader = iter(loader)
        # Iterate through 10 batches
        for i in range(10):
            _, target, __ = next(loader)
            for targ in target.tolist():
                self.assertEqual(targ, expected)

    def test_multiple_targets(self):
        expected = [1, 4, 5, 10, 19]
        loader = DataLoader(
            MIDILoaderTargeted(
                'train',
                target=expected,
                normalize_velocity=True,
                multichannel=True,
                data_split_dir="25class_0min"
            ),
            batch_size=10,
            shuffle=True,
        )
        loader = iter(loader)
        # Iterate through 10 batches
        for i in range(10):
            _, target, __ = next(loader)
            for targ in target.tolist():
                self.assertIn(targ, expected)


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
