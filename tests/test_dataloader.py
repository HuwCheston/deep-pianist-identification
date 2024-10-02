#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader in deep_pianist_identification/dataloader.py."""

import unittest

import torch
from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.utils import seed_everything, PIANO_KEYS, CLIP_LENGTH, FPS


class DataloaderTest(unittest.TestCase):
    batch_size = 1
    n_batches = 1

    def test_singlechannel_output(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                multichannel=False,
                normalize_velocity=True,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Shape should be (batch, 1, 88, 3000)
        self.assertEqual((self.batch_size, 1, PIANO_KEYS, CLIP_LENGTH * FPS), roll.shape)
        # Array should contain values other than just zero
        self.assertTrue(roll.any())

    def test_multichannel_output(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                multichannel=True,
                normalize_velocity=True,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Test the size of the array
        self.assertEqual((self.batch_size, 4, PIANO_KEYS, CLIP_LENGTH * FPS), roll.shape)
        # Array should contain values other than just zero
        self.assertTrue(roll.any())

    def test_multichannel_invidualconcept_output(self):
        loader = DataLoader(
            MIDILoader(
                'train',
                multichannel=True,
                normalize_velocity=True,
                use_concepts=[0, 1, 3]
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Shape should be (batch, 3, 88, 3000)
        self.assertEqual((self.batch_size, 3, PIANO_KEYS, CLIP_LENGTH * FPS), roll.shape)
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
                multichannel=True,
                normalize_velocity=False,
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
            MIDILoader(
                'train',
                multichannel=True,
                normalize_velocity=True,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=remove_bad_clips_from_batch,
        )
        roll, _, __ = next(iter(loader))
        # Max velocity should equal 1
        self.assertTrue(roll[0, 0, :, :].max() == 1.0)


if __name__ == '__main__':
    seed_everything(42)
    unittest.main()
