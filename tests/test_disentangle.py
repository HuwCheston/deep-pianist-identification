#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for encoder in deep_pianist_identification/encoders/disentangle.py."""

import unittest

from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.encoders import DisentangleNet
from deep_pianist_identification.utils import DEVICE, N_CLASSES


class DisentangledTest(unittest.TestCase):
    LOADER = DataLoader(
        MIDILoader(
            "test",
            n_clips=10,
            classify_dataset=False,
            multichannel=True
        ),
        shuffle=True,
        batch_size=1,
        collate_fn=remove_bad_clips_from_batch
    )

    def test_forward_features(self):
        model = DisentangleNet(num_layers=4, classify_dataset=False).to(DEVICE)
        roll, _, __ = next(iter(self.LOADER))
        roll = roll.to(DEVICE)
        features = model.forward_features(roll)
        self.assertTrue(len(features) == 4)
        self.assertTrue(features[0].size() == (1, 1, 512))

    def test_forward_pooled(self):
        model = DisentangleNet(num_layers=4, classify_dataset=False).to(DEVICE)
        roll, _, __ = next(iter(self.LOADER))
        roll = roll.to(DEVICE)
        pooled = model(roll)
        self.assertTrue(pooled.size() == (1, N_CLASSES))

    def test_binary_output(self):
        binary_loader = DataLoader(
            MIDILoader(
                "test",
                n_clips=10,
                classify_dataset=True,
                multichannel=True
            ),
            shuffle=True,
            batch_size=1,
            collate_fn=remove_bad_clips_from_batch
        )
        model = DisentangleNet(num_layers=4, classify_dataset=True).to(DEVICE)
        roll, _, __ = next(iter(binary_loader))
        roll = roll.to(DEVICE)
        pooled = model(roll)
        self.assertTrue(pooled.size() == (1, 2))


if __name__ == '__main__':
    unittest.main()
