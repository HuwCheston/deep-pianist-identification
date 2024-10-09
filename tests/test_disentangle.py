#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for encoder in deep_pianist_identification/encoders/disentangle.py."""

import unittest

import torch
from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.encoders import DisentangleNet
from deep_pianist_identification.encoders.shared import MaskedAvgPool, LinearFlatten, Conv1x1
from deep_pianist_identification.utils import DEVICE, N_CLASSES, seed_everything, SEED


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

    def test_final_pooling(self):
        # Shape is (batch, features, channels) --- i.e., no permutation is required
        fake_input = torch.rand((8, 512, 4), device=DEVICE)
        # Test average pooling
        model_avg = DisentangleNet(num_layers=4, classify_dataset=False, use_masking=False, pool_type="avg")
        expected_avg = fake_input.mean(dim=2)  # Average over channel (last) dimension
        actual_avg = model_avg.pooling(fake_input).squeeze(2)  # Use pool and remove singleton dimension
        self.assertTrue(torch.all(expected_avg == actual_avg).item())
        # Test max pooling
        model_max = DisentangleNet(num_layers=4, classify_dataset=False, use_masking=False, pool_type="max")
        expected_max = fake_input.max(dim=2)[0]  # Max over channel (last) dimension
        actual_max = model_max.pooling(fake_input).squeeze(2)  # Use pool and remove singleton dimension
        self.assertTrue(torch.all(expected_max == actual_max).item())

    def test_masked_avg_pooling(self):
        pooler = MaskedAvgPool()
        # With one mask (column 1), should return mean of remaining columns
        big = torch.cat([torch.zeros((3, 512, 1)), torch.rand((3, 512, 3))], dim=2)
        expected = big[:, :, 1:].mean(dim=2).unsqueeze(2)
        actual = pooler(big)
        self.assertTrue(torch.all(actual == expected).item())
        # With three dimensions masked (columns 1:3), should be equal to max pooling
        big = torch.cat([torch.zeros((3, 512, 3)), torch.rand((3, 512, 1))], dim=2)
        expected = big.max(dim=2)[0].unsqueeze(2)
        actual = pooler(big)
        self.assertTrue(torch.all(actual == expected).item())
        # With all dimensions masked, should just return NaN
        big = torch.zeros(3, 512, 4)
        actual = pooler(big)
        self.assertTrue(torch.all(torch.isnan(actual)).item())

    def test_linear_transformation(self):
        flat = LinearFlatten()
        tester = torch.cat([torch.zeros((4, 512, 2)), torch.ones((4, 512, 2))], dim=2)
        actual = flat(tester)
        # Check the size of the output
        expected_size = (4, 512 * 4, 1)
        self.assertTrue(expected_size, actual.size())
        # First half of each tensor should be all ones, second should be all zeros
        self.assertTrue(torch.all(actual[0, :1024] == 0).item())
        self.assertTrue(torch.all(actual[0, 1024:] == 1).item())
        # We shouldn't be able to use both the attention module and a linear pooling
        self.assertRaises(NotImplementedError, lambda: DisentangleNet(use_attention=True, pool_type="linear"))

    def test_conv1x1(self):
        # Sample input of (batch, channels, features)
        tester = torch.rand(4, 512, 4)
        # Pass the test input through the layer
        conv = Conv1x1(in_channels=4)
        x = conv(tester)
        # We'd expect to convolve along the channels dimension
        expected = (4, 512, 1)
        actual = x.size()
        # Nb., we'll remove the singleton channel in DisentangleNet.forward_pooled
        self.assertTrue(expected == actual)

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
    seed_everything(SEED)
    unittest.main()
