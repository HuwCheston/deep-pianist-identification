#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for encoder in deep_pianist_identification/encoders/disentangle.py."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.encoders import DisentangleNet, Concept
from deep_pianist_identification.encoders.shared import MaskedAvgPool, LinearFlatten, Conv1x1
from deep_pianist_identification.utils import DEVICE, N_CLASSES, seed_everything, SEED, PIANO_KEYS, FPS, CLIP_LENGTH


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
        model_avg = DisentangleNet(classify_dataset=False, use_masking=False, pool_type="avg")
        expected_avg = fake_input.mean(dim=2)  # Average over channel (last) dimension
        actual_avg = model_avg.pooling(fake_input).squeeze(2)  # Use pool and remove singleton dimension
        self.assertTrue(torch.all(expected_avg == actual_avg).item())
        # Test max pooling
        model_max = DisentangleNet(classify_dataset=False, use_masking=False, pool_type="max")
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
        model = DisentangleNet(classify_dataset=False).to(DEVICE)
        roll, _, __ = next(iter(self.LOADER))
        roll = roll.to(DEVICE)
        features = model.forward_features(roll)
        self.assertTrue(len(features) == 4)
        self.assertTrue(features[0].size() == (1, 1, 512))

    def test_forward_pooled(self):
        model = DisentangleNet(classify_dataset=False).to(DEVICE)
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
        model = DisentangleNet(classify_dataset=True).to(DEVICE)
        roll, _, __ = next(iter(binary_loader))
        roll = roll.to(DEVICE)
        pooled = model(roll)
        self.assertTrue(pooled.size() == (1, 2))

    def test_concept_layer_parsing(self):
        # Define different configurations of layers to test
        layers2 = [
            dict(in_channels=1, out_channels=128, has_pool=False),
            dict(in_channels=128, out_channels=512, has_pool=True, pool_kernel_size=4)
        ]
        layers4_1 = [
            dict(in_channels=1, out_channels=64, has_pool=False),
            dict(in_channels=64, out_channels=128, has_pool=True, pool_kernel_size=2),
            dict(in_channels=128, out_channels=256, has_pool=False),
            dict(in_channels=256, out_channels=512, has_pool=True, pool_kernel_size=4),
        ]
        layers4_2 = [
            dict(in_channels=1, out_channels=128, has_pool=False),
            dict(in_channels=128, out_channels=128, has_pool=True, pool_kernel_size=2),
            dict(in_channels=128, out_channels=512, has_pool=False),
            dict(in_channels=512, out_channels=512, has_pool=True, pool_kernel_size=4),
        ]
        layers4_3 = [
            dict(in_channels=1, out_channels=64, has_pool=True, pool_kernel_size=2),
            dict(in_channels=64, out_channels=128, has_pool=True, pool_kernel_size=2),
            dict(in_channels=128, out_channels=256, has_pool=True, pool_kernel_size=2),
            dict(in_channels=256, out_channels=512, has_pool=False),
        ]
        layers8 = [
            dict(in_channels=1, out_channels=64, has_pool=False),
            dict(in_channels=64, out_channels=64, has_pool=True, pool_kernel_size=2),
            dict(in_channels=64, out_channels=128, has_pool=False),
            dict(in_channels=128, out_channels=128, has_pool=True, pool_kernel_size=2),
            dict(in_channels=128, out_channels=256, has_pool=False),
            dict(in_channels=256, out_channels=256, has_pool=True, pool_kernel_size=2),
            dict(in_channels=256, out_channels=512, has_pool=False),
            dict(in_channels=512, out_channels=512, has_pool=False),
        ]
        layers = [layers2, layers4_1, layers4_2, layers4_3, layers8]
        for layer in layers:
            conc = Concept(layers=layer)
            # Test number of convolutional layers
            expected_n_convs = len(layer)
            actual_n_convs = len([i for i in conc.layers if hasattr(i, "conv")])
            self.assertEqual(expected_n_convs, actual_n_convs)
            # Test number of pools
            expected_n_pools = len([i for i in layer if i["has_pool"]])
            actual_n_pools = len([i for i in conc.layers if hasattr(i, "avgpool")])
            self.assertEqual(expected_n_pools, actual_n_pools)
            # Test kernel size of pools
            expected_k = [(i["pool_kernel_size"], i["pool_kernel_size"]) for i in layer if i['has_pool']]
            actual_k = [i.avgpool.kernel_size for i in conc.layers if hasattr(i, "avgpool")]
            self.assertEqual(expected_k, actual_k)
            # Test placement of pools
            expected_i = [num for num, i in enumerate(layer) if i['has_pool']]
            actual_i = [num for num, i in enumerate(conc.layers) if hasattr(i, "avgpool")]
            self.assertEqual(expected_i, actual_i)
            # Test output size from convolutional layers
            kern = np.prod([i['pool_kernel_size'] for i in layer if i['has_pool']])
            expected_channels = layer[-1]["out_channels"]
            expected_height = PIANO_KEYS // kern
            expected_width = (CLIP_LENGTH * FPS) // kern
            expected_size = (expected_channels, expected_height, expected_width)
            actual_size = conc.output_channels, conc.output_height, conc.output_width
            self.assertEqual(expected_size, actual_size)
        # With only three convolutional layers, we should raise an error
        badlayers = [
            dict(in_channels=1, out_channels=128, has_pool=False),
            dict(in_channels=128, out_channels=256, has_pool=True, pool_kernel_size=4),
            dict(in_channels=256, out_channels=512, has_pool=False),
        ]
        self.assertRaises(AssertionError, Concept, layers=badlayers)


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
