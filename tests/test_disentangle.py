#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for encoder in deep_pianist_identification/encoders/disentangle.py."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.encoders import DisentangleNet, Concept
from deep_pianist_identification.encoders.shared import MaskedAvgPool, Conv1x1, Identity
from deep_pianist_identification.utils import (
    DEVICE, seed_everything, SEED, PIANO_KEYS, FPS, CLIP_LENGTH
)


class DisentangledTest(unittest.TestCase):
    LOADER = DataLoader(
        MIDILoader(
            "test",
            n_clips=10,
            multichannel=True,
            classify_dataset=False,
            data_split_dir="25class_0min"
        ),
        shuffle=True,
        batch_size=1,
        collate_fn=remove_bad_clips_from_batch
    )
    NUM_CLASSES = 25

    def test_final_pooling(self):
        # Shape is (batch, features, channels) --- i.e., no permutation is required
        fake_input = torch.rand((8, 512, 4), device=DEVICE)
        # Test average pooling
        model_avg = DisentangleNet(num_classes=self.NUM_CLASSES, use_masking=False, pool_type="avg")
        expected_avg = fake_input.mean(dim=2)  # Average over channel (last) dimension
        actual_avg = model_avg.pooling(fake_input).squeeze(2)  # Use pool and remove singleton dimension
        self.assertTrue(torch.all(expected_avg == actual_avg).item())
        # Test max pooling
        model_max = DisentangleNet(num_classes=self.NUM_CLASSES, use_masking=False, pool_type="max")
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

    def test_conv1x1(self):
        # Sample input of (batch, channels, features)
        tester = torch.rand(4, 512, 4)
        # Pass the test input through the layer
        conv = Conv1x1(in_channels=4)
        x = conv(tester)
        # We'd expect to convolve along the channels dimension
        expected = (4, 512, 1)
        actual = x.size()
        # Nb., we'll remove the singleton channel in DisentangleNet.forward_features
        self.assertTrue(expected == actual)

    def test_forward_concepts(self):
        model = DisentangleNet(num_classes=self.NUM_CLASSES).to(DEVICE)
        roll, _, __ = next(iter(self.LOADER))
        roll = roll.to(DEVICE)
        features = model.extract_concepts(roll)
        self.assertTrue(len(features) == 4)
        self.assertTrue(features[0].size() == (1, 1, 512))

    def test_forward_features(self):
        model = DisentangleNet(num_classes=self.NUM_CLASSES).to(DEVICE)
        roll, _, __ = next(iter(self.LOADER))
        roll = roll.to(DEVICE)
        features = model.forward_features(roll)
        expected = (1, 512)
        actual = features.size()
        self.assertTrue(expected == actual)

    def test_forward(self):
        model = DisentangleNet(num_classes=self.NUM_CLASSES).to(DEVICE)
        roll, _, __ = next(iter(self.LOADER))
        roll = roll.to(DEVICE)
        pooled = model(roll)
        self.assertTrue(pooled.size() == (1, self.NUM_CLASSES))

    def test_binary_output(self):
        binary_loader = DataLoader(
            MIDILoader(
                "test",
                n_clips=10,
                multichannel=True,
                classify_dataset=True,
                data_split_dir="25class_0min"
            ),
            shuffle=True,
            batch_size=1,
            collate_fn=remove_bad_clips_from_batch
        )
        model = DisentangleNet(num_classes=2).to(DEVICE)
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

    def test_masking(self):
        # Test with no masking being applied
        nomask = DisentangleNet(num_classes=20, use_masking=False, mask_probability=0.0)
        self.assertFalse(nomask.use_masking and nomask.training)
        # We should never apply masking
        nomask_embeddings = nomask.mask([torch.rand(2, 1, 512) for _ in range(4)])
        self.assertTrue(all((torch.count_nonzero(emb) > 0).item() for emb in nomask_embeddings))

        # Test with masking always being applied
        yesmask = DisentangleNet(num_classes=20, use_masking=True, mask_probability=1.0)
        masked_embeddings = yesmask.mask([torch.rand(2, 1, 512) for _ in range(4)])
        # At least one embedding should be replaced with all zeros
        self.assertTrue(any((torch.count_nonzero(emb) == 0).item() for emb in masked_embeddings))
        self.assertTrue((torch.count_nonzero(yesmask.dropper(torch.rand(2, 1, 512))) == 0).item())

        # Test with masking being applied usually, but we're testing
        testmask = DisentangleNet(num_classes=20, use_masking=True, mask_probability=1.0)
        testmask.eval()
        test_embeddings = testmask.mask([torch.rand(2, 1, 512) for _ in range(4)])
        # No tensors should be masked
        self.assertTrue(all((torch.count_nonzero(emb) > 0).item() for emb in test_embeddings))
        self.assertFalse((torch.count_nonzero(testmask.dropper(torch.rand(2, 1, 512))) == 0).item())

    def test_max_masked_concepts(self):
        # Test with masking being applied to only one embedding
        onemask = DisentangleNet(num_classes=20, use_masking=True, mask_probability=1.0, max_masked_concepts=1)
        onemask_embeddings = onemask.mask([torch.rand(2, 1, 512) for _ in range(4)])
        ismasked = [(torch.count_nonzero(emb) == 0).item() for emb in onemask_embeddings]
        self.assertEqual(len([i for i in ismasked if i]), 1)

        # Test with masking being applied to up to three embeddings
        threemask = DisentangleNet(num_classes=20, use_masking=True, mask_probability=1.0, max_masked_concepts=3)
        threemask_embeddings = threemask.mask([torch.rand(2, 1, 512) for _ in range(4)])
        ismasked = [(torch.count_nonzero(emb) == 0).item() for emb in threemask_embeddings]
        self.assertGreaterEqual(len([i for i in ismasked if i]), 1)

    def test_resnet(self):
        # Iterate over both resnet types and layer configurations
        for resnet_cls, resnet_layers in zip(["resnet18", "resnet34"], [[2, 2, 2, 2], [3, 4, 6, 3]]):
            # Create the DisentangleNet with this resnet type
            rn = DisentangleNet(num_classes=20, _use_resnet=True, _resnet_cls=resnet_cls)
            # Number of feature channels should always be 512
            self.assertEqual(rn.concept_channels, 512)
            # Iterate over all the concept types
            for concept in ["melody", "harmony", "rhythm", "dynamics"]:
                # Get the concept resnet
                concept_rn = getattr(rn, f'{concept}_concept')
                # We should replace the FC with an Identity block
                self.assertIsInstance(concept_rn.fc, Identity)
                # We should replace the default 3 channel input with 1 channel
                self.assertEqual(concept_rn.conv1.in_channels, 1)
                # Iterate over all the layers
                for layer_num in range(1, 5):
                    # We should have the expected number of blocks in each layer
                    rn_layer = getattr(concept_rn, f'layer{layer_num}')
                    self.assertEqual(len(rn_layer), resnet_layers[layer_num - 1])


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
