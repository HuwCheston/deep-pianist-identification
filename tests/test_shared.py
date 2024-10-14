#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for shared NN modules in deep_pianist_identification/encoders/shared.py"""

import unittest

import torch

from deep_pianist_identification.encoders.shared import *
from deep_pianist_identification.utils import seed_everything, SEED


class SharedTest(unittest.TestCase):
    def test_masked_avg_pool(self):
        pool = MaskedAvgPool()
        # 0, 1, 0, 2 === avg 0.75, masked avg 1.5
        x = torch.cat([
            torch.zeros(1, 512, 1),
            torch.ones(1, 512, 1),
            torch.zeros(1, 512, 1),
            torch.full((1, 512, 1), 2)
        ], dim=2)
        expected = torch.full((1, 512, 1), 1.5)
        actual = pool(x)
        self.assertTrue(torch.all(torch.eq(expected, actual)).item())

    def test_linearflatten(self):
        lf = LinearFlatten()
        features, channels = 256, 4
        x = torch.rand(2, features, channels)
        expected = (2, features * channels, 1)
        actual = lf(x)
        self.assertEqual(actual.size(), expected)

    def test_conv1x1(self):
        cv1x1 = Conv1x1(in_channels=4)
        x = torch.rand(2, 512, 4)
        actual = cv1x1(x)
        expected = (2, 512, 1)
        self.assertEqual(actual.size(), expected)

    def test_gem(self):
        # With p = 1.0, GeM should be equal to average pooling, so just test that
        gem = GeM(p=1.0)
        x = torch.rand(6, 512, 4)
        expected = torch.nn.functional.avg_pool1d(x, x.size(-1))
        actual = gem(x)
        self.assertTrue(torch.all(torch.isclose(expected, actual)).item())

    def test_gru(self):
        gru = GRU(input_size=512 * 11, hidden_size=256)
        # Just check the size of the GRU is correct
        x = torch.rand(1, 512, 11, 375)
        actual = gru(x)
        # i.e., (batch, features, width)
        expected = (1, 512, 375)
        self.assertEqual(actual.size(), expected)

    def test_convlayer(self):
        # Test convolutional layer with pooling
        cv_nopool = ConvLayer(in_channels=64, out_channels=128, has_pool=False)
        x = torch.rand(1, 64, 11, 375)
        expected = (1, 128, 11, 375)
        actual = cv_nopool(x)
        self.assertEqual(actual.size(), expected)
        self.assertEqual(len(cv_nopool), 4)
        # Test convolutional layer without pooling
        cv_pool = ConvLayer(in_channels=64, out_channels=128, has_pool=True, pool_kernel_size=4)
        x = torch.rand(1, 64, 44, 1500)
        actual = cv_pool(x)  # We can keep the expected size the same
        self.assertEqual(actual.size(), expected)
        self.assertEqual(len(cv_pool), 5)

    def test_linearlayer(self):
        ll = LinearLayer(in_channels=64, out_channels=25)
        x = torch.rand(1, 64)
        actual = ll(x)
        expected = (1, 25)
        self.assertEqual(actual.size(), expected)

    def test_tripletmarginloss(self):
        # Set the batch size and margin
        batch_size = 4
        margin = 0.01
        # Compute random anchor and positive vectors
        anc = torch.rand(batch_size, 512)
        pos = torch.rand(batch_size, 512)
        loss = TripletMarginLoss(margin=margin, similarity="cosine")
        # Compute the similarity matrix
        actual = loss.compute_similarity_matrix(anc, pos)
        self.assertEqual(actual.size()[0], batch_size)  # Size should just be the batch size
        self.assertTrue(all([-1. <= i <= 1. for i in actual.flatten()]))  # All values should be between -1 and 1
        # Compute the loss matrix
        loss_matrix = loss.compute_loss(actual)
        self.assertTrue(all([i == 0. for i in loss_matrix.diagonal().unsqueeze(1)]))  # All diagonals should be 0
        # Set one of the values so it falls outside the margin and recompute matrix
        actual[0, 1] = actual[0, 0] - (margin * 2)
        loss_matrix = loss.compute_loss(actual)
        self.assertEqual(loss_matrix[0, 1].item(), 0)  # this value should now be set to 0


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
