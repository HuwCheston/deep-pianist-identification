#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for blackbox models"""

import unittest

import torch

from deep_pianist_identification.encoders import CNNet, TangCNN, CRNNet, ResNet50
from deep_pianist_identification.encoders.shared import IBN


class BlackboxNNTest(unittest.TestCase):
    def test_cnn(self):
        # Test max pool and BN
        cn = CNNet(num_classes=15, pool_type="max", norm_type="bn")
        self.assertEqual(cn.fc2.fc.out_features, 15)
        self.assertTrue(cn.layer4.has_pool)
        self.assertFalse(cn.layer8.has_pool)
        self.assertIsInstance(cn.layer2.norm, torch.nn.BatchNorm2d)
        self.assertIsInstance(cn.pooling, torch.nn.AdaptiveMaxPool2d)
        # Test avg pool and IBN
        cn = CNNet(num_classes=15, pool_type="avg", norm_type="ibn")
        self.assertIsInstance(cn.layer2.norm, IBN)
        self.assertIsInstance(cn.pooling, torch.nn.AdaptiveAvgPool2d)
        # Test forward
        x = torch.rand(4, 1, 88, 3000)
        expected = (4, 15)
        actual = cn(x)
        self.assertEqual(actual.size(), expected)

    def test_tang_cnn(self):
        tcnn = TangCNN(num_classes=15)
        self.assertEqual(tcnn.fc2.out_features, 15)
        # Test forward
        x = torch.rand(4, 1, 88, 3000)
        expected = (4, 15)
        actual = tcnn(x)
        self.assertEqual(actual.size(), expected)

    def test_crnn(self):
        # Test default parameters
        crn = CRNNet(num_classes=15)
        self.assertEqual(crn.fc2.fc.out_features, 15)
        self.assertTrue(crn.layer4.has_pool)
        self.assertFalse(crn.layer8.has_pool)
        self.assertIsInstance(crn.layer2.norm, torch.nn.BatchNorm2d)
        self.assertIsInstance(crn.pooling, torch.nn.AdaptiveMaxPool2d)
        # Test avg pooling
        crn = CRNNet(num_classes=15, pool_type="avg")
        self.assertIsInstance(crn.pooling, torch.nn.AdaptiveAvgPool2d)
        # Test forward
        x = torch.rand(4, 1, 88, 3000)
        expected = (4, 15)
        actual = crn(x)
        self.assertEqual(actual.size(), expected)
        # Test GRU
        x_after_conv = torch.rand(4, 512, 11, 375)
        expected_gru_size = (4, 512, 375)
        actual_gru_size = crn.gru(x_after_conv).size()
        self.assertEqual(actual_gru_size, expected_gru_size)

    def test_resnet(self):
        rn = ResNet50(num_classes=15)
        self.assertEqual(rn.fc.out_features, 15)
        self.assertIsInstance(rn.layer1[0].bn1, torch.nn.BatchNorm2d)
        # self.assertIsInstance(rn.pooling.__module__, torch.nn.AdaptiveMaxPool2d)
        # Test forward
        x = torch.rand(4, 1, 88, 3000)
        expected = (4, 15)
        actual = rn(x)
        self.assertEqual(actual.size(), expected)

    def test_forward_features(self):
        models = [ResNet50, CRNNet, CNNet]
        sizes = [2048, 512, 512]  # ResNet has 2048-dim embeddings, all others have 512-dim
        x = torch.rand(4, 1, 88, 3000)
        for mod, expected in zip(models, sizes):
            mod = mod(num_classes=20)
            actual = mod.forward_features(x).size()
            self.assertEqual(actual, (4, expected))
