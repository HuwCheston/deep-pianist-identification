#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for baselines (random forest and C(R)NN/resnet)"""

import unittest

import numpy as np
import torch
from pretty_midi import PrettyMIDI, Instrument, Note

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
import deep_pianist_identification.utils as utils
from deep_pianist_identification.encoders import CNNet, CRNNet, ResNet50, TangCNN
from deep_pianist_identification.encoders.shared import IBN


class ForestTest(unittest.TestCase):
    def test_valid_ngrams_extract(self):
        # Each dictionary corresponds to one track; the keys are the n-grams, and the values are the count in the track
        tester = [
            {"[1, 2, 3]": 3, "[2, 3, 4]": 2},
            {"[1, 2, 3]": 5},
            {"[3, 4]": 1},
            {"[1, 2, 3]": 3, "[3, 4, 5]": 2}
        ]
        # This is the only n-gram that appears in at least three dictionaries (i.e., tracks)
        predicted = ["[1, 2, 3]"]
        actual = rf_utils.get_valid_ngrams(tester, valid_count=3)
        self.assertEqual(predicted, actual)
        # This is what the results should look like once we've dropped all the invalid n-grams
        predicted_dropped = [
            {"[1, 2, 3]": 3},
            {"[1, 2, 3]": 5},
            {},
            {"[1, 2, 3]": 3}
        ]
        actual_dropped = rf_utils.drop_invalid_ngrams(tester, predicted)
        self.assertEqual(predicted_dropped, actual_dropped)

    def test_format_features(self):
        # These are equivalent to the n-grams extracted from two splits of the dataset
        tester_1 = [
            {"[1, 2, 3]": 3, "[2, 3, 4]": 2},
            {"[1, 2, 3]": 5},
        ]
        tester_2 = [
            {"[2, 3, 4]": 2, "[2, 3, 5]": 1},
            {"[1, 2, 3]": 5},
            {"[1, 2, 3]": 3}
        ]
        # Once we format the dataset, we expect arrays looking like this
        expected_1 = np.array([
            [3, 2, 0],
            [5, 0, 0]
        ])
        expected_2 = np.array([
            [0, 2, 1],
            [5, 0, 0],
            [3, 0, 0]
        ])
        actual_1, actual_2 = rf_utils.format_features(tester_1, tester_2)
        self.assertTrue(np.array_equal(expected_1, actual_1, equal_nan=True))
        self.assertTrue(np.array_equal(expected_2, actual_2, equal_nan=True))

    def test_melody_ngram_extraction(self):
        from deep_pianist_identification.rf_baselines.melody import extract_fn

        # Create the pretty MIDI object
        midi = PrettyMIDI()
        instr = Instrument(program=0)
        instr.notes.extend([
            Note(pitch=20, start=1, end=1.5, velocity=60),
            Note(pitch=30, start=2, end=2.5, velocity=60),
            Note(pitch=35, start=3, end=3.5, velocity=60),
            Note(pitch=38, start=4, end=4.5, velocity=60),
            Note(pitch=43, start=5, end=5.5, velocity=60),
            Note(pitch=46, start=6, end=6.5, velocity=60),
            Note(pitch=51, start=7, end=7.5, velocity=60),
        ])
        midi.instruments.append(instr)
        # These are the n-grams we expect to extract from the PrettyMIDI object
        expected_3grams = [[10, 5, 3], [5, 3, 5], [3, 5, 3], [5, 3, 5]]
        expected_4grams = [[10, 5, 3, 5], [5, 3, 5, 3], [3, 5, 3, 5]]
        expected_34grams = [expected_3grams, expected_4grams]
        # These are the n-grams we are actually extracting
        actual_3grams = list(extract_fn(midi, [3]))[0]
        actual_4grams = list(extract_fn(midi, [4]))[0]
        actual_34grams = list(extract_fn(midi, [3, 4]))
        # Should both be the same
        self.assertEqual(expected_3grams, actual_3grams)
        self.assertEqual(expected_4grams, actual_4grams)
        self.assertEqual(expected_34grams, actual_34grams)

    def test_harmony_chord_extraction(self):
        from deep_pianist_identification.rf_baselines.harmony import extract_fn

        # Create the pretty MIDI object
        midi = PrettyMIDI()
        instr = Instrument(program=0)
        instr.notes.extend([
            Note(pitch=25, start=1, end=1.5, velocity=60),
            Note(pitch=30, start=1, end=2.5, velocity=60),
            Note(pitch=35, start=1, end=3.5, velocity=60),
            Note(pitch=40, start=1, end=4.5, velocity=60),
            Note(pitch=45, start=5, end=5.5, velocity=60),
            Note(pitch=55, start=5, end=6.5, velocity=60),
            Note(pitch=65, start=5, end=7.5, velocity=60),
        ])
        midi.instruments.append(instr)
        # These are the chords we expect to extract from the PrettyMIDI object, with transposition
        expected_chords_transpose = [[0, 5, 10, 15], [0, 10, 20]]
        actual_chords_transpose = list(extract_fn(midi, transpose=True))
        self.assertEqual(expected_chords_transpose, actual_chords_transpose)
        # These are the chords we expect to extract from the PrettyMIDI object, without transposition
        expected_chords_notranspose = [[25, 30, 35, 40], [45, 55, 65]]
        actual_chords_notranspose = list(extract_fn(midi, transpose=False))
        self.assertEqual(expected_chords_notranspose, actual_chords_notranspose)


class BaselineNNTest(unittest.TestCase):
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


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
