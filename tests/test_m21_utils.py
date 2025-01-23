#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for working with music21 objects"""

import unittest
from unittest import TestCase

from deep_pianist_identification.preprocessing import m21_utils


class TestM21Utils(TestCase):
    def test_center_feature_below_threshold(self):
        feature = [50, 52, 55, 57, 59]  # Mean is 54.6, below threshold
        expected = [62, 64, 67, 69, 71]  # Transposed by one octave (12 semitones)
        result = m21_utils.center_feature(feature, threshold=60)
        self.assertEqual(result, expected)

    def test_center_feature_above_threshold(self):
        feature = [62, 64, 66, 68, 70]  # Mean is 66, above threshold
        expected = feature  # No transposition expected
        result = m21_utils.center_feature(feature, threshold=64)
        self.assertEqual(result, expected)

    def test_center_feature_multiple_octaves(self):
        feature = [40, 42, 45, 47, 49]  # Mean is 44.6, far below threshold
        expected = [64, 66, 69, 71, 73]  # Transposed by two octaves
        result = m21_utils.center_feature(feature, threshold=60)
        self.assertEqual(result, expected)

    def test_convert_partitura_note(self):
        pt_note = ('C', 1, 5)
        expected = 'C#5'
        actual = m21_utils.partitura_note_to_name(pt_note)
        self.assertEqual(actual, expected)
        pt_note = ('D', -1, 7)
        expected = 'D-7'
        actual = m21_utils.partitura_note_to_name(pt_note)
        self.assertEqual(actual, expected)
        pt_note = ('E', 0, 2)
        expected = 'E2'
        actual = m21_utils.partitura_note_to_name(pt_note)
        self.assertEqual(actual, expected)

    def test_intervals_to_pitches(self):
        feature = [-1, -1, -1]
        expected = (0, -1, -2, -3)
        actual = m21_utils.intervals_to_pitches(feature)
        self.assertEqual(actual, expected)
        feature = "[1, 3, -1]"
        expected = (0, 1, 4, 3)
        actual = m21_utils.intervals_to_pitches(feature)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
