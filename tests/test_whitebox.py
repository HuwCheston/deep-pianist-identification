#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for whitebox models (feature extraction and modelling)"""

import os
import unittest

import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note

import deep_pianist_identification.utils as utils
from deep_pianist_identification.whitebox import wb_utils
from deep_pianist_identification.whitebox.features import get_valid_ngrams, drop_invalid_ngrams, format_features


class WhiteBoxFeatureExtractTest(unittest.TestCase):
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
        actual = get_valid_ngrams(tester, min_count=3)
        self.assertEqual(predicted, actual)
        # This is what the results should look like once we've dropped all the invalid n-grams
        predicted_dropped = [
            {"[1, 2, 3]": 3},
            {"[1, 2, 3]": 5},
            {},
            {"[1, 2, 3]": 3}
        ]
        actual_dropped = drop_invalid_ngrams(tester, predicted)
        self.assertEqual(predicted_dropped, actual_dropped)
        # Test dropping n-grams that appear in too many tracks
        predicted = ["[2, 3, 4]", "[3, 4]", "[3, 4, 5]"]
        actual = get_valid_ngrams(tester, min_count=1, max_count=2)
        self.assertEqual(predicted, actual)
        # Results should look like this
        predicted_dropped = [
            {"[2, 3, 4]": 2},
            {},
            {"[3, 4]": 1},
            {"[3, 4, 5]": 2}
        ]
        actual_dropped = drop_invalid_ngrams(tester, predicted)
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
        expected_feature_names = ["[1, 2, 3]", "[2, 3, 4]", "[2, 3, 5]"]
        actual_1, actual_2, actual_feature_names = format_features(tester_1, tester_2)
        self.assertTrue(np.array_equal(expected_1, actual_1, equal_nan=True))
        self.assertTrue(np.array_equal(expected_2, actual_2, equal_nan=True))
        self.assertTrue(expected_feature_names == actual_feature_names)

    def test_melody_ngram_extraction(self):
        from deep_pianist_identification.whitebox.features import _extract_fn_melody

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
        actual_3grams = list(_extract_fn_melody(midi, [3]))[0]
        actual_4grams = list(_extract_fn_melody(midi, [4]))[0]
        actual_34grams = list(_extract_fn_melody(midi, [3, 4]))
        # Should both be the same
        self.assertEqual(expected_3grams, actual_3grams)
        self.assertEqual(expected_4grams, actual_4grams)
        self.assertEqual(expected_34grams, actual_34grams)

    def test_remove_leaps_melody(self):
        from deep_pianist_identification.whitebox.features import extract_melody_ngrams

        # Define the testing track
        tester = os.path.join(utils.get_project_root(), 'tests/resources')
        # Extract the ngrams, making sure to remove leaps
        restrict_ngrams, _ = extract_melody_ngrams(tester, nclips=1, pianist=18, ngrams=[3], remove_leaps=True)
        # Iterate through all the n-grams
        for ng, _ in restrict_ngrams.items():
            # Convert the interval classes from strings to integers
            ics = [abs(int(i.replace('[', '').replace(']', ''))) for i in ng.split(', ')]
            # Quick check that we're only extracting 3-gram
            self.assertEqual(len(ics), 3)
            # All interval classes should be below the threshold
            for ic in ics:
                self.assertTrue(ic < wb_utils.MAX_LEAP)
        # Extract the ngrams, allowing for leaps
        all_ngrams, _ = extract_melody_ngrams(tester, nclips=1, pianist=18, ngrams=[3], remove_leaps=False)
        # Total number of ngrams should be larger when allowing for leaps
        self.assertLess(len(list(restrict_ngrams.keys())), len(list(all_ngrams.keys())))

    def test_remove_leaps_harmony(self):
        from deep_pianist_identification.whitebox.features import extract_chords, MAX_LEAPS_IN_CHORD

        # Define the testing track
        tester = os.path.join(utils.get_project_root(), 'tests/resources')
        # Extract the ngrams, making sure to remove leaps
        restrict_ngrams, _ = extract_chords(tester, nclips=1, pianist=18, ngrams=[3], remove_leaps=True)
        # Iterate through all the n-grams
        for ng, _ in restrict_ngrams.items():
            # Convert the interval classes from strings to integers
            ics = [abs(int(i.replace('[', '').replace(']', ''))) for i in ng.split(', ')]
            # Quick check that we're only extracting 3-gram
            self.assertEqual(len(ics), 3)
            # Should not have more than MAX_LEAPS_IN_CHORD leaps of MAX_LEAP semitones
            above_thresh = [i1 for i1, i2 in zip(ics, ics[1:]) if i2 - i1 >= wb_utils.MAX_LEAP]
            self.assertTrue(len(above_thresh) < MAX_LEAPS_IN_CHORD)
        # Extract the ngrams, allowing for leaps
        all_ngrams, _ = extract_chords(tester, nclips=1, pianist=18, ngrams=[3], remove_leaps=False)
        # Total number of ngrams should be larger when allowing for leaps
        self.assertLessEqual(len(list(restrict_ngrams.keys())), len(list(all_ngrams.keys())))

    def test_harmony_chord_extraction(self):
        from deep_pianist_identification.whitebox.features import _extract_fn_harmony

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
        expected_chords_transpose = [[5, 10, 15], [10, 20]]
        actual_chords_transpose = list(_extract_fn_harmony(midi, transpose=True))
        self.assertEqual(expected_chords_transpose, actual_chords_transpose)
        # These are the chords we expect to extract from the PrettyMIDI object, without transposition
        expected_chords_notranspose = [[25, 30, 35, 40], [45, 55, 65]]
        actual_chords_notranspose = list(_extract_fn_harmony(midi, transpose=False))
        self.assertEqual(expected_chords_notranspose, actual_chords_notranspose)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
