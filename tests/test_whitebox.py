#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for whitebox models (feature extraction and modelling)"""

import os
import unittest

import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note

import deep_pianist_identification.utils as utils
from deep_pianist_identification.whitebox.features import (
    get_valid_features_by_track_usage, drop_invalid_features_by_track_usage, format_features
)


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
        actual = get_valid_features_by_track_usage(tester, min_count=3)
        self.assertEqual(predicted, actual)
        # This is what the results should look like once we've dropped all the invalid n-grams
        predicted_dropped = [
            {"[1, 2, 3]": 3},
            {"[1, 2, 3]": 5},
            {},
            {"[1, 2, 3]": 3}
        ]
        actual_dropped = drop_invalid_features_by_track_usage(tester, predicted)
        self.assertEqual(predicted_dropped, actual_dropped)
        # Test dropping n-grams that appear in too many tracks
        predicted = ["[2, 3, 4]", "[3, 4]", "[3, 4, 5]"]
        actual = get_valid_features_by_track_usage(tester, min_count=1, max_count=2)
        self.assertEqual(predicted, actual)
        # Results should look like this
        predicted_dropped = [
            {"[2, 3, 4]": 2},
            {},
            {"[3, 4]": 1},
            {"[3, 4, 5]": 2}
        ]
        actual_dropped = drop_invalid_features_by_track_usage(tester, predicted)
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
        from deep_pianist_identification.whitebox.features import extract_valid_melody_ngrams

        # Create the pretty MIDI object
        midi = PrettyMIDI()
        instr = Instrument(program=0)
        instr.notes.extend([
            Note(pitch=20, start=1, end=1.5, velocity=60),
            Note(pitch=21, start=2, end=2.5, velocity=60),
            Note(pitch=23, start=3, end=3.5, velocity=60),
            Note(pitch=26, start=4, end=4.5, velocity=60),
            Note(pitch=25, start=5, end=5.5, velocity=60),
            Note(pitch=24, start=6, end=6.5, velocity=60),
        ])
        midi.instruments.append(instr)
        # These are the n-grams we expect to extract from the PrettyMIDI object
        expected_3grams = [[1, 2, 3], [2, 3, -1], [3, -1, -1]]
        expected_4grams = [[1, 2, 3, -1], [2, 3, -1, -1]]
        # These are the n-grams we are actually extracting
        actual_3grams = list(extract_valid_melody_ngrams(midi, 4))
        actual_4grams = list(extract_valid_melody_ngrams(midi, 5))
        # Should both be the same
        self.assertEqual(expected_3grams, actual_3grams)
        self.assertEqual(expected_4grams, actual_4grams)

    def test_validate_ngram(self):
        from deep_pianist_identification.whitebox.features import validate_ngram

        # Should pass the test
        passing = [
            Note(pitch=20, start=1, end=1.5, velocity=60),
            Note(pitch=21, start=2, end=2.5, velocity=60),
            Note(pitch=23, start=3, end=3.5, velocity=60),
        ]
        self.assertTrue(validate_ngram(passing))
        # Should fail as one note is too high, making the span too wide
        fail_span = [
            Note(pitch=20, start=1, end=1.5, velocity=60),
            Note(pitch=21, start=2, end=2.5, velocity=60),
            Note(pitch=53, start=3, end=3.5, velocity=60),
        ]
        self.assertFalse(validate_ngram(fail_span))
        # Should fail as the rest between note 2 and 3 is too long
        fail_duration = [
            Note(pitch=20, start=1, end=1.5, velocity=60),
            Note(pitch=21, start=2, end=2.5, velocity=60),
            Note(pitch=22, start=6.0, end=8.0, velocity=60),
        ]
        self.assertFalse(validate_ngram(fail_duration))

    def test_extract_ngram_counts(self):
        from deep_pianist_identification.whitebox.features import extract_melody_ngrams

        # Define the testing track
        tester = os.path.join(utils.get_project_root(), 'tests/resources')
        # Extract 3 and 4 grams
        getting_34grams, _ = extract_melody_ngrams(tester, nclips=1, pianist=18, ngrams=[3, 4])
        extracted_grams = [eval(k) for k in getting_34grams.keys()]
        # Remember: these ngrams are expressed as interval counts, so a 3-gram is actually 4 "notes"
        self.assertTrue(all(len(eg) == 3 or len(eg) == 4 for eg in extracted_grams))
        # Extract 3 grams only
        getting_3grams, _ = extract_melody_ngrams(tester, nclips=1, pianist=18, ngrams=[3])
        extracted_grams = [eval(k) for k in getting_3grams.keys()]
        self.assertTrue(all(len(eg) == 3 for eg in extracted_grams))

    def test_harmony_chord_extract(self):
        from deep_pianist_identification.whitebox.features import (
            extract_chords, HARMONY_MAX_LEAPS_IN_CHORD, HARMONY_MAX_LEAP
        )

        # Define the testing track
        tester = os.path.join(utils.get_project_root(), 'tests/resources')
        # Extract the ngrams, making sure to remove leaps
        restrict_ngrams, _ = extract_chords(tester, nclips=1, pianist=18, feature_sizes=[3])
        # Iterate through all the n-grams
        for ng, _ in restrict_ngrams.items():
            # Convert the interval classes from strings to integers
            ics = [abs(int(i.replace('[', '').replace(']', ''))) for i in ng.split(', ')]
            # Quick check that we're only extracting 3-gram
            self.assertEqual(len(ics), 3)
            # Should not have more than MAX_LEAPS_IN_CHORD leaps of MAX_LEAP semitones
            above_thresh = [i1 for i1, i2 in zip(ics, ics[1:]) if i2 - i1 >= HARMONY_MAX_LEAP]
            self.assertTrue(len(above_thresh) < HARMONY_MAX_LEAPS_IN_CHORD)

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
        actual_chords_transpose = list(_extract_fn_harmony(midi))
        self.assertEqual(expected_chords_transpose, actual_chords_transpose)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
