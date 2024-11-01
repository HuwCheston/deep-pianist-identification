#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for augmentation functions in deep_pianist_identification/explainability."""

import os
import unittest

import numpy as np

from deep_pianist_identification import utils
from deep_pianist_identification.explainability.cav_dataloader import VoicingLoaderReal


class CAVVoicingTest(unittest.TestCase):
    # This test works, but my computer doesn't have enough memory to run it without getting a MemoryError
    # def test_real_fake_dataloaders(self):
    #     # Create both real and fake dataloaders
    #     r = VoicingLoaderReal(1)
    #     f = VoicingLoaderFake(1, n_clips=len(r))
    #     # Both dataloaders should have the same length
    #     self.assertEqual(len(r), len(f))
    #     # Neither dataloader should overlap
    #     self.assertTrue(len(list(set(r.cav_midis) & set(f.cav_midis))) == 0)

    def test_combine_hands(self):
        # Test a left-hand with multiple notes
        lh = [(0.0, 2.0, 40, 1), (0.0, 2.0, 50, 1), (4.0, 6.0, 60, 1)]
        rh = [(0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1), (4.0, 6.0, 70, 1)]
        # Upper left hand note should be added to the right hand
        expected_lh = [(0.0, 2.0, 40, 1), (4.0, 6.0, 60, 1)]
        expected_rh = [(0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1), (4.0, 6.0, 70, 1)]
        actual_lh, actual_rh = VoicingLoaderReal.combine_hands(lh, rh)
        self.assertEqual(actual_lh, expected_lh)
        self.assertEqual(actual_rh, expected_rh)

        # Test a left-hand with only single notes per chord
        lh = [(0.0, 2.0, 40, 1), (4.0, 6.0, 60, 1)]
        rh = [(0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1), (4.0, 6.0, 70, 1)]
        # Upper left hand note should be added to the right hand
        expected_lh = lh
        expected_rh = rh
        actual_lh, actual_rh = VoicingLoaderReal.combine_hands(lh, rh)
        self.assertEqual(actual_lh, expected_lh)
        self.assertEqual(actual_rh, expected_rh)

    def test_get_inversions(self):
        # Test with only a single chord
        rh = [(0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1)]
        expected = [
            [(0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1)],  # Root
            [(0.0, 2.0, 60, 1), (0.0, 2.0, 62, 1), (0.0, 2.0, 70, 1)],  # First inversion
            [(0.0, 2.0, 62, 1), (0.0, 2.0, 70, 1), (0.0, 2.0, 72, 1)]  # Second inversion
            # No third inversion as would be identical to root (but up an octave)
        ]
        actual = VoicingLoaderReal.get_inversions(rh)
        self.assertEqual(actual, expected)

        # Test with multiple chords
        rh = [
            (0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1),  # chord 1
            (2.0, 4.0, 50, 1), (2.0, 4.0, 60, 1), (2.0, 4.0, 70, 1), (2.0, 4.0, 80, 1)  # chord 2
        ]
        expected = [
            # Root positions
            [
                (0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1),
                (2.0, 4.0, 50, 1), (2.0, 4.0, 60, 1), (2.0, 4.0, 70, 1), (2.0, 4.0, 80, 1)
            ],
            # First inversions
            [
                (0.0, 2.0, 60, 1), (0.0, 2.0, 62, 1), (0.0, 2.0, 70, 1),
                (2.0, 4.0, 60, 1), (2.0, 4.0, 62, 1), (2.0, 4.0, 70, 1), (2.0, 4.0, 80, 1)
            ],
            # Second inversions
            [
                (0.0, 2.0, 62, 1), (0.0, 2.0, 70, 1), (0.0, 2.0, 72, 1),
                (2.0, 4.0, 62, 1), (2.0, 4.0, 70, 1), (2.0, 4.0, 72, 1), (2.0, 4.0, 80, 1)
            ]
        ]
        actual = VoicingLoaderReal.get_inversions(rh)
        self.assertEqual(actual, expected)

    def test_adjust_durations_and_transposition(self):
        # Test with no transposition
        rh = [
            (0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1),  # chord 1
            (2.0, 4.0, 50, 1), (2.0, 4.0, 60, 1), (2.0, 4.0, 70, 1)  # chord 2
        ]
        expected = [
            (0.0, 15.0, 50, 1), (0.0, 15.0, 60, 1), (0.0, 15.0, 70, 1),  # chord 1
            (15.0, 30.0, 50, 1), (15.0, 30.0, 60, 1), (15.0, 30.0, 70, 1)  # chord 2
        ]
        actual = VoicingLoaderReal.adjust_durations_and_transpose(rh, transpose_value=0)
        self.assertEqual(actual, expected)
        # Test with transposition -2 semitones
        expected = [
            (0.0, 15.0, 48, 1), (0.0, 15.0, 58, 1), (0.0, 15.0, 68, 1),  # chord 1
            (15.0, 30.0, 48, 1), (15.0, 30.0, 58, 1), (15.0, 30.0, 68, 1)  # chord 2
        ]
        actual = VoicingLoaderReal.adjust_durations_and_transpose(rh, transpose_value=-2)
        self.assertEqual(actual, expected)
        # Test with three chords and +3 semitones
        rh = [
            (0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1),  # chord 1
            (2.0, 4.0, 50, 1), (2.0, 4.0, 60, 1), (2.0, 4.0, 70, 1),  # chord 2
            (4.0, 9.0, 50, 1), (4.0, 9.0, 60, 1), (4.0, 9.0, 70, 1)  # chord 3
        ]
        expected = [
            (0.0, 10.0, 53, 1), (0.0, 10.0, 63, 1), (0.0, 10.0, 73, 1),  # chord 1
            (10.0, 20.0, 53, 1), (10.0, 20.0, 63, 1), (10.0, 20.0, 73, 1),  # chord 2
            (20.0, 30.0, 53, 1), (20.0, 30.0, 63, 1), (20.0, 30.0, 73, 1)  # chord 3
        ]
        actual = VoicingLoaderReal.adjust_durations_and_transpose(rh, transpose_value=3)
        self.assertEqual(actual, expected)

    def test_midi_creation(self):
        r = VoicingLoaderReal(1, transpose_range=1, use_rootless=True)
        # This is simply the notes C - C - E - G - B (major 7)
        tester = os.path.join(
            utils.get_project_root(),
            'references/_unsupervised_resources/voicings/midi_final/'
            '1_cav/1.mid'
        )
        created = r._create_midi(tester)
        # With transposing by +/-1 semitones, we have 2 * 3 * 4 options (both rootless/with root and all inversions)
        self.assertEqual(len(created), 24)
        # Test shape of each individual created MIDI
        for t in created:
            self.assertTrue(t.shape == (utils.PIANO_KEYS, utils.CLIP_LENGTH * utils.FPS))
            # Get the number of unique notes: should be either 4 or 5 depending on if LH is included
            n_notes = np.unique(t, axis=1).sum()
            self.assertTrue(n_notes == 4 or n_notes == 5)
        # Disable using rootless voicing and re-create the MIDI
        r.use_rootless = False
        created = r._create_midi(tester)
        # Without using rootless voicings, we should only have 3 * 4 clips
        self.assertEqual(len(created), 12)
        for t in created:
            # Should always be 5 notes in this case as the bass note should always be there
            n_notes = np.unique(t, axis=1).sum()
            self.assertTrue(n_notes == 5)


if __name__ == '__main__':
    unittest.main()
