#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for augmentation functions in deep_pianist_identification/explainability."""

import os
import unittest

import numpy as np
import torch

from deep_pianist_identification import utils
from deep_pianist_identification.explainability import cav_utils


class VoicingLoaderTest(unittest.TestCase):
    def test_real_fake_dataloaders(self):
        # Create both real and fake dataloaders
        try:
            r = cav_utils.VoicingLoaderReal(1, n_clips=10)
            f = cav_utils.VoicingLoaderFake(1, n_clips=len(r))
        # Little hack for my local machine which doesn't have enough memory to create the above classes
        except (OSError, MemoryError):
            self.fail("Not enough system resources to create dataloaders!")
        # Test __getitem__ functions return expected output
        for target_class, loader in enumerate([f, r]):
            roll, cls = loader.__getitem__(0)
            self.assertEqual(roll.shape, (4, utils.PIANO_KEYS, utils.FPS * utils.CLIP_LENGTH))
            self.assertEqual(cls, target_class)
        # Both dataloaders should have the same length
        self.assertEqual(len(r), len(f))
        # Neither dataloader should overlap
        self.assertTrue(len(list(set(r.cav_midis) & set(f.cav_midis))) == 0)
        # Fake dataloader should use fewer than 20 concepts (target concept should be missing)
        assert len(set([int(i.split(os.path.sep)[-2].replace('_cav', '')) for i in f.cav_midis])) < 20

    def test_fake_dataloader_any_midi(self):
        # Setting cav_number=None to this class will allow us to use any MIDI file in the dataloader
        try:
            loader = cav_utils.VoicingLoaderFake(None, n_clips=250)
        # Little hack for my local machine which doesn't have enough memory to create the above classes
        except (OSError, MemoryError):
            self.fail("Not enough system resources to create dataloaders!")
        # Fake dataloader should use all 20 concepts
        assert len(set([int(i.split(os.path.sep)[-2].replace('_cav', '')) for i in loader.cav_midis])) == 20

    def test_combine_hands(self):
        # Test a left-hand with multiple notes
        lh = [(0.0, 2.0, 40, 1), (0.0, 2.0, 50, 1), (4.0, 6.0, 60, 1)]
        rh = [(0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1), (4.0, 6.0, 70, 1)]
        # Upper left hand note should be added to the right hand
        expected_lh = [(0.0, 2.0, 40, 1), (4.0, 6.0, 60, 1)]
        expected_rh = [(0.0, 2.0, 50, 1), (0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1), (4.0, 6.0, 70, 1)]
        actual_lh, actual_rh = cav_utils.VoicingLoaderReal.combine_hands(lh, rh)
        self.assertEqual(actual_lh, expected_lh)
        self.assertEqual(actual_rh, expected_rh)

        # Test a left-hand with only single notes per chord
        lh = [(0.0, 2.0, 40, 1), (4.0, 6.0, 60, 1)]
        rh = [(0.0, 2.0, 60, 1), (0.0, 2.0, 70, 1), (4.0, 6.0, 70, 1)]
        # Upper left hand note should be added to the right hand
        expected_lh = lh
        expected_rh = rh
        actual_lh, actual_rh = cav_utils.VoicingLoaderReal.combine_hands(lh, rh)
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
        actual = cav_utils.VoicingLoaderReal.get_inversions(rh)
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
        actual = cav_utils.VoicingLoaderReal.get_inversions(rh)
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
        actual = cav_utils.VoicingLoaderReal.adjust_durations_and_transpose(rh, transpose_value=0)
        self.assertEqual(actual, expected)
        # Test with transposition -2 semitones
        expected = [
            (0.0, 15.0, 48, 1), (0.0, 15.0, 58, 1), (0.0, 15.0, 68, 1),  # chord 1
            (15.0, 30.0, 48, 1), (15.0, 30.0, 58, 1), (15.0, 30.0, 68, 1)  # chord 2
        ]
        actual = cav_utils.VoicingLoaderReal.adjust_durations_and_transpose(rh, transpose_value=-2)
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
        actual = cav_utils.VoicingLoaderReal.adjust_durations_and_transpose(rh, transpose_value=3)
        self.assertEqual(actual, expected)

    def test_midi_creation(self):
        r = cav_utils.VoicingLoaderReal(1, transpose_range=1, use_rootless=True)
        # This is simply the notes C - C - E - G - B (major 7)
        tester = os.path.join(
            utils.get_project_root(),
            'references/cav_resources/voicings/midi_final/'
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


class ExplainerTest(unittest.TestCase):
    def test_metrics(self):
        # Test sign counts metric works as expected
        test_vector = torch.tensor([-1.1, 1.2, 0.1, -5.5])
        expected = 0.5
        actual = cav_utils.CAV.get_sign_count(test_vector)
        self.assertAlmostEqual(expected, actual)
        # Test magnitude metric works
        test_vector = torch.tensor([1.2, 0.8, -1.1, -0.9, -1., -1.])
        expected = 1 / 3
        actual = cav_utils.CAV.get_magnitude(test_vector)
        self.assertAlmostEqual(expected, actual)

    # TODO: rewrite these tests
    # def test_pvals(self):
    #     # Create two arrays, one with a much higher mean than another
    #     a1 = np.array([[9.1, 7.7, 8.8, 9.3, 5.5]])
    #     a2 = np.array([[0.1, 0.2, 0.6, 0.7, 0.3]])
    #     # Get p-values: these should be significant at a low value of p
    #     pval1 = cav_utils.get_pvals(a1, a2)
    #     self.assertLess(pval1[0], 1e-4)
    #     # Create another array, similar to the first
    #     a3 = np.array([[0.1, 0.3, 0.6, 0.7, 0.1]])
    #     # Should not be significant at a low value of p
    #     pval2 = cav_utils.get_pvals(a2, a3)
    #     self.assertGreater(pval2[0], 1e-4)
    #
    # def test_pvals_to_asterisks(self):
    #     # First, test without correction
    #     p_matrix = np.array([
    #         [0.04, 0.03, 0.012, 0.045],
    #         [0.54, 0.009, 0.44, 0.99],
    #         [1e-9, 1e-7, 1e-4, 1e-8]
    #     ])
    #     expected = np.array([
    #         ['*', '*', '*', '*'],
    #         ['', '**', '', ''],
    #         ['***', '***', '***', '***']
    #     ])
    #     ast_matrix = cav_utils.get_pval_significance(p_matrix, bonferroni=False)
    #     self.assertEqual(ast_matrix.shape, expected.shape)
    #     self.assertTrue(np.array_equal(ast_matrix, expected))
    #     # Now, test with correction
    #     expected = np.array([
    #         ['', '', '*', ''],
    #         ['', '*', '', ''],
    #         ['***', '***', '***', '***']
    #     ])
    #     ast_matrix = cav_utils.get_pval_significance(p_matrix, bonferroni=True)
    #     self.assertEqual(ast_matrix.shape, expected.shape)
    #     self.assertTrue(np.array_equal(ast_matrix, expected))


if __name__ == '__main__':
    unittest.main()
