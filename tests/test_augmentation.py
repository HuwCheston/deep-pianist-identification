#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for augmentation functions in deep_pianist_identification/data_augmentation.py."""

import os
import unittest

from pretty_midi import PrettyMIDI

from deep_pianist_identification.data_augmentation import (
    data_augmentation_transpose, data_augmentation_dilate, data_augmentation_velocity_change
)
from deep_pianist_identification.utils import get_project_root


class AugmentationTest(unittest.TestCase):
    # Load a test MIDI file: 45 seconds long, containing some scales with a variety of velocities
    fname = os.path.join(get_project_root(), 'references/test_midi_long.MID')
    long_midi = PrettyMIDI(fname)
    # Get metadata from notes in original midi file
    input_notes = long_midi.instruments[0].notes
    input_starts = sorted([n.start for n in input_notes])
    input_ends = sorted([n.end for n in input_notes])
    input_pitches = [n.pitch for n in input_notes]
    input_velocities = [n.velocity for n in input_notes]

    def test_transpose_augmentation(self):
        augmented = data_augmentation_transpose(self.long_midi)
        augmented_pitches = [i.pitch for i in augmented.instruments[0].notes]
        # Number of notes should be the same or lower, if we're transposing out of bounds
        self.assertLessEqual(len(augmented_pitches), len(self.input_pitches))
        # Augmented pitches should be different to input pitches
        self.assertFalse(all(aug == inp for aug, inp in zip(augmented_pitches, self.input_pitches)))

    def test_velocity_augmentation(self):
        augmented = data_augmentation_velocity_change(self.long_midi)
        augmented_velocities = [i.velocity for i in augmented.instruments[0].notes]
        # Number of notes should be identical
        self.assertEqual(len(augmented_velocities), len(self.input_velocities))
        # Augmented velocities should be different to input velocities
        self.assertFalse(all(aug == inp for aug, inp in zip(augmented_velocities, self.input_velocities)))

    def test_time_dilate_augmentation(self):
        augmented = data_augmentation_dilate(self.long_midi)
        augmented_starts = sorted([i.start for i in augmented.instruments[0].notes])
        augmented_ends = sorted([i.end for i in augmented.instruments[0].notes])
        # Note start and end times should be different after augmentation
        self.assertFalse(any([i in self.input_starts for i in augmented_starts]))
        self.assertFalse(any([i in self.input_ends for i in augmented_ends]))


if __name__ == '__main__':
    unittest.main()
