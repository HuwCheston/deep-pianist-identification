#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for augmentation functions in deep_pianist_identification/extractors.py."""

import os
import unittest

from pretty_midi import PrettyMIDI, Instrument, Note

from deep_pianist_identification.extractors import augment_midi
from deep_pianist_identification.utils import get_project_root, seed_everything, MAX_VELOCITY


class AugmentationTest(unittest.TestCase):
    # Load a test MIDI file: 45 seconds long, containing some scales with a variety of velocities
    fname = os.path.join(get_project_root(), 'tests/resources/test_midi_long.MID')
    long_midi = PrettyMIDI(fname)
    # Get metadata from notes in original midi file
    input_notes = long_midi.instruments[0].notes
    input_starts = sorted([n.start for n in input_notes])
    input_ends = sorted([n.end for n in input_notes])
    input_pitches = [n.pitch for n in input_notes]
    input_velocities = [n.velocity for n in input_notes]

    def test_transpose_augmentation(self):
        # Augment the input MIDI clip, setting a massive transpose limit to ensure we're unlikely to shift 0 semitones
        augmented = augment_midi(self.long_midi, transpose_limit=30)
        augmented_pitches = [i.pitch for i in augmented.instruments[0].notes]
        # Number of notes should be the same or lower, if we're transposing out of bounds
        self.assertLessEqual(len(augmented_pitches), len(self.input_pitches))
        # Augmented pitches should be different to input pitches
        self.assertFalse(all(aug in self.input_pitches for aug in augmented_pitches))

    def test_balance_augmentation(self):
        # Augment the input MIDI clip
        augmented = augment_midi(self.long_midi)
        augmented_velocities = [i.velocity for i in augmented.instruments[0].notes]
        # Number of notes should be identical
        self.assertEqual(len(augmented_velocities), len(self.input_velocities))
        # Augmented velocities should be different to input velocities
        self.assertFalse(all(aug == inp for aug, inp in zip(augmented_velocities, self.input_velocities)))

    def test_velocity_clipping(self):
        # Input MIDI with velocity values above 127 and below 0
        # This should never happen in reality, but we just want to test that these values are successfully clipped
        input_midi = PrettyMIDI()
        instr = Instrument(0, is_drum=False)
        instr.notes.extend([
            Note(velocity=5000, pitch=50, start=5, end=6), Note(velocity=-5000, pitch=50, start=6, end=7)
        ])
        input_midi.instruments.append(instr)
        # Apply the augmentation
        augmented = augment_midi(input_midi)
        # Test the first note is clipped to MAX_VELOCITY (127)
        loud_note = augmented.instruments[0].notes[0]
        self.assertEqual(loud_note.velocity, MAX_VELOCITY)
        # Test the second note is clipped to 0
        quiet_note = augmented.instruments[0].notes[-1]
        self.assertEqual(quiet_note.velocity, 0)

    def test_dilate_augmentation(self):
        # Augment the input MIDI clip
        augmented = augment_midi(self.long_midi)
        augmented_starts = sorted([i.start for i in augmented.instruments[0].notes])
        augmented_ends = sorted([i.end for i in augmented.instruments[0].notes])
        # Note start and end times should be different after augmentation
        self.assertFalse(any([i in self.input_starts for i in augmented_starts]))
        self.assertFalse(any([i in self.input_ends for i in augmented_ends]))


if __name__ == '__main__':
    seed_everything(42)
    unittest.main()
