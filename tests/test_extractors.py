#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for extractor classes in deep_pianist_identification/extractors.py."""

import os
import unittest

from pretty_midi import PrettyMIDI

from deep_pianist_identification.extractors import (
    get_singlechannel_piano_roll, get_multichannel_piano_roll,
    MelodyExtractor, HarmonyExtractor, DynamicsExtractor, RhythmExtractor
)
from deep_pianist_identification.utils import get_project_root, CLIP_LENGTH, PIANO_KEYS, FPS

EXTRACTORS = [MelodyExtractor, HarmonyExtractor, DynamicsExtractor, RhythmExtractor]


class ExtractorTest(unittest.TestCase):
    # Load a test MIDI file: 45 seconds long, containing some scales with a variety of velocities
    fname = os.path.join(get_project_root(), 'references/test_midi_long.MID')
    long_midi = PrettyMIDI(fname)
    # Get metadata from notes in original midi file
    input_notes = [n for n in long_midi.instruments[0].notes if n.start <= CLIP_LENGTH]
    input_starts = sorted([n.start for n in input_notes])
    input_ends = sorted([n.end for n in input_notes])
    input_pitches = sorted([n.pitch for n in input_notes])
    input_velocities = sorted([n.velocity for n in input_notes])
    # Expected number of notes, after truncating those outside the clip range
    input_notes = len([i for i in input_starts if i <= CLIP_LENGTH])

    def test_clip_boundaries(self):
        # The final note in this MIDI occurs at ~45 seconds, so it should be removed with 30 second clips
        final_note = self.long_midi.instruments[0].notes[-1]
        # Iterate over all extractor objects
        for extractor in EXTRACTORS:
            # Create the extractor, get the input and output MIDI files
            rendered = extractor(self.long_midi, create_output_midi=True)
            input_notes = rendered.input_midi.instruments[0].notes
            output_notes = rendered.output_midi.instruments[0].notes
            # Output notes should be changed
            self.assertFalse(input_notes == output_notes)
            # Final note should be removed
            self.assertIn(final_note, input_notes)
            self.assertNotIn(final_note, output_notes)
            # No notes should be outside clip boundaries
            self.assertFalse(any(n.start > CLIP_LENGTH for n in output_notes))
            self.assertFalse(any(n.start < 0 for n in output_notes))
            # Input notes can be outside clip boundary
            self.assertTrue(any(n.start > CLIP_LENGTH for n in input_notes))

    def test_multichannel_roll_shape(self):
        multichannel_roll = get_multichannel_piano_roll(self.long_midi)
        self.assertEqual((len(EXTRACTORS), PIANO_KEYS, CLIP_LENGTH * FPS), multichannel_roll.shape)

    def test_singlechannel_roll_shape(self):
        singlechannel_roll = get_singlechannel_piano_roll(self.long_midi)
        self.assertEqual((1, PIANO_KEYS, CLIP_LENGTH * FPS), singlechannel_roll.shape)

    def test_melody_extractor(self):
        # Create extractor and get note data
        extractor = MelodyExtractor(self.long_midi, create_output_midi=True)
        extracted_starts = sorted([n.start for n in extractor.output_midi.instruments[0].notes])
        extracted_ends = sorted([n.end for n in extractor.output_midi.instruments[0].notes])
        extracted_pitches = sorted([n.pitch for n in extractor.output_midi.instruments[0].notes])
        extracted_velocities = sorted([n.pitch for n in extractor.output_midi.instruments[0].notes])
        # All melody pitches should be within input
        self.assertTrue(all(e in self.input_pitches for e in extracted_pitches))
        # Onset start and end times, velocities should be different
        self.assertFalse(all(extracted in self.input_starts for extracted in extracted_starts))
        self.assertFalse(all(extracted in self.input_ends for extracted in extracted_ends))
        self.assertFalse(all(inp in extracted_velocities for inp in self.input_velocities))
        # Should be more notes in the input roll than in the extracted melody
        self.assertGreater(len(self.input_pitches), len(extracted_pitches))

    def test_harmony_extractor(self):
        # Create extractor and get note data
        extractor = HarmonyExtractor(self.long_midi, create_output_midi=True)
        extracted_starts = sorted([n.start for n in extractor.output_midi.instruments[0].notes])
        extracted_ends = sorted([n.end for n in extractor.output_midi.instruments[0].notes])
        extracted_pitches = sorted([n.pitch for n in extractor.output_midi.instruments[0].notes])
        extracted_velocities = sorted([n.pitch for n in extractor.output_midi.instruments[0].notes])
        # All pitches should be within input
        self.assertTrue(all(e in self.input_pitches for e in extracted_pitches))
        # Onset start and end times, velocities should be different
        self.assertFalse(all(extracted in self.input_starts for extracted in extracted_starts))
        self.assertFalse(all(extracted in self.input_ends for extracted in extracted_ends))
        self.assertFalse(all(inp in extracted_velocities for inp in self.input_velocities))
        # Should be more notes in the input roll than in the extracted melody
        self.assertGreater(len(self.input_pitches), len(extracted_pitches))

    def test_dynamics_extractor(self):
        # Create extractor and get note data
        extractor = DynamicsExtractor(self.long_midi, create_output_midi=True)
        extracted_starts = sorted([n.start for n in extractor.output_midi.instruments[0].notes])
        extracted_ends = sorted([n.end for n in extractor.output_midi.instruments[0].notes])
        extracted_pitches = sorted([n.pitch for n in extractor.output_midi.instruments[0].notes])
        extracted_velocities = sorted([n.velocity for n in extractor.output_midi.instruments[0].notes])
        # Onset start and end times, pitches should be different
        self.assertFalse(all(extracted in self.input_starts for extracted in extracted_starts))
        self.assertFalse(all(extracted in self.input_ends for extracted in extracted_ends))
        self.assertFalse(all(extracted in self.input_pitches for extracted in extracted_pitches))
        # Velocity should be identical
        self.assertTrue(all(extracted in self.input_velocities for extracted in extracted_velocities))
        # Number of notes should be identical
        self.assertEqual(self.input_notes, len(extracted_starts))

    def test_rhythm_extractor(self):
        # Create extractor and get note data
        extractor = RhythmExtractor(self.long_midi, create_output_midi=True)
        extracted_starts = sorted([n.start for n in extractor.output_midi.instruments[0].notes])
        extracted_ends = sorted([n.end for n in extractor.output_midi.instruments[0].notes])
        extracted_pitches = sorted([n.pitch for n in extractor.output_midi.instruments[0].notes])
        extracted_velocities = sorted([n.velocity for n in extractor.output_midi.instruments[0].notes])
        # Pitches and velocities should be different
        self.assertFalse(all(extracted in self.input_pitches for extracted in extracted_pitches))
        self.assertFalse(all(inp in extracted_velocities for inp in self.input_velocities))
        # Original and extracted onset start and end times should be identical
        self.assertTrue(all(extracted in self.input_starts for extracted in extracted_starts))
        self.assertTrue(all(extracted in self.input_ends for extracted in extracted_ends))
        # Number of onsets should be identical
        self.assertEqual(len(extracted_starts), self.input_notes)


if __name__ == '__main__':
    unittest.main()