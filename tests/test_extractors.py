#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for extractor classes in deep_pianist_identification/extractors.py."""

import os
import unittest

from pretty_midi import PrettyMIDI, Instrument, Note

from deep_pianist_identification.extractors import (
    ExtractorError, MelodyExtractor, HarmonyExtractor, DynamicsExtractor, RhythmExtractor, RollExtractor
)
from deep_pianist_identification.utils import (
    get_project_root, CLIP_LENGTH, PIANO_KEYS, MIDI_OFFSET, seed_everything
)

EXTRACTORS = [MelodyExtractor, HarmonyExtractor, DynamicsExtractor, RhythmExtractor]


class ExtractorTest(unittest.TestCase):
    # Load a test MIDI file: 45 seconds long, containing some scales with a variety of velocities
    fname = os.path.join(get_project_root(), 'tests/resources/test_midi_long.MID')
    long_midi = PrettyMIDI(fname)
    # Get metadata from notes in original midi file
    input_notes = [n for n in long_midi.instruments[0].notes if n.start <= CLIP_LENGTH]
    input_starts = sorted([n.start for n in input_notes])
    input_ends = sorted([n.end for n in input_notes])
    input_pitches = sorted([n.pitch for n in input_notes])
    input_velocities = sorted([n.velocity for n in input_notes])
    # Expected number of notes, after truncating those outside the clip range
    input_notes = len([i for i in input_starts if i <= CLIP_LENGTH])

    def test_output_midi(self):
        # The final note in this MIDI occurs at ~45 seconds, so it should be removed with 30 second clips
        final_note = self.long_midi.instruments[0].notes[-1]
        # Iterate over all extractor objects
        for extractor in EXTRACTORS:
            # Create the extractor, get the input and output MIDI files
            rendered = extractor(self.long_midi, clip_start=0.0)
            input_notes = rendered.input_midi.instruments[0].notes
            output_notes = rendered.output_midi.instruments[0].notes
            # Output notes should be changed
            self.assertFalse(input_notes == output_notes)
            # Number of notes should be reduced in output
            self.assertLessEqual(len(output_notes), len(input_notes))
            # Final note should be removed in output
            self.assertNotIn(final_note, output_notes)
            # No notes should be outside clip boundaries in output
            self.assertTrue(all(n.start <= CLIP_LENGTH for n in output_notes))
            self.assertTrue(all(n.start >= 0 for n in output_notes))
            # Output notes should be within the range acceptable for the piano
            self.assertTrue(all(n.pitch <= MIDI_OFFSET + PIANO_KEYS for n in output_notes))
            self.assertTrue(all(n.pitch >= MIDI_OFFSET for n in output_notes))

    def test_melody_extractor(self):
        # Create extractor and get note data
        extractor = MelodyExtractor(self.long_midi, clip_start=0.0)
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
        extractor = HarmonyExtractor(self.long_midi, clip_start=0.0, remove_highest_pitch=False)
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

    def test_harmony_remove_highest_pitch(self):
        # Create the pretty MIDI object
        midi = PrettyMIDI()
        instr = Instrument(program=0)
        instr.notes.extend([
            Note(pitch=25, start=1.1, end=1.5, velocity=60),
            Note(pitch=30, start=1.1, end=2.5, velocity=60),
            Note(pitch=35, start=1.2, end=3.5, velocity=60),
            Note(pitch=38, start=1.1, end=4.5, velocity=60),  # highest pitch: removed
            Note(pitch=43, start=5.6, end=6.5, velocity=60),
            Note(pitch=46, start=5.6, end=6.5, velocity=60),
            Note(pitch=51, start=5.7, end=7.5, velocity=60),  # highest pitch: removed
            Note(pitch=60, start=10.0, end=11.0, velocity=60)
        ])
        midi.instruments.append(instr)
        # Test extractor that will remove top pitches in each chord
        extractor = HarmonyExtractor(midi, clip_start=0.0, remove_highest_pitch=True, quantize_resolution=0.3)
        expected_pitches = [25, 30, 35]
        actual_pitches = sorted([i.pitch for i in extractor.output_midi.instruments[0].notes])
        self.assertEqual(expected_pitches, actual_pitches)
        # Test extractor that will maintain top pitches in each chord
        extractor = HarmonyExtractor(midi, clip_start=0.0, remove_highest_pitch=False, quantize_resolution=0.3)
        expected_pitches = [25, 30, 35, 38, 43, 46, 51]
        actual_pitches = sorted([i.pitch for i in extractor.output_midi.instruments[0].notes])
        self.assertEqual(expected_pitches, actual_pitches)

    def test_dynamics_extractor(self):
        # Create extractor and get note data
        extractor = DynamicsExtractor(self.long_midi, clip_start=0.0)
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

    def test_rhythm_extractor(self):
        # Create extractor and get note data
        extractor = RhythmExtractor(self.long_midi, clip_start=0.0)
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

    def test_jittering(self):
        # Create a test MIDI object
        test_midi = PrettyMIDI()
        inst = Instrument(program=0)
        inst.notes.extend(
            [Note(pitch=50, start=5, end=6, velocity=60), Note(pitch=50, start=10, end=11, velocity=1)]
        )
        test_midi.instruments.append(inst)
        # We should crop out the first MIDI note
        clip_start = 9.0
        output_midi = RollExtractor(test_midi, clip_start=clip_start).output_midi
        self.assertGreater(len(test_midi.instruments[0].notes), len(output_midi.instruments[0].notes))
        # Get the final note for both our test and output MIDI
        input_note = test_midi.instruments[0].notes[-1]
        output_note = output_midi.instruments[0].notes[-1]
        # The start and end time of the output notes should be adjusted to match the start of the clip
        self.assertTrue(output_note.start == input_note.start - clip_start)
        self.assertTrue(output_note.end == input_note.end - clip_start)

    def test_bad_clips(self):
        # Contains no instrument objects
        bad_empty = PrettyMIDI()
        self.assertRaises(ExtractorError, lambda x: MelodyExtractor(x, clip_start=0.0), bad_empty)
        # Not a PrettyMIDI object
        bad_wrongtype = dict(i_am="bad")
        self.assertRaises(ExtractorError, lambda x: MelodyExtractor(x, clip_start=0.0), bad_wrongtype)
        # Contains an instrument object, but no notes
        bad_nonotes = PrettyMIDI()
        badinst = Instrument(program=0)
        bad_nonotes.instruments.append(badinst)
        self.assertRaises(ExtractorError, lambda x: RhythmExtractor(x, clip_start=0.0), bad_nonotes)
        # Create a new MIDI object with only two notes
        bm = PrettyMIDI()
        badinst = Instrument(program=0)
        badinst.notes.extend(
            [Note(pitch=50, start=5, end=6, velocity=60), Note(pitch=50, start=10, end=11, velocity=1)]
        )
        bm.instruments.append(badinst)
        # This clip should always raise an ExtractorError when extracting features, as there won't be any chords
        self.assertRaises(ExtractorError, lambda x: HarmonyExtractor(x), bm)


if __name__ == '__main__':
    seed_everything(42)
    unittest.main()
