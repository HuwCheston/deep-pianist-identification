#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aebersold Scale Syllabus - Extracted Scales

Scale names and notes extracted from the Scale Syllabus
"""

from dataclasses import dataclass
from typing import List


NOTE_NAMES = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


_SCALE_SYLLABUS_NOTES = {
    # 1. MAJOR SCALE
    "major": ["C", "D", "E", "F", "G", "A", "B",],
    "major_pentatonic": ["C", "D", "E", "G", "A",],
    "lydian": ["C", "D", "E", "F#", "G", "A", "B",],
    "bebop_major": ["C", "D", "E", "F", "G", "G#", "A", "B",],
    "harmonic_major": ["C", "D", "E", "F", "G", "Ab", "B",],
    "lydian_augmented": ["C", "D", "E", "F#", "G#", "A", "B",],
    "augmented": ["C", "D#", "E", "G", "Ab", "B",],
    "sixth_mode_harmonic_minor": ["C", "D#", "E", "F#", "G", "A", "B",],
    # "diminished_major": ["C", "Db", "D#", "E", "F#", "G", "A", "Bb",],
    "blues_scale": ["C", "Eb", "F", "F#", "G", "Bb",],

    # 2. DOMINANT 7th
    "dominant_7th": ["C", "D", "E", "F", "G", "A", "Bb",],
    # "major_pentatonic_dom7": ["C", "D", "E", "G", "A",],
    "bebop_dominant": ["C", "D", "E", "F", "G", "A", "Bb", "B",],
    "spanish_jewish": ["C", "Db", "E", "F", "G", "Ab", "Bb",],
    "lydian_dominant": ["C", "D", "E", "F#", "G", "A", "Bb",],
    "hindu": ["C", "D", "E", "F", "G", "Ab", "Bb",],
    "whole_tone": ["C", "D", "E", "F#", "G#", "Bb",],
    "diminished_dom7": ["C", "Db", "D#", "E", "F#", "G", "A", "Bb",],
    "diminished_whole_tone": ["C", "Db", "D#", "E", "F#", "G#", "Bb",],
    # "blues_scale_dom7": ["C", "Eb", "F", "F#", "G", "Bb",],

    # DOMINANT 7th SUSPENDED 4th
    # "dominant_7th_sus4": ["C", "D", "E", "F", "G", "A", "Bb",],
    # "major_pentatonic_sus4": ["C", "D", "F", "G", "Bb"],
    # "bebop_sus4": ["C", "D", "E", "F", "G", "A", "Bb", "B",],

    # 3. MINOR SCALE
    "dorian": ["C", "D", "Eb", "F", "G", "A", "Bb",],
    "minor_pentatonic": ["C", "Eb", "F", "G", "Bb",],
    "bebop_minor": ["C", "D", "Eb", "E", "F", "G", "A", "Bb",],
    "melodic_minor": ["C", "D", "Eb", "F", "G", "A", "B",],
    "bebop_minor_no2": ["C", "D", "Eb", "F", "G", "G#", "A", "B",],
    # "blues_scale_minor": ["C", "Eb", "F", "F#", "G", "Bb",],
    "harmonic_minor": ["C", "D", "Eb", "F", "G", "Ab", "B",],
    "diminished_minor": ["C", "D", "Eb", "F", "F#", "G#", "A", "B",],
    "phrygian": ["C", "Db", "Eb", "F", "G", "Ab", "Bb",],
    "aeolian": ["C", "D", "Eb", "F", "G", "Ab", "Bb",],

    # 4. HALF DIMINISHED
    "locrian": ["C", "Db", "Eb", "F", "Gb", "Ab", "Bb",],
    "locrian_sharp2": ["C", "D", "Eb", "F", "Gb", "Ab", "Bb",],
    "bebop_half_dim": ["C", "Db", "Eb", "F", "Gb", "G", "Ab", "Bb",],

    # 5. DIMINISHED SCALE + CHROMATIC
    "diminished_8tone": ["C", "D", "Eb", "F", "Gb", "Ab", "A", "B",],
    # TODO: should chromatic be included?
    # "chromatic": ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",],
}


class Scale:
    family: str = None
    notes: list[str] = None

    @property
    def binary_distribution(self) -> List[int]:
        return [1 if i in self.note_numbers else 0 for i in range(12)]

    @property
    def binary_matrix(self) -> List[List[int]]:
        return [rotate(self.binary_distribution, i) for i in range(12)]

    @property
    def note_numbers(self) -> List[int]:
        raise NotImplementedError()

    def get_diatonic_scale_steps(self, notes: List[str]) -> List[int]:
        pass

    def diatonic_scale_steps_to_notes(self, scale_steps: List[int]) -> List[int]:
        pass


class Major(Scale):
    family = "major"
    notes = ["C", "D", "E", "F", "G", "A", "B"]


class MajorPentatonic(Scale):
    family = "major"
    notes = ["C", "D", "E", "G", "A"]


class Lydian(Scale):
    family = "major"
    notes = ["C", "D", "E", "F#", "G", "A", "B"]


class BebopMajor(Scale):
    family = "major"
    notes = ["C", "D", "E", "F", "G", "G#", "A", "B"]


class HarmonicMajor(Scale):
    family = "major"
    notes = ["C", "D", "E", "F", "G", "Ab", "B"]


class LydianAugmented(Scale):
    family = "major"
    notes = ["C", "D", "E", "F#", "G#", "A", "B"]


class Augmented(Scale):
    family = "major"
    notes = ["C", "D#", "E", "G", "Ab", "B"]


class SixthModeHarmonicMinor(Scale):
    family = "major"
    notes = ["C", "D#", "E", "F#", "G", "A", "B"]


class Blues(Scale):
    family = "major"
    notes = ["C", "Eb", "F", "F#", "G", "Bb"]


class Dominant7th(Scale):
    family = "dominant_7th"
    notes = ["C", "D", "E", "F", "G", "A", "Bb"]


class BebopDominant(Scale):
    family = "dominant_7th"
    notes = ["C", "D", "E", "F", "G", "A", "Bb", "B"]


class SpanishJewish(Scale):
    family = "dominant_7th"
    notes = ["C", "Db", "E", "F", "G", "Ab", "Bb"]


class LydianDominant(Scale):
    family = "dominant_7th"
    notes = ["C", "D", "E", "F#", "G", "A", "Bb"]


class Hindu(Scale):
    family = "dominant_7th"
    notes = ["C", "D", "E", "F", "G", "Ab", "Bb"]


class WholeTone(Scale):
    family = "dominant_7th"
    notes = ["C", "D", "E", "F#", "G#", "Bb"]


class DiminishedDom7th(Scale):
    family = "dominant_7th"
    notes = ["C", "Db", "D#", "E", "F#", "G", "A", "Bb"]


class DiminishedWholeTone(Scale):
    family = "dominant_7th"
    notes = ["C", "Db", "D#", "E", "F#", "G#", "Bb"]


class Dorian(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "F", "G", "A", "Bb"]


class MinorPentatonic(Scale):
    family = "minor"
    notes = ["C", "Eb", "F", "G", "Bb"]


class BebopMinor(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "E", "F", "G", "A", "Bb"]


class MelodicMinor(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "F", "G", "A", "B"]


class BebopMinorNo2(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "F", "G", "G#", "A", "B"]


class HarmonicMinor(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "F", "G", "Ab", "B"]


class DiminishedMinor(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "F", "F#", "G#", "A", "B"]


class Phrygian(Scale):
    family = "minor"
    notes = ["C", "Db", "Eb", "F", "G", "Ab", "Bb"]


class Aeolian(Scale):
    family = "minor"
    notes = ["C", "D", "Eb", "F", "G", "Ab", "Bb"]


class Locrian(Scale):
    family = "half_diminished"
    notes = ["C", "Db", "Eb", "F", "Gb", "Ab", "Bb"]


class LocrianSharp2(Scale):
    family = "half_diminished"
    notes = ["C", "D", "Eb", "F", "Gb", "Ab", "Bb"]


class BebopHalfDiminished(Scale):
    family = "half_diminished"
    notes = ["C", "Db", "Eb", "F", "Gb", "G", "Ab", "Bb"]


class Diminished8Tone(Scale):
    family = "diminished"
    notes = ["C", "D", "Eb", "F", "Gb", "Ab", "A", "B"]


SCALE_SYLLABUS = {k: [NOTE_NAMES[v_] for v_ in v] for k, v in _SCALE_SYLLABUS_NOTES.items()}
