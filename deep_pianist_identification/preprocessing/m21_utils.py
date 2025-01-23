#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for working with musical notation in music21 and partitura"""

import os

import numpy as np
import partitura as pt
from music21.chord import Chord
from music21.clef import TrebleClef, BassClef
from music21.duration import Duration
from music21.meter import TimeSignature
from music21.note import Note, Rest
from music21.stream import Part, Measure, Score
from pretty_midi import note_number_to_name

from deep_pianist_identification import utils


def invisible_rest(duration: float = 0.5):
    """Add an (invisible) rest with a custom duration"""
    spacer = Rest()
    spacer.duration = Duration(duration)
    spacer.style.hideObjectOnPrint = True  # Ensure it's invisible in the score
    return spacer


def invisible_time_signature(numerator: int = 2, denominator: int = 4):
    """Returns a (hidden) time signature with a given numerator"""
    ts = TimeSignature(f"{numerator}/{denominator}")
    ts.style.hideObjectOnPrint = True  # hides the time signature
    return ts


def center_feature(feature: list[int], threshold: int = utils.MIDDLE_C):
    """Centers the notes in a feature by transposing them by octaves until the mean pitch is below the threshold"""
    # Note that we're expecting our feature to be a list of integers here, i.e. that returned by .format_feature
    while np.mean(feature) <= threshold:
        feature = [i + utils.OCTAVE for i in feature]
    return feature


def melody_feature_to_m21_measure(feature: list[int]) -> Measure:
    """Converts a feature (list of integers) to a music21 Measure, populated with the desired notes"""
    # Melody plots only use treble clef, with the time signature set according to the number of notes
    measure = Measure([TrebleClef(), invisible_time_signature(len(feature) + 4), invisible_rest(0.5)])
    for midi_number in feature:
        midi_name = note_number_to_name(midi_number)
        measure.append(Note(midi_name, quarterLength=1))
    return measure


def partitura_note_to_name(partitura_note: tuple) -> str:
    """Parses a structured note object from partitura (in the format note, accidental, octave) to a MIDI note name"""
    note, accidental, octave = partitura_note
    # if the accidental is flat
    if accidental == -1:
        accidental_fmt = '-'
    # If it is sharp
    elif accidental == 1:
        accidental_fmt = '#'
    # otherwise, no accidental
    else:
        accidental_fmt = ''
    # parse everything as a MIDI note, e.g. C#5
    return f'{note}{accidental_fmt}{octave}'


def estimate_m21_score_spelling(score: Score) -> list[str]:
    """Gets the estimated spelling for all notes within a `score` object using partitura"""
    # Export to MusicXML file
    musicxml_path = 'tmp_to_partitura.xml'
    score.write('musicxml', fp=musicxml_path)
    # Load the Musicxml in partitura and remove the temporary file
    parti = pt.load_musicxml(musicxml_path)
    os.remove(musicxml_path)
    # Estimate the spelling and convert the structured array to a list of MIDI note names
    estimated_spelling = pt.musicanalysis.pitch_spelling.estimate_spelling(parti)
    return [partitura_note_to_name(n) for n in estimated_spelling]


def measure_to_part(measure: Measure) -> Part:
    """Converts a measure to a part and sets the final barline to be invisible"""
    part = Part(measure)
    # Remove the final bar line from the part
    part[-1].rightBarline = None
    return part


def part_to_score(part: Part | list) -> Score:
    """Converts a part (or list of parts) to a score"""
    if not isinstance(part, list):
        part = [part]
    return Score(part)


def measure_to_chord(measure: Measure) -> Chord:
    """Convert all music21 notes in a measure to a chord while preserving non-note elements"""
    is_note = [i for i in measure if isinstance(i, Note)]
    is_not_note = [i for i in measure if not isinstance(i, Note)]
    return Measure([*is_not_note, Chord(is_note)])


def adjust_m21_melody_to_partitura(m21_notes: Score, pt_notes: list) -> None:
    for m21_note, pt_note in zip(m21_notes.recurse().notes, pt_notes):
        if str(m21_note.pitch) != pt_note:
            m21_note.pitch.getEnharmonic(inPlace=True)


def adjust_m21_hands_to_partitura(m21_notes: Score, pt_notes: list) -> None:
    rh, lh = m21_notes
    counter = 0  # Counter used to get the required partitura note from our list
    # Iterate in order of left hand, right hand (by default, music21 would do the inverse)
    for part in [lh, rh]:
        # Iterate over every element
        for element in part.recurse():
            # If we have a chord, iterate over each note
            if isinstance(element, Chord):
                for m21_note in element:
                    # If partitura is saying we should spell the note differently
                    if str(m21_note.pitch) != pt_notes[counter]:
                        # Get the enharmonic spelling, adjusted in place
                        m21_note.pitch.getEnharmonic(inPlace=True)
                    counter += 1


def intervals_to_pitches(feature: str | list[int]) -> tuple:
    """Formats the feature from a string to a tuple"""
    if isinstance(feature, str):
        feature = eval(feature)
    pitch_set = [0]
    current_pitch = 0
    for interval in feature:
        current_pitch += interval
        pitch_set.append(current_pitch)
    return tuple(pitch_set)


def feature_to_separate_hands(feature: list[int]) -> tuple[Measure, Measure]:
    """Populate two separate measure objects with notes from the feature: one each for left/right hands"""
    # The 2/4 time signature means we can have one invisible 1/4 note rest and one 1/4 note for the chord itself
    right_hand = Measure([TrebleClef(), invisible_time_signature(2), invisible_rest(0.5)])
    left_hand = Measure([BassClef(), invisible_time_signature(2), invisible_rest(0.5)])
    # Iterate over every note in the feature
    for midi_number in feature:
        # Converting to music21 format
        midi_name = note_number_to_name(midi_number)
        note = Note(midi_name, quarterLength=1)
        # Notes at and below middle C go to left hand, everything else goes to right
        if midi_number <= utils.MIDDLE_C:
            left_hand.append(note)
        else:
            right_hand.append(note)
    return right_hand, left_hand
