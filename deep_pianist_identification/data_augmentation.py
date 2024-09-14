#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Some simple data augmentation functions"""

import numpy as np
from pretty_midi import PrettyMIDI, Note, Instrument

from deep_pianist_identification.utils import MIDI_OFFSET, PIANO_KEYS, CLIP_LENGTH


def data_augmentation_transpose(pretty: PrettyMIDI, shift_limit: int = 6) -> PrettyMIDI:
    """Randomly transposes all notes in a MIDI file by +/- shif_limit"""
    # Pick a random value to shift by
    shift_val = np.random.randint(-shift_limit, shift_limit)
    temp = PrettyMIDI(resolution=pretty.resolution)
    newinstruments = []
    for instrument in pretty.instruments:
        newnotes = []
        for note in instrument.notes:
            newpitch = note.pitch + shift_val
            if newpitch >= 109 or newpitch <= 21:
                continue
            else:
                newnote = Note(
                    pitch=newpitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity
                )
                newnotes.append(newnote)
        newinstrument = Instrument(program=instrument.program)
        newinstrument.notes = newnotes
        newinstruments.append(newinstrument)
    temp.instruments = newinstruments
    return temp


def data_augmentation_pitch_occlude(pretty: PrettyMIDI, pitch_range: int = 6) -> PrettyMIDI:
    """Occludes all pitches within a random one octave range"""
    # Pick a random value to shift by
    shift_val = np.random.randint(MIDI_OFFSET, MIDI_OFFSET + PIANO_KEYS)
    temp = PrettyMIDI(resolution=pretty.resolution)
    newinstruments = []
    for instrument in pretty.instruments:
        newnotes = []
        for note in instrument.notes:
            if not (shift_val - pitch_range) < note.pitch < (shift_val + pitch_range):
                newnote = Note(
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity
                )
                newnotes.append(newnote)
        newinstrument = Instrument(program=instrument.program)
        newinstrument.notes = newnotes
        newinstruments.append(newinstrument)
    temp.instruments = newinstruments
    return temp


def data_augmentation_time_occlude(pretty: PrettyMIDI, time_range: float = 2.5) -> PrettyMIDI:
    """Occludes all notes within a given time range"""
    # Pick a random value to shift by
    shift_val = np.random.randint(0 + time_range, CLIP_LENGTH - time_range)
    temp = PrettyMIDI(resolution=pretty.resolution)
    newinstruments = []
    for instrument in pretty.instruments:
        newnotes = []
        for note in instrument.notes:
            if not (shift_val - time_range) < note.start < (shift_val + time_range):
                newnote = Note(
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity
                )
                newnotes.append(newnote)
        newinstrument = Instrument(program=instrument.program)
        newinstrument.notes = newnotes
        newinstruments.append(newinstrument)
    temp.instruments = newinstruments
    return temp


def data_augmentation_velocity_change(pretty: PrettyMIDI, velocity_limit: int = 12) -> PrettyMIDI:
    """Randomly changes note velocity by +/- velocity_limit"""
    temp = PrettyMIDI(resolution=pretty.resolution)
    newinstruments = []
    for instrument in pretty.instruments:
        newnotes = []
        for note in instrument.notes:
            # Pick a random value to shift by
            shift_val = np.random.randint(-velocity_limit, velocity_limit)
            if note.velocity + shift_val >= 127:
                shift_val = 0
            newnote = Note(
                pitch=note.pitch + shift_val,
                start=note.start,
                end=note.end,
                velocity=note.velocity + shift_val
            )
            newnotes.append(newnote)
        newinstrument = Instrument(program=instrument.program)
        newinstrument.notes = newnotes
        newinstruments.append(newinstrument)
    temp.instruments = newinstruments
    return temp


# add new augmentation functions in here
augment_funcs = [
    data_augmentation_transpose,
    data_augmentation_velocity_change,
    data_augmentation_time_occlude,
    data_augmentation_pitch_occlude,
]
