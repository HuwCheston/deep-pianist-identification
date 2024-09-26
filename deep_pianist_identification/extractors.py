#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extractor classes for splitting a MIDI piano roll into melody, harmony, rhythm, and dynamics components"""

from copy import deepcopy
from itertools import groupby
from operator import itemgetter

import numpy as np
from numba import jit
from pretty_midi import PrettyMIDI, Instrument, Note

from deep_pianist_identification.utils import PIANO_KEYS, CLIP_LENGTH, FPS, MIDI_OFFSET

MINIMUM_FRAMES = 1  # a note must last this number of frames to be used
QUANTIZE_RESOLUTION = (1 / FPS) * 10  # 100 ms, the duration of a triplet eighth note at the mean JTD tempo (200 BPM)

__all__ = [
    "normalize_array", "get_multichannel_piano_roll", "get_piano_roll", "quantize", "get_singlechannel_piano_roll",
    "MelodyExtractor", "DynamicsExtractor", "RhythmExtractor", "HarmonyExtractor",
]


@jit(nopython=True)
def normalize_array(track_array: np.ndarray) -> np.ndarray:
    return (track_array - np.min(track_array)) / (np.max(track_array) - np.min(track_array))


@jit(nopython=True, fastmath=True)
def get_piano_roll(clip_notes) -> np.ndarray:
    """Largely the same as PrettyMIDI.get_piano_roll, but with optimizations using Numba"""
    clip_array = np.zeros((PIANO_KEYS, CLIP_LENGTH * FPS), dtype=np.float32)
    frame_iter = np.linspace(0, CLIP_LENGTH, (CLIP_LENGTH * FPS) + 1)
    for frame_idx, (frame_start, frame_end) in enumerate(zip(frame_iter, frame_iter[1:])):
        for note in clip_notes:
            note_start, note_end, note_pitch, note_velocity = note
            if note_start < frame_end and note_end >= frame_start:
                clip_array[note_pitch - MIDI_OFFSET, frame_idx] = note_velocity
    return clip_array


def quantize(midi_time: float, resolution: float = QUANTIZE_RESOLUTION) -> float:
    return np.round(midi_time / resolution) * resolution


class BaseExtractor:
    """Base extractor class from which all others inherit"""

    def __init__(self, midi_obj: PrettyMIDI):
        self.input_midi = self.remove_invalid_midi_notes(midi_obj)
        self.output_midi = None  # Will be added in child classes after processing

    @staticmethod
    def remove_invalid_midi_notes(midi_obj: PrettyMIDI) -> PrettyMIDI:
        temp = PrettyMIDI(resolution=midi_obj.resolution)
        newinstruments = []
        for instrument in midi_obj.instruments:
            newnotes = []
            for note in instrument.notes:
                note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
                if all((
                        note_end <= CLIP_LENGTH,
                        (note_end - note_start) > (MINIMUM_FRAMES / FPS),
                        note_pitch < PIANO_KEYS + MIDI_OFFSET,
                        note_pitch >= MIDI_OFFSET
                )):
                    newnotes.append(Note(start=note_start, end=note_end, pitch=note_pitch, velocity=note_velocity))
            newinstrument = Instrument(program=instrument.program)
            newinstrument.notes = newnotes
            newinstruments.append(newinstrument)
            temp.instruments = newinstruments
        return temp

    def get_midi_events(self):
        """Overwritten in child classes"""
        pass

    def create_output_midi(self, notes: list[tuple]) -> PrettyMIDI:
        """Converts a list of notes represented as tuples into a new PrettyMIDI object that can be synthesized, etc."""
        # Create empty PrettyMIDI and Instrument classes
        pm_obj = PrettyMIDI(resolution=self.input_midi.resolution)
        instr = Instrument(program=self.input_midi.instruments[0].program)
        # Iterate through each note and convert them to PrettyMIDI format
        for note in notes:
            note_start, note_end, note_pitch, note_velocity = note
            instr.notes.append(Note(start=note_start, end=note_end, pitch=note_pitch, velocity=note_velocity))
        # Set the notes to the instrument object, and the instrument object to the PrettyMIDI object
        pm_obj.instruments.append(instr)
        return pm_obj


class MelodyExtractor(BaseExtractor):
    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        super().__init__(midi_obj)
        # Get parameter settings from kwargs
        self.lower_bound = kwargs.get('lower_bound', MIDI_OFFSET)  # Use all notes by default
        self.quantize_resolution = kwargs.get('quantize_resolution', 1 / FPS)  # Snap to nearest 10 ms
        # Load the MIDI events, quantize, skyline, and snap the duration to fill the complete excerpt
        quantized = list(self.get_midi_events())
        skylined = list(self.apply_skyline(quantized))
        adjusted = list(self.adjust_durations(skylined))
        # Get the piano roll
        self.roll = get_piano_roll(adjusted)
        if kwargs.get('create_output_midi', False):
            self.output_midi = self.create_output_midi(adjusted)

    def get_midi_events(self) -> tuple:
        """Quantize note start and end, randomize velocity, and keep pitch"""
        for note in self.input_midi.instruments[0].notes:
            note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
            if note_pitch >= self.lower_bound:
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                note_velocity = 127  # 1 = note on, 0 = note off
                yield note_start, note_end, note_pitch, note_velocity

    @staticmethod
    def apply_skyline(quantized_notes: list):
        """For all notes with the same (quantized) start time, keep only the highest pitch note"""
        note_starts = itemgetter(0)
        # Group all the MIDI events by the (quantized) start time
        grouper = groupby(sorted(quantized_notes, key=note_starts), note_starts)
        # Iterate through each group and keep only the note with the highest pitch
        for quantized_note_start, subiter in grouper:
            # TODO: check this is getting the highest note, not the lowest!
            yield max(subiter, key=itemgetter(2))

    @staticmethod
    def adjust_durations(skylined_notes: list):
        """For a list of notes N and piece with duration T, adjust duration of notes n = T / len(N)"""
        # TODO: we'll get a divide by 0 here if we have no notes in the clip!
        note_duration = CLIP_LENGTH / len(skylined_notes)
        for note_idx, note in enumerate(skylined_notes):
            # Keep the pitch and (randomized) velocity of each note
            _, __, note_pitch, note_velocity = note
            # Adjust the note start and end time so that the duration of each note is the same
            note_start = note_idx * note_duration
            note_end = (note_idx + 1) * note_duration
            if note_idx == len(skylined_notes):
                note_end = CLIP_LENGTH
            yield note_start, note_end, note_pitch, note_velocity


class HarmonyExtractor(BaseExtractor):
    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        super().__init__(midi_obj)
        # Get the parameter settings from the kwargs, with defaults
        self.upper_bound = kwargs.get('upper_bound', PIANO_KEYS + MIDI_OFFSET)  # Use all notes by default
        self.minimum_notes = kwargs.get('minimum_notes', 3)  # Only consider triads or greater
        self.quantize_resolution = kwargs.get('quantize_resolution', 0.3)  # Snap to 300ms, i.e. 1/4 note at mean JTD
        # Quantize the MIDI events, group into chords, then adjust the durations
        quantized = list(self.get_midi_events())
        chorded = list(self.group_chords(quantized))
        adjusted = list(self.adjust_durations(chorded))
        # Create the piano roll
        self.roll = get_piano_roll(adjusted)
        if kwargs.get('create_output_midi', False):
            self.output_midi = self.create_output_midi(adjusted)

    def get_midi_events(self) -> tuple:
        """Quantize note start and end, randomize velocity, and keep pitch"""
        for note in self.input_midi.instruments[0].notes:
            note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
            if note_pitch <= self.upper_bound:
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                yield note_start, note_end, note_pitch, note_velocity

    def group_chords(self, notes: list):
        """Group simultaneous notes with at least MINIMUM_NOTES into chords (iterables of notes)"""
        note_starts = itemgetter(0)
        # Group all the MIDI events by the (quantized) start time
        grouper = groupby(sorted(notes, key=note_starts), note_starts)
        # Iterate through each group and keep only the note with the highest pitch
        for quantized_note_start, subiter in grouper:
            chord = list(subiter)
            # Get the unique pitches in the chord
            pitches = set(c[2] for c in chord)
            # Keep the chord if it has at least MINIMUM_NOTES unique pitches (i.e., not counting duplicates)
            if all((
                    len(pitches) >= self.minimum_notes,
                    # max(pitches) - min(pitches) <= self.MAX_SPAN
            )):
                yield chord

    @staticmethod
    def adjust_durations(chords: list):
        """For a list of notes N and piece with duration T, adjust duration of notes n = T / len(N)"""
        chord_duration = CLIP_LENGTH / len(chords)
        # Iterate through every chord (this is an iterable of notes)
        for note_idx, chord in enumerate(chords):
            # Get the adjusted note start and end time
            chord_start = note_idx * chord_duration
            chord_end = (note_idx + 1) * chord_duration
            chord_velocity = 127  # 1 = note on, 0 = note off
            if chord_end > CLIP_LENGTH:
                chord_end = CLIP_LENGTH
            # Iterate through each note in the chord and set the start and end time correctly, keeping pitch + velocity
            for note in chord:
                _, __, note_pitch, ___ = note
                yield float(chord_start), float(chord_end), note_pitch, chord_velocity


class RhythmExtractor(BaseExtractor):
    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        super().__init__(midi_obj)
        midi = list(self.get_midi_events())
        self.roll = get_piano_roll(midi)
        if kwargs.get('create_output_midi', False):
            self.output_midi = self.create_output_midi(midi)

    def get_midi_events(self) -> tuple:
        """Keep note onset and offset times, randomize pitch and velocity values"""
        for note in self.input_midi.instruments[0].notes:
            note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
            # Randomise the pitch and velocity of the note
            note_pitch = np.random.randint(MIDI_OFFSET, (PIANO_KEYS + MIDI_OFFSET) - 1)
            note_velocity = 127  # 1 = note on, 0 = note off
            yield note_start, note_end, note_pitch, note_velocity


class DynamicsExtractor(BaseExtractor):
    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        super().__init__(midi_obj)
        # Get parameter settings from kwargs
        self.quantize_resolution = kwargs.get('quantize_resolution', 1 / FPS)
        # Quantize the MIDI notes, randomize the pitch, and adjust the duration to fill the excerpt
        quantized = list(self.get_midi_events())
        adjusted = list(self.adjust_durations(quantized))
        self.roll = get_piano_roll(adjusted)
        if kwargs.get('create_output_midi', False):
            self.output_midi = self.create_output_midi(adjusted)

    def get_midi_events(self) -> tuple:
        """Keep note onset and offset times, randomize pitch and velocity values"""
        for note in self.input_midi.instruments[0].notes:
            note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
            # Randomise the pitch and velocity of the note
            note_pitch = np.random.randint(MIDI_OFFSET, (PIANO_KEYS + MIDI_OFFSET) - 1)
            # Snap note start and end with the given time resolution
            note_start = quantize(note_start, self.quantize_resolution)
            note_end = quantize(note_end, self.quantize_resolution)
            yield note_start, note_end, note_pitch, note_velocity

    @staticmethod
    def adjust_durations(notes: list):
        note_starts = itemgetter(0)
        # Group by the note start time and get the duration of each quantized bin
        grouper = groupby(sorted(notes, key=note_starts), note_starts)
        note_duration = CLIP_LENGTH / len(list(deepcopy(grouper)))
        # Iterate through the notes in each quantized bin
        for note_idx, (quantized_note_start, subiter) in enumerate(grouper):
            # Set the start and end time for this bin
            note_start = note_idx * note_duration
            note_end = (note_idx + 1) * note_duration
            # Iterate through each individual note in the bin
            for note in subiter:
                # Maintain the velocity and (randomised) pitch
                _, __, note_pitch, note_velocity = note
                yield note_start, note_end, note_pitch, note_velocity


def get_multichannel_piano_roll(midi_obj: PrettyMIDI, use_concepts: list = None, normalize: bool = False) -> np.ndarray:
    """For a single MIDI clip, make a four channel piano roll corresponding to Melody, Harmony, Rhythm, and Dynamics"""
    # Create all the extractors for our piano roll
    concept_mapping = {0: MelodyExtractor, 1: HarmonyExtractor, 2: RhythmExtractor, 3: DynamicsExtractor}
    if use_concepts is None:
        use_concepts = [0, 1, 2, 3]
    extractors = []
    for concept in use_concepts:
        extractor = concept_mapping[concept]
        extractors.append(extractor(midi_obj, create_output_midi=False))
    # Normalize velocity dimension if required and stack arrays
    return np.array([normalize_array(concept.roll) if normalize else concept.roll for concept in extractors])


def get_singlechannel_piano_roll(pm_obj: PrettyMIDI, normalize: bool = False) -> np.ndarray:
    """Gets a single channel piano roll, including all data (i.e., without splitting into concepts)"""
    # We can use the base extractor, which will perform checks e.g., removing invalid notes
    extract = BaseExtractor(pm_obj)
    extracted_notes = extract.input_midi.instruments[0].notes
    # Get valid notes, and then create the piano roll (single channel) from these
    roll = get_piano_roll([(note.start, note.end, note.pitch, note.velocity) for note in extracted_notes])
    # Squeeze to create a new channel, for parity with our multichannel approach (i.e., batch, channel, pitch, time)
    roll = np.expand_dims(roll, axis=0)
    if normalize:
        roll = normalize_array(roll)
    return roll


def create_outputs_from_extractors(track_name: str, extractors: list, raw_midi: PrettyMIDI) -> None:
    """Creates .WAV and .PNG files for a particular track with extractors and raw_midi"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import subprocess

    ref_folder = os.path.join(get_project_root(), 'reports/midi_test', track_name)
    if not os.path.exists(ref_folder):
        os.makedirs(ref_folder)

    rhythm, melody, harmony, dynamics = extractors
    fig, ax = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(10, 10))
    # Remove extra MIDI notes
    init_roll = get_singlechannel_piano_roll(raw_midi, normalize=True)
    xt = range(0, 3500, 500)
    yt = range(0, PIANO_KEYS, 12)
    for a, roller, tit in zip(
            ax.flatten(),
            [init_roll[0], melody.roll, harmony.roll, rhythm.roll, dynamics.roll],
            ['Raw', 'Melody', 'Harmony', 'Rhythm', 'Dynamics']
    ):
        sns.heatmap(normalize_array(roller), ax=a, mask=roller == 0, cmap='mako', cbar=None)
        a.patch.set_edgecolor('black')
        a.patch.set_linewidth(1)
        a.set(title=tit, yticks=yt, xticks=xt, xticklabels=xt, yticklabels=yt)
        a.grid()
    ax[0].invert_yaxis()
    fig.supylabel('MIDI Note')
    fig.supxlabel('Frame')
    fig.tight_layout()
    fig.savefig(os.path.join(ref_folder, 'midi_test.png'))

    t = BaseExtractor(raw_midi)
    for mid, name in zip(
            [raw_midi, t.input_midi, melody.output_midi, rhythm.output_midi, harmony.output_midi, dynamics.output_midi],
            ['input', 'truncated', 'melody', 'rhythm', 'harmony', 'dynamics']
    ):
        mid.write(os.path.join(ref_folder, f'{name}.mid'))
        cmd = f'fluidsynth -ni -r 44100 -F "{ref_folder}/{name}.wav" "{ref_folder}/{name}.mid"'
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    import os
    from time import time
    from tqdm import tqdm
    from deep_pianist_identification.utils import get_project_root
    from loguru import logger

    # Get the tracks we'll create clips from
    tracks_to_test = 50
    use_in_test = np.random.choice(os.listdir(os.path.join(get_project_root(), 'data/clips/pijama')), tracks_to_test)
    # We'll store benchmark results in here
    res = {'rhythm': [], 'melody': [], 'harmony': [], 'dynamics': [], 'all': []}
    # Iterate through each track
    for track_idx, test_track in tqdm(enumerate(use_in_test), desc=f'Making {tracks_to_test} clips...',
                                      total=tracks_to_test):
        # Create the filepath and skip if it does not exist (e.g., `.gitkeep` files)
        test_fpath = os.path.join(get_project_root(), 'data/clips/pijama', test_track, 'clip_000.mid')
        if not os.path.isfile(test_fpath):
            continue

        # Load in the MIDI object
        test_midi_obj = PrettyMIDI(test_fpath)
        allstart = time()
        track_extractors = []
        # Iterate through all extractor objects and their names
        for ex, n in zip([RhythmExtractor, MelodyExtractor, HarmonyExtractor, DynamicsExtractor], list(res.keys())):
            # Create the extractor and append the time to the list
            start = time()
            extracted = ex(test_midi_obj, create_output_midi=True)
            res[n].append(time() - start)
            track_extractors.append(extracted)
        res['all'].append(time() - allstart)
        # For every 10 items, create a graph and WAV files
        if track_idx % 10 == 0:
            create_outputs_from_extractors(test_track, track_extractors, test_midi_obj)
    # Don't use the timer from the first result as the Numba functions will be compiling then
    time_results = ', '.join(f'{i}: mean {np.mean(v[1:]):.2f}s' for i, v in res.items())
    logger.info(f'Results -  {time_results}')
