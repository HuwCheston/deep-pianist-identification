#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extractor classes for splitting a MIDI piano roll into melody, harmony, rhythm, and dynamics components"""

from copy import deepcopy
from itertools import groupby
from operator import itemgetter
from typing import Optional

import numpy as np
from numba import jit
from pretty_midi import PrettyMIDI, Instrument, Note

from deep_pianist_identification.utils import PIANO_KEYS, CLIP_LENGTH, FPS, MIDI_OFFSET

MINIMUM_FRAMES = 1  # a note must last this number of frames to be used
QUANTIZE_RESOLUTION = (1 / FPS) * 10  # 100 ms, the duration of a triplet eighth note at the mean JTD tempo (200 BPM)

__all__ = [
    "normalize_array", "get_multichannel_piano_roll", "get_piano_roll", "quantize", "get_singlechannel_piano_roll",
    "MelodyExtractor", "DynamicsExtractor", "RhythmExtractor", "HarmonyExtractor", "ExtractorError", "BaseExtractor"
]


class ExtractorError(Exception):
    """Custom extractor error for catching bad clips in individual piano roll extractor classes"""
    pass


@jit(nopython=True)
def normalize_array(track_array: np.ndarray) -> np.ndarray:
    """Normalize values in a 2D array between 0 and 1"""
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
    """Snap a given `midi_time` value to the nearest `resolution` interval"""
    return np.round(midi_time / resolution) * resolution


class BaseExtractor:
    """Base extractor class from which all others inherit. Defines basic methods that are overriden in child classes"""

    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        self.input_midi = midi_obj
        # Convert all note objects into a list of tuples to work with them easier
        raw = [(note.start, note.end, note.pitch, note.velocity) for note in self.input_midi.instruments[0].notes]
        # Apply data augmentation
        augmented = list(self.augmentation(raw))
        # Remove notes which fail validation checks
        validated = list(self.validation(augmented))
        # Transform data into correct format
        transformed = list(self.transformation(validated))
        # Create output in both PrettyMIDI and piano roll format
        self.roll = get_piano_roll(transformed)
        self.output_midi = self.output(transformed)

    def validate_note(self, midi_note: tuple[float, float, int, int]) -> bool:
        """Returns True if a note fulfils all required conditions for the extractor, False if not"""
        note_start, note_end, note_pitch, note_velocity = midi_note
        return all((
            note_end <= CLIP_LENGTH,  # Does the note fit within the boundaries of our clip?
            (note_end - note_start) > (MINIMUM_FRAMES / FPS),  # If the note at least N frames long?
            note_pitch < PIANO_KEYS + MIDI_OFFSET,  # Does the note fit within the boundaries of the piano?
            note_pitch >= MIDI_OFFSET  # Does the note fit within the boundaries of the piano?
        ))

    def augmentation(self, midi_notes: list[tuple]) -> list[tuple]:
        for note in midi_notes:
            yield note

    def validation(self, augmented_midi_notes: list[tuple]) -> list[tuple]:
        for note in augmented_midi_notes:
            if self.validate_note(note):
                yield note

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        for note in validated_midi_notes:
            yield note

    def output(self, transformed_midi_notes: list[tuple]) -> PrettyMIDI:
        # Create empty PrettyMIDI and Instrument classes using the same settings as our input
        pm_obj = PrettyMIDI(resolution=self.input_midi.resolution)
        instr = Instrument(program=self.input_midi.instruments[0].program)
        # Iterate through each note and convert them to PrettyMIDI format
        for note in transformed_midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            instr.notes.append(Note(start=note_start, end=note_end, pitch=note_pitch, velocity=note_velocity))
        # Set the notes to the instrument object, and the instrument object to the PrettyMIDI object
        pm_obj.instruments.append(instr)
        return pm_obj

    def heatmap(self, ax: Optional = None):
        """Creates a heatmap of the piano roll in seaborn, returns a matplotlib Axes object for use in a Figure"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a new axis object, if none is passed
        if ax is None:
            ax = plt.gca()
        # Create the heatmap in seaborn
        sns.heatmap(normalize_array(self.roll), ax=ax, mask=self.roll == 0, cmap='mako', cbar=None)
        # Set some plot aesthetics
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        ax.grid()
        # Set axes limits correctly
        ax.set(
            xticks=range(0, 3500, 500), yticks=range(0, PIANO_KEYS, 12),
            xticklabels=range(0, 3500, 500), yticklabels=range(0, PIANO_KEYS, 12)
        )
        # Return the axes object to be incorporated in a larger plot
        return ax

    def synthesize(self, output_path: str) -> None:
        """Renders the output MIDI from the extractor as a .wav file using fluidsynth"""
        import os
        import subprocess

        # Render the transformed MIDI as a file
        self.output_midi.write(output_path + '.mid')
        # Call Fluidsynth in subprocess to render the MIDI
        cmd = f'fluidsynth -ni -r 44100 -F "{output_path}.wav" "{output_path}.mid"'
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # Remove the MIDI file
        os.remove(output_path + '.mid')


class MelodyExtractor(BaseExtractor):
    """Extracts melody using skyline algorithm, while removing rhythm, harmony, and velocity"""

    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        # Get parameter settings from kwargs
        self.lower_bound = kwargs.get('lower_bound', MIDI_OFFSET)  # Use all notes by default
        self.quantize_resolution = kwargs.get('quantize_resolution', 1 / FPS)  # Snap to nearest 10 ms
        # Augmentation settings from kwargs
        self.shift_limit = kwargs.get('shift_limit', 6)
        self.data_augmentation = kwargs.get('data_augmentation', True)
        self.augmentation_probability = kwargs.get('augmentation_probability', 0.5)
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj)

    def augmentation(self, midi_notes: list[tuple]) -> list[tuple]:
        """Randomly transposes all notes in a MIDI file by +/- shif_limit"""
        should_augment = np.random.uniform(0., 1.) <= self.augmentation_probability
        # If we're augmenting
        if self.data_augmentation and should_augment:
            # Pick a random value to shift all notes by
            shift_val = np.random.randint(-self.shift_limit, self.shift_limit)
        # No shifting of notes
        else:
            shift_val = 0
        # Iterate through all notes
        for note in midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # Shift the pitch by the desired value
            note_pitch += shift_val
            yield note_start, note_end, note_pitch, note_velocity

    def validate_note(self, midi_note: Note) -> bool:
        """Returns True if a note fulfils all required conditions for the extractor, False if not"""
        note_start, note_end, note_pitch, note_velocity = midi_note
        return all((
            note_end <= CLIP_LENGTH,  # Does the note fit within the boundaries of our clip?
            (note_end - note_start) > (MINIMUM_FRAMES / FPS),  # If the note at least N frames long?
            note_pitch < PIANO_KEYS + MIDI_OFFSET,  # Does the note fit within the boundaries of the piano?
            note_pitch >= MIDI_OFFSET,  # Does the note fit within the boundaries of the piano?
            note_pitch >= self.lower_bound  # Is the note within the specified boundary?
        ))

    def validation(self, augmented_midi_notes: list[tuple]) -> list[tuple]:
        for note in augmented_midi_notes:
            if self.validate_note(note):
                note_start, note_end, note_pitch, note_velocity = note
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                note_velocity = 127  # 1 = note on, 0 = note off
                yield note_start, note_end, note_pitch, note_velocity

    def transformation(self, validated_midi_notes: list[tuple]):
        skylined = list(self.apply_skyline(validated_midi_notes))
        return self.adjust_durations(skylined)

    @staticmethod
    def apply_skyline(quantized_notes: list):
        """For all notes with the same (quantized) start time, keep only the highest pitch note"""
        note_starts = itemgetter(0)
        # Group all the MIDI events by the (quantized) start time
        grouper = groupby(sorted(quantized_notes, key=note_starts), note_starts)
        # Iterate through each group and keep only the note with the highest pitch
        for quantized_note_start, subiter in grouper:
            yield max(subiter, key=itemgetter(2))

    @staticmethod
    def adjust_durations(skylined_notes: list):
        """For a list of notes N and piece with duration T, adjust duration of notes n = T / len(N)"""
        try:
            note_duration = CLIP_LENGTH / len(skylined_notes)
        # This error will be caught in the dataloader collate_fn and the bad clip will be skipped.
        # This logic here allows us to note which clips are broken.
        except ZeroDivisionError:
            raise ExtractorError('no notes present in clip when extracting melody.')
        else:
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
    """Extracts harmony by grouping together near-simultaneous notes, and removes rhythm, dynamics, and melody"""

    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        # Get the parameter settings from the kwargs, with defaults
        self.upper_bound = kwargs.get('upper_bound', PIANO_KEYS + MIDI_OFFSET)  # Use all notes by default
        self.minimum_notes = kwargs.get('minimum_notes', 3)  # Only consider triads or greater
        self.quantize_resolution = kwargs.get('quantize_resolution', 0.3)  # Snap to 300ms, i.e. 1/4 note at mean JTD
        # Augmentation settings from kwargs
        self.shift_limit = kwargs.get('shift_limit', 6)
        self.data_augmentation = kwargs.get('data_augmentation', True)
        self.augmentation_probability = kwargs.get('augmentation_probability', 0.5)
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj)

    def augmentation(self, midi_notes: list[tuple]) -> list[tuple]:
        """Randomly transposes all notes in a MIDI file by +/- shift_limit"""
        should_augment = np.random.uniform(0., 1.) <= self.augmentation_probability
        # If we're augmenting
        if self.data_augmentation and should_augment:
            # Pick a random value to shift all notes by
            shift_val = np.random.randint(-self.shift_limit, self.shift_limit)
        # No shifting of notes
        else:
            shift_val = 0
        # Iterate through all notes
        for note in midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # Shift the pitch by the desired value
            note_pitch += shift_val
            yield note_start, note_end, note_pitch, note_velocity

    def validate_note(self, midi_note: Note) -> bool:
        """Returns True if a note fulfils all required conditions for the extractor, False if not"""
        note_start, note_end, note_pitch, note_velocity = midi_note
        return all((
            note_end <= CLIP_LENGTH,  # Does the note fit within the boundaries of our clip?
            (note_end - note_start) > (MINIMUM_FRAMES / FPS),  # If the note at least N frames long?
            note_pitch < PIANO_KEYS + MIDI_OFFSET,  # Does the note fit within the boundaries of the piano?
            note_pitch >= MIDI_OFFSET,  # Does the note fit within the boundaries of the piano?
            note_pitch <= self.upper_bound  # Is the note within the specified boundary?
        ))

    def validation(self, augmented_midi_notes: list[tuple]) -> list[tuple]:
        """Quantize note start and end, randomize velocity, and keep pitch"""
        for note in augmented_midi_notes:
            if self.validate_note(note):
                note_start, note_end, note_pitch, note_velocity = note
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                yield note_start, note_end, note_pitch, note_velocity

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        chorded = list(self.group_chords(validated_midi_notes))
        return self.adjust_durations(chorded)

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
        try:
            chord_duration = CLIP_LENGTH / len(chords)
        # This error will be caught in the dataloader collate_fn and the bad clip will be skipped.
        # This logic here allows us to note which clips are broken.
        except ZeroDivisionError:
            raise ExtractorError('no notes present in clip when extracting harmony.')
        else:
            # Iterate through every chord (this is an iterable of notes)
            for note_idx, chord in enumerate(chords):
                # Get the adjusted note start and end time
                chord_start = note_idx * chord_duration
                chord_end = (note_idx + 1) * chord_duration
                # Set the velocity to binary
                chord_velocity = 127  # 1 = note on, 0 = note off
                if chord_end > CLIP_LENGTH:
                    chord_end = CLIP_LENGTH
                # Iterate through each note in the chord and set the times correctly, keeping pitch + velocity
                for note in chord:
                    _, __, note_pitch, ___ = note
                    yield float(chord_start), float(chord_end), note_pitch, chord_velocity


class RhythmExtractor(BaseExtractor):
    """Extracts rhythm (note onset and offset times), while shuffling pitch and removing velocity"""

    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        # Augmentation settings from kwargs
        self.shift_limit = kwargs.get('shift_limit', 0.2)
        self.data_augmentation = kwargs.get('data_augmentation', True)
        self.augmentation_probability = kwargs.get('augmentation_probability', 0.5)
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj)

    def augmentation(self, midi_notes: list[tuple]) -> list[tuple]:
        """Randomly dilate all onset and offset times through multiplying by 1 +/- shift_limit"""
        should_augment = np.random.uniform(0., 1.) <= self.augmentation_probability
        # If we're augmenting
        if self.data_augmentation and should_augment:
            # Choose a random floating point value to shift note onset and offsets by
            shift_val = np.random.uniform(1 - self.shift_limit, 1 + self.shift_limit)
        # No shifting of note onset and offset times
        else:
            shift_val = 1.
        # Iterate through all notes
        for note in midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # Shift the onset and offset times by the desired value
            note_start *= shift_val
            note_end *= shift_val
            yield note_start, note_end, note_pitch, note_velocity

    # No need to overwrite validation logic, BaseExtractor code works fine

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        for note in validated_midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # Randomise the pitch
            note_pitch = np.random.randint(MIDI_OFFSET, (PIANO_KEYS + MIDI_OFFSET) - 1)
            # Set the velocity to binary
            note_velocity = 127  # 1 = note on, 0 = note off
            yield note_start, note_end, note_pitch, note_velocity


class DynamicsExtractor(BaseExtractor):
    """Extracts velocity while randomising pitch and removing note onset and offset times"""

    def __init__(self, midi_obj: PrettyMIDI, **kwargs):
        # Get parameter settings from kwargs
        self.quantize_resolution = kwargs.get('quantize_resolution', 1 / FPS)
        # Augmentation settings from kwargs
        self.shift_limit = kwargs.get('shift_limit', 12)
        self.data_augmentation = kwargs.get('data_augmentation', True)
        self.augmentation_probability = kwargs.get('augmentation_probability', 0.5)
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj)

    def augmentation(self, midi_notes: list[tuple]) -> list[tuple]:
        """Randomly transposes all notes in a MIDI file by +/- shift_limit"""
        should_augment = np.random.uniform(0., 1.) <= self.augmentation_probability
        # If we're augmenting
        if self.data_augmentation and should_augment:
            for note in midi_notes:
                note_start, note_end, note_pitch, note_velocity = note
                # Randomly choose a value to shift the note velocity by
                shift_val = np.random.randint(-self.shift_limit, self.shift_limit)
                # If shifting the note by this value would take us outside the MIDI velocity range
                if (note_velocity + shift_val >= 127) or (note_velocity + shift_val <= 0):
                    # Set the shift value to 0 so that we keep the note
                    shift_val = 0
                note_velocity += shift_val
                yield note_start, note_end, note_pitch, note_velocity
        # Not augmenting, so just return a generator of notes
        else:
            for note in midi_notes:
                yield note

    def validation(self, augmented_midi_notes: list[tuple]) -> list[tuple]:
        """Remove invalid notes and quantize note onset and offset times"""
        for note in augmented_midi_notes:
            if self.validate_note(note):
                note_start, note_end, note_pitch, note_velocity = note
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                yield note_start, note_end, note_pitch, note_velocity

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        """Adjust note durations and randomize pitch"""
        note_starts = itemgetter(0)
        # Group by the note start time and get the duration of each quantized bin
        grouper = groupby(sorted(validated_midi_notes, key=note_starts), note_starts)
        try:
            note_duration = CLIP_LENGTH / len(list(deepcopy(grouper)))
        # This error will be caught in the dataloader collate_fn and the bad clip will be skipped.
        # This logic here allows us to note which clips are broken.
        except ZeroDivisionError:
            raise ExtractorError('no notes present in clip when extracting velocity.')
        else:
            # Iterate through the notes in each quantized bin
            for note_idx, (quantized_note_start, subiter) in enumerate(grouper):
                # Set the start and end time for this bin
                note_start = note_idx * note_duration
                note_end = (note_idx + 1) * note_duration
                # Iterate through each individual note in the bin
                for note in subiter:
                    _, __, ___, note_velocity = note
                    # Randomise the pitch
                    note_pitch = np.random.randint(MIDI_OFFSET, (PIANO_KEYS + MIDI_OFFSET) - 1)
                    yield note_start, note_end, note_pitch, note_velocity


def get_multichannel_piano_roll(
        midi_obj: PrettyMIDI, use_concepts: list = None, normalize: bool = False
) -> np.ndarray:
    """For a single MIDI clip, make a four channel piano roll corresponding to Melody, Harmony, Rhythm, and Dynamics"""
    # Create all the extractors for our piano roll
    concept_mapping = {0: MelodyExtractor, 1: HarmonyExtractor, 2: RhythmExtractor, 3: DynamicsExtractor}
    if use_concepts is None:
        use_concepts = [0, 1, 2, 3]
    extractors = []
    for concept_idx in use_concepts:
        # Get the correct extractor object
        ext = concept_mapping[concept_idx]
        # Create the extractor using our provided raw MIDI, get the piano roll, then append to the list
        extractors.append(ext(midi_obj).roll)
    # Normalize velocity dimension if required and stack arrays
    return np.array([normalize_array(concept) if normalize else concept for concept in extractors])


def get_singlechannel_piano_roll(pm_obj: PrettyMIDI, normalize: bool = False) -> np.ndarray:
    """Gets a single channel piano roll, including all data (i.e., without splitting into concepts)"""
    # We can use the base extractor, which will perform checks e.g., removing invalid notes
    extract = BaseExtractor(pm_obj)
    # Squeeze to create a new channel, for parity with our multichannel approach (i.e., batch, channel, pitch, time)
    roll = np.expand_dims(extract.roll, axis=0)
    if normalize:
        roll = normalize_array(roll)
    return roll


def create_outputs_from_extractors(track_name: str, extractors: list) -> None:
    """Creates .WAV and .PNG files for a particular track with extractors and raw_midi"""
    import matplotlib.pyplot as plt

    # Create the folder for this track, in our reports directory
    ref_folder = os.path.join(get_project_root(), 'reports/midi_test', track_name)
    if not os.path.exists(ref_folder):
        os.makedirs(ref_folder)
    # Create the stacked piano roll object
    extractors_names = ["Raw", "Melody", "Harmony", "Rhythm", "Dynamics"]
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True, sharey=True)
    for a, ext, nme in zip(ax.flatten(), extractors, extractors_names):
        if ext is not None:
            a = ext.heatmap(a)
            a.set(title=nme)
            # Create the synthesized .WAV file, while we're iterating
            ext.synthesize(os.path.join(ref_folder, nme.lower()))
    # Plot aesthetics
    ax[0].invert_yaxis()
    fig.supylabel('MIDI Note')
    fig.supxlabel('Frame')
    fig.tight_layout()
    # Save inside our reports directory
    fig.savefig(os.path.join(ref_folder, 'heatmap.png'))


if __name__ == "__main__":
    import os
    from time import time
    from tqdm import tqdm
    from deep_pianist_identification.utils import get_project_root, seed_everything
    from loguru import logger

    seed_everything(10)
    # Get the tracks we'll create clips from
    n_clips = 1000
    use_in_test = np.random.choice(os.listdir(os.path.join(get_project_root(), 'data/clips/pijama')), n_clips)
    res = {"base": [], "melody": [], "harmony": [], "rhythm": [], "dynamics": []}
    all_extractors = [BaseExtractor, MelodyExtractor, HarmonyExtractor, RhythmExtractor, DynamicsExtractor]

    # Iterate through each track
    for track_idx, test_track in tqdm(enumerate(use_in_test), desc=f'Making {n_clips} clips...', total=n_clips):
        # Create the filepath and skip if it does not exist (e.g., `.gitkeep` files)
        test_fpath = os.path.join(get_project_root(), 'data/clips/pijama', test_track, 'clip_000.mid')
        if not os.path.isfile(test_fpath):
            continue
        # Load in the MIDI object
        test_midi_obj = PrettyMIDI(test_fpath)
        extracted_results = []
        # Create each extractor
        for name, extractor in zip(res.keys(), all_extractors):
            start = time()
            try:
                # No data augmentation for now
                created = extractor(test_midi_obj, data_augmentation=False)
            # Log a warning if we failed to create the extractor
            except ExtractorError as err:
                logger.warning(f'Failed to create {name} for {test_track}, skipping! {err}')
                extracted_results.append(None)
            else:
                extracted_results.append(created)
            finally:
                res[name].append(time() - start)
        # For every 10 items, create a graph and WAV files
        if track_idx % 10 == 0:
            create_outputs_from_extractors(test_track, extracted_results)
    # Create the results string and log
    fmt_res = ", ".join(
        f"{k}: total {np.mean(v):.2f}s, mean {np.mean(v):.2f}s, SD {np.std(v):.2f}s" for k, v in res.items()
    )
    logger.info(f"For {n_clips} clips, results: {fmt_res}")
