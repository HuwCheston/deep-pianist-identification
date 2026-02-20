#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract features used in creating white box classification models"""

import os
from collections import defaultdict
from itertools import groupby
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from pretty_midi import PrettyMIDI, Note

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import HarmonyExtractor, ExtractorError, MelodyExtractor

HARMONY_QUANTIZE_RESOLUTION = 0.1
HARMONY_MAX_LEAPS_IN_CHORD = 2  # remove any features with more than two leaps of HARMONY_MAX_LEAP
HARMONY_MAX_LEAP = 15

MELODY_QUANTIZE_RESOLUTION = 0.1
MELODY_MAX_SPAN = 12  # remove any n-grams that span more intervals than this
MELODY_MAX_LEAP = 12  # remove any n-grams with individual leaps larger than this
MELODY_MAX_REST = 2.  # remove any n-grams with longer than this in successive onset-offset times

FEATURE_MIN_TRACKS = 10  # features in fewer than this number of tracks are removed
FEATURE_MAX_TRACKS = 1000  # features in more than this number of tracks

DEFAULT_FEATURE_SIZES = (2, 3, 4, 5, 6)  # We'll consider features with this size by default

PARALLEL_FEATURE_EXTRACT_JOBS = 1  # Run this number of feature extraction jobs in parallel


def _extract_fn_harmony(roll: PrettyMIDI):
    chorded = groupby(roll.instruments[0].notes, lambda x: x.start)
    for _, chord in chorded:
        pitches = sorted(i.pitch for i in chord)
        # Express as interval classes rather than pitch classes
        yield [i - min(pitches) for i in pitches][1:]


def extract_chords(
        tpath: str,
        nclips: int,
        pianist: str,
        feature_sizes: list[int] = DEFAULT_FEATURE_SIZES,
):
    # Create a list of empty default dictionaries, one per each value of n we wish to extract (e.g., 3-gram, 4-gram...)
    track_chords = {ng: defaultdict(int) for ng in feature_sizes}
    for clip_idx in range(nclips):
        clip_str = f'clip_{str(clip_idx).zfill(3)}.mid'
        loaded = PrettyMIDI(os.path.join(utils.get_project_root(), tpath, clip_str))
        try:
            he = HarmonyExtractor(loaded, quantize_resolution=HARMONY_QUANTIZE_RESOLUTION)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {clip_idx}, skipping! {e}')
            continue
        for chord in _extract_fn_harmony(he.output_midi):
            # Calculate the number of semitones between successive notes of the chord
            leaps = [i1 - i2 for i2, i1 in zip(chord, chord[1:])]
            # Calculate the number of semitone leaps that are above our threshold
            above_thresh = [i for i in leaps if i >= HARMONY_MAX_LEAP]
            # If we have more than N (default N = 2) leaps above the threshold, skip the chord
            if len(above_thresh) >= HARMONY_MAX_LEAPS_IN_CHORD:
                continue
            # Increment the counter by one
            if len(chord) in feature_sizes:
                track_chords[len(chord)][str(chord)] += 1
    # Collapse all the n-gram dictionaries into one
    return {k: v for d in track_chords.values() for k, v in d.items()}, pianist


def get_harmony_features(
        train_clips,
        test_clips,
        validation_clips,
        feature_sizes: list[int],
) -> tuple:
    logger.info("Extracting chord n-grams from training tracks..")
    temp_x_har, train_y_har = extract_features_from_clips(
        train_clips, extract_chords, feature_sizes
    )
    logger.info("Extracting chord n-grams from testing tracks...")
    test_x_har, test_y_har = extract_features_from_clips(
        test_clips, extract_chords, feature_sizes
    )
    logger.info("Extracting chord n-grams from validation tracks...")
    valid_x_har, valid_y_har = extract_features_from_clips(
        validation_clips, extract_chords, feature_sizes
    )
    # These are lists of dictionaries with one dictionary == one track, so we can just combine them
    all_ngs = temp_x_har + test_x_har + valid_x_har
    # Get all the unique chords and log the total number
    unique_ngs = list(set(k for d in all_ngs for k in d.keys()))
    logger.info(f'... found {len(unique_ngs)} chord n-grams!')
    return temp_x_har, test_x_har, valid_x_har, train_y_har, test_y_har, valid_y_har


def drop_invalid_features(
        train_x: np.ndarray,
        test_x: np.ndarray,
        valid_x: np.ndarray,
        min_count: int = FEATURE_MIN_TRACKS,
        max_count: int = FEATURE_MAX_TRACKS,
) -> tuple:
    all_features = train_x + test_x + valid_x
    # Get those features that appear in at least N tracks in the entire dataset
    logger.info(f"Extracting features which appear in min {min_count}, max {max_count} tracks...")
    valid_features = get_valid_features_by_track_usage(all_features, min_count, max_count)
    logger.info(f"... found {len(valid_features)} valid features!")
    # Subset both datasets to ensure we only keep valid features
    train_x = drop_invalid_features_by_track_usage(train_x, valid_features)
    test_x = drop_invalid_features_by_track_usage(test_x, valid_features)
    valid_x = drop_invalid_features_by_track_usage(valid_x, valid_features)
    return format_features(train_x, test_x, valid_x)


def get_span(notes_list: list[Note]) -> int:
    """Gets the pitch span of a list of pretty_midi.Note objects"""
    highest_note = max(notes_list, key=lambda x: x.pitch)
    lowest_note = min(notes_list, key=lambda x: x.pitch)
    return highest_note.pitch - lowest_note.pitch


def validate_ngram(notes_list: list[Note]) -> bool:
    """Runs checks on a list of pretty_midi.Note objects and returns True when passing, False when failing"""
    offset_onsets = [n2.start - n1.end for n1, n2 in zip(notes_list[0:], notes_list[1:])]
    intervals = [n2.pitch - n1.pitch for n1, n2 in zip(notes_list[0:], notes_list[1:])]
    return all((
        get_span(notes_list) <= MELODY_MAX_SPAN,
        all(abs(interval) <= MELODY_MAX_LEAP for interval in intervals),  # need to use absolute interval here
        all(overlap <= MELODY_MAX_REST for overlap in offset_onsets),
        all(interval != 0 for interval in intervals)
    ))


def extract_valid_melody_ngrams(melody: PrettyMIDI, ngram_size: int) -> list[int]:
    """From a pretty_midi object, return intervals for n-grams of size `ngram_size`"""
    # Sort all notes by onset time
    sorted_notes = sorted(melody.instruments[0].notes, key=lambda x: x.start)
    # Remove notes with duration of only one frame
    cleaned_notes = [i for i in sorted_notes if i.start != i.end]
    # This index is the left "edge" of our window
    for idx in range(len(cleaned_notes) - ngram_size + 1):
        # Grab the notes inside the window
        notes_in_window = cleaned_notes[idx:idx + ngram_size]
        # If the notes pass our check, continue
        if validate_ngram(notes_in_window) and len(notes_in_window) == ngram_size:
            yield [n2.pitch - n1.pitch for n1, n2 in zip(notes_in_window[0:], notes_in_window[1:])]


def extract_melody_ngrams(
        tpath: str,
        nclips: int,
        pianist: str,
        ngrams: list = DEFAULT_FEATURE_SIZES,
) -> tuple[dict, int]:
    """For a given track path, number of clips, and pianist, extract all melody `ngrams` and return a dictionary"""
    # Create a list of empty default dictionaries, one per each value of n we wish to extract (e.g., 3-gram, 4-gram...)
    ngram_res = [defaultdict(int) for _ in range(len(ngrams))]
    for idx in range(nclips):
        # Load the clip in as a PrettyMIDI object
        clip_path = os.path.join(utils.get_project_root(), tpath, f'clip_{str(idx).zfill(3)}.mid')
        loaded = PrettyMIDI(clip_path)
        # Try and create the piano roll, but skip if we end up with zero notes
        try:
            roll = MelodyExtractor(loaded, clip_start=0., quantize_resolution=MELODY_QUANTIZE_RESOLUTION)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {idx}, skipping! {e}')
            continue
        # Iterate through all the n-gram sizes we want to grab
        for ngram_dict, ngram_size in zip(ngram_res, ngrams):
            # Extract valid n-grams of this size
            valid_ngrams_of_size = extract_valid_melody_ngrams(roll.skylined, ngram_size + 1)
            # Iterate through all the valid n-grams we've grabbed
            for valid_ngram in valid_ngrams_of_size:
                # Update the counter for this n-gram by one
                ngram_dict[str(valid_ngram)] += 1
    # Collapse all the ngram dictionaries to a single dictionary and return alongside the ground truth label
    return {k: v for d in ngram_res for k, v in d.items()}, pianist


def get_melody_features(
        train_clips,
        test_clips,
        validation_clips,
        feature_sizes: list[int],
) -> tuple:
    # MELODY EXTRACTION
    logger.info("Extracting melody n-grams from training tracks..")
    temp_x_mel, train_y_mel = extract_features_from_clips(
        train_clips, extract_melody_ngrams, feature_sizes
    )
    logger.info("Extracting melody n-grams from testing tracks...")
    test_x_mel, test_y_mel = extract_features_from_clips(
        test_clips, extract_melody_ngrams, feature_sizes
    )
    logger.info("Extracting melody n-grams from validation tracks...")
    valid_x_mel, valid_y_mel = extract_features_from_clips(
        validation_clips, extract_melody_ngrams, feature_sizes
    )
    # These are lists of dictionaries with one dictionary == one track, so we can just combine them
    all_ngs = temp_x_mel + test_x_mel + valid_x_mel
    # Get all the unique n-grams
    unique_ngs = list(set(k for d in all_ngs for k in d.keys()))
    logger.info(f'... found {len(unique_ngs)} melody n-grams!')
    return temp_x_mel, test_x_mel, valid_x_mel, train_y_mel, test_y_mel, valid_y_mel


def extract_features_from_clips(split_clips: Iterable, extract_func: Callable, *args) -> tuple[list[dict], list[int]]:
    """For clips from a given split, applies an `extract_func` in parallel"""
    # Iterate over each track and apply our extraction function
    with Parallel(n_jobs=PARALLEL_FEATURE_EXTRACT_JOBS, verbose=5) as par:
        res = par(delayed(extract_func)(p, n, pi, *args) for p, n, pi in list(split_clips))
    # Returns a tuple of ngrams (dict) and targets (int) for every *clip* in the dataset
    return zip(*res)


def get_valid_features_by_track_usage(
        list_of_ngram_dicts: list[dict],
        min_count: int = FEATURE_MIN_TRACKS,
        max_count: int = FEATURE_MAX_TRACKS
) -> list[str]:
    """From a list of n-gram dictionaries (one dictionary per track), get n-grams that appear in at least N tracks"""
    ngram_counts = defaultdict(int)
    # Iterate through each dictionary (= one track)
    for d in list_of_ngram_dicts:
        # Iterate through each key (= one n-gram in the track)
        for key in d:
            # Increase the global count for this n-gram
            ngram_counts[key] += 1
    # Return a list of n-grams that appear in at least N tracks
    return [key for key, count in ngram_counts.items() if min_count <= count <= max_count]


def drop_invalid_features_by_track_usage(
        list_of_ngram_dicts: list[dict],
        valid_ngrams: list[str]
) -> list[dict]:
    """From a list of n-gram dictionaries (one dictionary per track), remove n-grams that do not appear in N tracks"""
    # Define the removal function
    rm = lambda feat: {k: v for k, v in feat.items() if k in valid_ngrams}
    # In parallel, remove non-valid n-grams from all dictionaries
    with Parallel(n_jobs=PARALLEL_FEATURE_EXTRACT_JOBS, verbose=5) as par:
        return par(delayed(rm)(f) for f in list_of_ngram_dicts)


def format_features(*features: list[dict]) -> tuple:
    """For an arbitrary number of ngram dictionaries, format these as NumPy arrays with the same number of columns"""
    fmt = []
    for split_count, feature in enumerate(features):
        # Convert features to a dataframe
        df = pd.DataFrame(feature)
        # Set ID column
        df['split'] = str(split_count)
        fmt.append(df)
    # Combine all dataframes row-wise, which ensures the correct number of columns in all, and fill NaNs with 0
    conc = pd.concat(fmt, axis=0).fillna(0)
    # Return a tuple of numpy arrays, one for each split initially provided, PLUS the names of the features themselves
    return (
        *(conc[conc['split'] == str(sp)].drop(columns='split').to_numpy() for sp in range(len(features))),
        conc.drop(columns='split').columns.tolist()
    )
