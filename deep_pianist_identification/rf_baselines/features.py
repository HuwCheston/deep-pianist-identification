#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to harmony"""

import os
from collections import defaultdict
from itertools import groupby

from loguru import logger
from pretty_midi import PrettyMIDI

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.extractors import HarmonyExtractor, ExtractorError, MelodyExtractor

__all__ = [
    "extract_chords", "get_harmony_features", "MAX_LEAPS_IN_CHORD", "extract_melody_ngrams", "get_melody_features"
]

MAX_LEAPS_IN_CHORD = 2


def _extract_fn_harmony(roll: PrettyMIDI, transpose: bool = True):
    chorded = groupby(roll.instruments[0].notes, lambda x: x.start)
    for _, chord in chorded:
        pitches = sorted(i.pitch for i in chord)
        if transpose:
            # Express as interval classes rather than pitch classes
            yield [i - min(pitches) for i in pitches][1:]
        else:
            yield sorted(list(pitches))


def extract_chords(
        tpath: str,
        nclips: int,
        pianist: str,
        ngrams: list[int] = None,
        remove_leaps: bool = False
):
    # Use the default ngram value, if not provided
    if ngrams is None:
        ngrams = rf_utils.NGRAMS
    # Create a list of empty default dictionaries, one per each value of n we wish to extract (e.g., 3-gram, 4-gram...)
    track_chords = {ng: defaultdict(int) for ng in ngrams}
    for clip_idx in range(nclips):
        clip_str = f'clip_{str(clip_idx).zfill(3)}.mid'
        loaded = PrettyMIDI(os.path.join(utils.get_project_root(), tpath, clip_str))
        try:
            he = HarmonyExtractor(loaded, quantize_resolution=0.3)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {clip_idx}, skipping! {e}')
            continue
        for chord in _extract_fn_harmony(he.output_midi):
            if remove_leaps:
                # Calculate the number of semitones between successive notes of the chord
                leaps = [i1 - i2 for i2, i1 in zip(chord, chord[1:])]
                # Calculate the number of semitone leaps that are above our threshold
                above_thresh = [i for i in leaps if i >= rf_utils.MAX_LEAP]
                # If we have more than N (default N = 2) leaps above the threshold, skip the chord
                if len(above_thresh) >= MAX_LEAPS_IN_CHORD:
                    continue
            # Increment the counter by one
            if len(chord) in ngrams:
                track_chords[len(chord)][str(chord)] += 1
    # Collapse all the n-gram dictionaries into one
    return {k: v for d in track_chords.values() for k, v in d.items()}, pianist


def get_harmony_features(
        train_clips,
        test_clips,
        validation_clips,
        ngrams: list[int],
        min_count: int,
        max_count: int,
        remove_leaps: bool
) -> tuple:
    logger.info("Extracting chord n-grams from training tracks..")
    temp_x_har, train_y_har = rf_utils.extract_ngrams_from_clips(
        train_clips, extract_chords, ngrams, remove_leaps
    )
    logger.info("Extracting chord n-grams from testing tracks...")
    test_x_har, test_y_har = rf_utils.extract_ngrams_from_clips(
        test_clips, extract_chords, ngrams, remove_leaps
    )
    logger.info("Extracting chord n-grams from validation tracks...")
    valid_x_har, valid_y_har = rf_utils.extract_ngrams_from_clips(
        validation_clips, extract_chords, ngrams, remove_leaps
    )
    # These are lists of dictionaries with one dictionary == one track, so we can just combine them
    all_ngs = temp_x_har + test_x_har + valid_x_har
    # Get all the unique n-grams
    unique_ngs = list(set(k for d in all_ngs for k in d.keys()))
    logger.info(f'... found {len(unique_ngs)} chord n-grams!')
    # Get those n-grams that appear in at least N tracks in the entire dataset
    logger.info(f"Extracting chord n-grams which appear in min {min_count}, max {max_count} tracks...")
    valid_chords = rf_utils.get_valid_ngrams(all_ngs, min_count, max_count)
    logger.info(f"... found {len(valid_chords)} chord n-grams!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting harmony features...")
    train_x_har = rf_utils.drop_invalid_ngrams(temp_x_har, valid_chords)
    test_x_har = rf_utils.drop_invalid_ngrams(test_x_har, valid_chords)
    valid_x_har = rf_utils.drop_invalid_ngrams(valid_x_har, valid_chords)
    return *rf_utils.format_features(train_x_har, test_x_har, valid_x_har), train_y_har, test_y_har, valid_y_har


def _extract_fn_melody(roll: PrettyMIDI, ngrams: list[int]):
    """From a given piano roll, extract required melody n-grams"""
    # Sort the notes by onset start time and calculate the intervals
    melody = sorted(roll.instruments[0].notes, key=lambda x: x.start)
    intervals = [i2.pitch - i1.pitch for i1, i2 in zip(melody[0:], melody[1:])]
    # Iterate through every ngram and corresponding results dictionary
    for n in ngrams:
        # Extract the n-grams from the clip
        yield [intervals[i: i + n] for i in range(len(intervals) - n + 1)]


def extract_melody_ngrams(
        tpath: str,
        nclips: int,
        pianist: str,
        ngrams: list = None,
        remove_leaps: bool = False
) -> tuple[dict, int]:
    """For a given track path, number of clips, and pianist, extract all melody `ngrams` and return a dictionary"""
    # Use the default ngram value, if not provided
    if ngrams is None:
        ngrams = rf_utils.NGRAMS
    # Create a list of empty default dictionaries, one per each value of n we wish to extract (e.g., 3-gram, 4-gram...)
    ngram_res = [defaultdict(int) for _ in range(len(ngrams))]
    for idx in range(nclips):
        # Load the clip in as a PrettyMIDI object
        clip_path = os.path.join(utils.get_project_root(), tpath, f'clip_{str(idx).zfill(3)}.mid')
        loaded = PrettyMIDI(clip_path)
        # Try and create the piano roll, but skip if we end up with zero notes
        try:
            roll = MelodyExtractor(loaded, clip_start=0)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {idx}, skipping! {e}')
            continue
        # Extract the n-grams from the output midi
        ngs = _extract_fn_melody(roll.output_midi, ngrams)
        # Iterate through every ngram and corresponding results dictionary
        for grams, di in zip(ngs, ngram_res):
            # Iterate through all n-grams
            for ng in grams:
                # If we want to remove leaps, and ONE of the intervals is above our maximum threshold, then drop
                if any(abs(val) >= rf_utils.MAX_LEAP for val in ng) and remove_leaps:
                    continue
                # Otherwise, increment the counter by one for this n-gram
                di[str(ng)] += 1
    # Collapse all the ngram dictionaries to a single dictionary and return alongside the ground truth label
    return {k: v for d in ngram_res for k, v in d.items()}, pianist


def get_melody_features(
        train_clips,
        test_clips,
        validation_clips,
        ngrams: list[int],
        min_count: int,
        max_count: int,
        remove_leaps: bool
) -> tuple:
    # MELODY EXTRACTION
    logger.info("Extracting melody n-grams from training tracks..")
    temp_x_mel, train_y_mel = rf_utils.extract_ngrams_from_clips(
        train_clips, extract_melody_ngrams, ngrams, remove_leaps
    )
    logger.info("Extracting melody n-grams from testing tracks...")
    test_x_mel, test_y_mel = rf_utils.extract_ngrams_from_clips(
        test_clips, extract_melody_ngrams, ngrams, remove_leaps
    )
    logger.info("Extracting melody n-grams from validation tracks...")
    valid_x_mel, valid_y_mel = rf_utils.extract_ngrams_from_clips(
        validation_clips, extract_melody_ngrams, ngrams, remove_leaps
    )
    # These are lists of dictionaries with one dictionary == one track, so we can just combine them
    all_ngs = temp_x_mel + test_x_mel + valid_x_mel
    # Get all the unique n-grams
    unique_ngs = list(set(k for d in all_ngs for k in d.keys()))
    logger.info(f'... found {len(unique_ngs)} melody n-grams!')
    # Get those n-grams that appear in at least N tracks in the entire dataset
    logger.info(f"Extracting melody n-grams which appear in min {min_count}, max {max_count} tracks...")
    valid_ngs = rf_utils.get_valid_ngrams(all_ngs, min_count, max_count)
    logger.info(f"... found {len(valid_ngs)} melody n-grams!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting features...")
    train_x_mel = rf_utils.drop_invalid_ngrams(temp_x_mel, valid_ngs)
    test_x_mel = rf_utils.drop_invalid_ngrams(test_x_mel, valid_ngs)
    valid_x_mel = rf_utils.drop_invalid_ngrams(valid_x_mel, valid_ngs)
    return *rf_utils.format_features(train_x_mel, test_x_mel, valid_x_mel), train_y_mel, test_y_mel, valid_y_mel
