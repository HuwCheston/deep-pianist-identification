#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract features used in creating white box classification models"""

import os
from ast import literal_eval
from collections import defaultdict
from itertools import groupby
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from aeberscale import find_scale
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

PARALLEL_FEATURE_EXTRACT_JOBS = -1  # Run this number of feature extraction jobs in parallel

DIATONIC_WINDOW_SIZE = 5    # Window size in seconds
DIATONIC_TARGET_NOTES = 64    # If `window` contains fewer than this number of notes, keep adding until it does


def get_valid_chords_chromatic(roll: PrettyMIDI) -> list[list[int]]:
    """Grab valid chords, expressed as chromatic scale steps"""
    chorded = groupby(roll.instruments[0].notes, lambda x: x.start)
    for _, chord in chorded:
        pitches = sorted(i.pitch for i in chord)
        # Express as interval classes rather than pitch classes
        yield [i - min(pitches) for i in pitches][1:]


def get_valid_chords_diatonic(harmony_roll: list[tuple], full_roll: PrettyMIDI, use_mode: bool = False) -> list[list[int]]:
    """Grab valid chords, expressed as diatonic scale steps (using aeberscale to estimate scale)"""
    full_notes = sorted(full_roll.instruments[0].notes, key=lambda x: x.start)

    # Iterate through all the "chords" we've extracted
    for chord in harmony_roll:

        # Grab the start and end time of the window
        window_end = max(chord, key=lambda x: x[1])[1]
        window_start = max(0, window_end - DIATONIC_WINDOW_SIZE)

        # Compute the window using these times
        full_roll_windowed = extract_diatonic_full_score_window(full_notes, window_start, window_end)

        # If not enough notes in the window, skip over
        if len(full_roll_windowed) < DIATONIC_TARGET_NOTES:
            continue

        # We have enough notes in the window to estimate the scale, so do this now
        estimated_scale = find_scale(
            notes=[i.pitch for i in full_roll_windowed],
            durations=[i.duration for i in full_roll_windowed],
        )

        # Mapper to convert from note names to numbers
        mapper = {n: nn for (nn, n) in enumerate(estimated_scale.notes)}

        # List of all MIDI note numbers in the chord
        pitches = sorted([pitch for (start, end, pitch, velocity) in chord])

        # Hold results of iteration here
        diatonic_scale_degrees = []
        valid = True

        # iterate through all the pitches we have
        for p in pitches:

            # calculate how many octaves this is away from the lowest pitch
            octaves_left = abs(p - min(pitches)) // 12

            # convert the pitch into the scale degree
            scale_pitch = estimated_scale[p % 12]   # noqa

            # if None, note isn't in scale == invalid
            if scale_pitch is None:
                valid = False
                break

            # this goes from note names back to number
            scale_degree = mapper[scale_pitch]

            # add the octave back on (so we keep register) and add to the list
            diatonic_scale_degrees.append(scale_degree + (octaves_left * 12))

        if valid and len(diatonic_scale_degrees) > 0:

            # If we want to return the name of the mode used as well
            if use_mode:
                yield diatonic_scale_degrees, get_parent_mode(estimated_scale)

            else:
                yield diatonic_scale_degrees


def extract_diatonic_full_score_window(window, start, end) -> list[Note]:
    """Extract notes from a full score according to window start and end times"""
    # Now, grab the notes that are in the window
    full_score_notes_in_window = [i for i in window if start <= i.start <= end]

    # Truncate or fill the window
    # Too many notes: truncate, keeping earliest notes only
    if len(full_score_notes_in_window) >= DIATONIC_TARGET_NOTES:
        full_score_notes_in_window = full_score_notes_in_window[:DIATONIC_TARGET_NOTES]

    # Not enough notes: grab notes from before the start of the window and keep appending until we have enough
    else:
        outside_window = [i for i in window if i.start < start]
        outside_sort = list(reversed(sorted(outside_window, key=lambda x: x.end)))
        extra_notes = outside_sort[:DIATONIC_TARGET_NOTES - len(full_score_notes_in_window)]
        full_score_notes_in_window = extra_notes + full_score_notes_in_window

    # Sort by onset time
    full_score_notes_windowed_sorted = sorted(full_score_notes_in_window, key=lambda x: x.start)
    return full_score_notes_windowed_sorted


def extract_chords(
        tpath: str,
        nclips: int,
        pianist: str,
        feature_sizes: list[int] = DEFAULT_FEATURE_SIZES,
        diatonic: bool = False,
        use_mode: bool = False
):
    # Create a list of empty default dictionaries, one per each value of n we wish to extract (e.g., 3-gram, 4-gram...)
    track_chords = {ng: defaultdict(int) for ng in feature_sizes}

    all_modes = {}

    # Iterate through all the clips for this track
    for clip_idx in range(nclips):

        # Load up the raw MIDI file
        clip_str = f'clip_{str(clip_idx).zfill(3)}.mid'
        loaded = PrettyMIDI(os.path.join(utils.get_project_root(), tpath, clip_str))

        # Try and extract chords
        try:
            he = HarmonyExtractor(loaded, quantize_resolution=HARMONY_QUANTIZE_RESOLUTION)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {clip_idx}, skipping! {e}')
            continue

        # Call the extract function: diatonic or chromatic
        if not diatonic:
            out_chords = get_valid_chords_chromatic(he.output_midi)
        else:
            out_chords = get_valid_chords_diatonic(he.chorded, he.input_midi, use_mode)

        # Iterate over all the chords we just found
        for out in out_chords:

            if use_mode:
                chord, mode = out
            else:
                chord, mode = out, None

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

            if mode is not None:
                if mode not in all_modes.keys():
                    all_modes[mode] = 1
                else:
                    all_modes[mode] += 1

    # Collapse all the n-gram dictionaries into one
    return {k: v for d in track_chords.values() for k, v in d.items()} | all_modes, pianist


def get_harmony_features(
        train_clips,
        test_clips,
        validation_clips,
        feature_sizes: list[int],
        diatonic: bool = False,
        use_mode: bool = False
) -> tuple:
    logger.info("Extracting chord n-grams from training tracks..")
    temp_x_har, train_y_har = extract_features_from_clips(
        train_clips, extract_chords, feature_sizes, diatonic, use_mode
    )
    logger.info("Extracting chord n-grams from testing tracks...")
    test_x_har, test_y_har = extract_features_from_clips(
        test_clips, extract_chords, feature_sizes, diatonic, use_mode
    )
    logger.info("Extracting chord n-grams from validation tracks...")
    valid_x_har, valid_y_har = extract_features_from_clips(
        validation_clips, extract_chords, feature_sizes, diatonic, use_mode
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
        subsume_ngrams: bool = False
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

    # format features
    train_x_fmt, test_x_fmt, valid_x_fmt, feats = format_features(train_x, test_x, valid_x)

    # subsuming n-grams: create mapping, get idxs, keep only these cols
    if subsume_ngrams:
        ss_mapper = create_subsumed_ngram_mapping([literal_eval(f) for f in feats])
        nonss_idx, feats = get_nonsubsumed_ngrams_idx(ss_mapper, feats)
        train_x_fmt = train_x_fmt[:, nonss_idx]
        test_x_fmt = test_x_fmt[:, nonss_idx]
        valid_x_fmt = valid_x_fmt[:, nonss_idx]

    return train_x_fmt, test_x_fmt, valid_x_fmt, feats


def create_subsumed_ngram_mapping(ngrams) -> dict[list, list]:
    def is_contiguous_sublist(smaller, larger):
        if len(smaller) >= len(larger):
            return False
        m = len(smaller)
        return any(larger[i:i + m] == smaller for i in range(len(larger) - m + 1))

    mapping = {}
    for item in ngrams:
        key = tuple(item)
        mapping[key] = [large for large in ngrams if is_contiguous_sublist(item, large)]

    return mapping


def get_nonsubsumed_ngrams_idx(ss_mapper, feats):
    fouts = []
    keep_cols = []
    for f_idx, f in enumerate(feats):
        f_ = tuple(literal_eval(f))
        if len(ss_mapper[f_]) == 0:
            keep_cols.append(f_idx)
        fouts.append(f)
    return keep_cols, fouts


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


def get_parent_mode(estimated_scale) -> str:
    """Given a aeberscale.Scale object, return the parent scale that this is rotationally equivalent to"""
    rotationally_equivalent = estimated_scale.get_rotationally_equivalent_instances()  # noqa

    # some scales have no rotational equivalence, so parent mode is just the estimated scale
    if len(rotationally_equivalent) == 0:
        parent_scale = estimated_scale.name.lower()

    # other scales are rotationally equivalent. this is a little hacky
    #  We get all the scales that this one is rotationally equivalent to
    #  e.g., for major == lydian, phyrigian, aeolian
    #  then, we sort the list and take the first scale
    #  this does mean that, for major + lydian, the parent is aeolian
    #  in practice this doesn't really matter as we're not plotting this anywhere
    #  the only thing that matters is that the SAME parent mode is returned for e.g. lydian/phrygian/major
    #  no matter what "it" is
    else:
        equivalent_no_root = list(set([i.name.lower() for i in rotationally_equivalent] + [estimated_scale.name.lower()]))
        parent_scale = sorted(equivalent_no_root)[0]

    return parent_scale


def extract_valid_melody_ngrams_diatonic(
        melody: PrettyMIDI,
        full_score: PrettyMIDI,
        ngram_size: int,
        use_mode: bool = False
) -> list[int]:
    """Diatonic equivalent of `extract_valid_melody_ngrams`, estimating scale with aeberscale first"""
    def preproc(notes):
        # Sort all notes by onset time
        sorted_notes = sorted(notes.instruments[0].notes, key=lambda x: x.start)
        # Remove notes with duration of only one frame
        cleaned_notes = [i for i in sorted_notes if i.start != i.end]
        return cleaned_notes

    cleaned_melody_notes = preproc(melody)
    cleaned_full_notes = preproc(full_score)

    # This index is the left "edge" of our window
    for idx in range(len(cleaned_melody_notes) - ngram_size + 1):
        # Grab the notes inside the window
        melody_notes_in_window = cleaned_melody_notes[idx:idx + ngram_size]

        # If the ngram is already invalid (due to e.g. interval span), skip over
        if not validate_ngram(notes_list=melody_notes_in_window):
            continue

        # Now, we need to grab notes in the full score within this window
        #  First, get the boundaries of the window
        window_end = melody_notes_in_window[-1].end
        window_start = max(0, window_end - DIATONIC_WINDOW_SIZE)

        # Now, grab the notes that are in the window
        full_score_notes_windowed_sorted = extract_diatonic_full_score_window(
            cleaned_full_notes, window_start, window_end
        )

        # If not enough notes in the window, skip over
        if len(full_score_notes_windowed_sorted) < DIATONIC_TARGET_NOTES:
            continue

        # We have enough notes in the window to estimate the scale, so do this now
        estimated_scale = find_scale(
            notes=[i.pitch for i in full_score_notes_windowed_sorted],
            durations=[i.duration for i in full_score_notes_windowed_sorted],
        )

        # Now, we can use the aeberscale object to estimate diatonic scale steps
        #  We need the pitch number, so we do % 12 here
        diatonic_scale_steps = estimated_scale.notes_to_diatonic_scale_steps(
            notes=[i.pitch % 12 for i in melody_notes_in_window]
        )   # noqa

        # If any of the notes are OUTSIDE the estimated scale, skip over
        if any([n is None for n in diatonic_scale_steps]):
           continue

        # Convert from note names to numbers
        mapper = {n: nn for (nn, n) in enumerate(estimated_scale.notes)}
        diatonic_scale_degrees = [mapper[i] for i in diatonic_scale_steps]

        # If we want to return the name of the mode used as well
        if use_mode:
            yield diatonic_scale_degrees, get_parent_mode(estimated_scale)

        # Otherwise, if we just want the diatonic scale degrees
        else:
            yield diatonic_scale_degrees


def extract_melody_ngrams(
        tpath: str,
        nclips: int,
        pianist: str,
        ngrams: list = DEFAULT_FEATURE_SIZES,
        diatonic_ngrams: bool = False,
        use_mode: bool = False,
) -> tuple[dict, int]:
    """For a given track path, number of clips, and pianist, extract all melody `ngrams` and return a dictionary"""
    # Create a list of empty default dictionaries, one per each value of n we wish to extract (e.g., 3-gram, 4-gram...)
    ngram_res = [defaultdict(int) for _ in range(len(ngrams))]

    # Iterate over all the clips
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

            # need to define this variable, will update it in case of using diatonic + mode
            valid_modes = []

            # Extract valid n-grams of this size, using chromatic scale steps
            if not diatonic_ngrams:
                valid_ngrams_of_size = extract_valid_melody_ngrams(roll.skylined, ngram_size + 1)

            # Extract valid n-grams of this size, using diatonic scale steps
            else:

                # need to account for having different return shapes depending on if we're returning the mode
                #  of each n-gram as well
                out = list(extract_valid_melody_ngrams_diatonic(roll.skylined, roll.input_midi, ngram_size + 1, use_mode=use_mode))

                if len(out) == 0:
                    # will happen if we've found no valid n-grams/modes of this size in the clip
                    continue
                elif use_mode:
                    # little tuple unpacking hack
                    valid_ngrams_of_size, valid_modes = zip(*out)
                else:
                    valid_ngrams_of_size = out

            # Iterate through all the valid n-grams we've grabbed
            for valid_ngram in valid_ngrams_of_size:
                # Update the counter for this n-gram by one
                ngram_dict[str(valid_ngram)] += 1

            # If we have any valid modes
            for valid_mode in valid_modes:
                ngram_dict[str(valid_mode)] += 1

    # Collapse all the ngram dictionaries to a single dictionary and return alongside the ground truth label
    return {k: v for d in ngram_res for k, v in d.items()}, pianist


def get_melody_features(
        train_clips,
        test_clips,
        validation_clips,
        feature_sizes: list[int],
        diatonic: bool = False,
        use_mode: bool = False,
) -> tuple:
    # MELODY EXTRACTION
    logger.info("Extracting melody n-grams from training tracks..")
    temp_x_mel, train_y_mel = extract_features_from_clips(
        train_clips, extract_melody_ngrams, feature_sizes, diatonic, use_mode
    )
    logger.info("Extracting melody n-grams from testing tracks...")
    test_x_mel, test_y_mel = extract_features_from_clips(
        test_clips, extract_melody_ngrams, feature_sizes, diatonic, use_mode
    )
    logger.info("Extracting melody n-grams from validation tracks...")
    valid_x_mel, valid_y_mel = extract_features_from_clips(
        validation_clips, extract_melody_ngrams, feature_sizes, diatonic, use_mode
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
