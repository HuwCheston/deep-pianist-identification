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
from deep_pianist_identification.extractors import HarmonyExtractor, ExtractorError

__all__ = ["extract_chords", "get_harmony_features", "MAX_LEAPS_IN_CHORD"]

MAX_LEAPS_IN_CHORD = 2


def _extract_fn(roll: PrettyMIDI, transpose: bool = True):
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
        for chord in _extract_fn(he.output_midi):
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


def rf_harmony(
        dataset: str,
        n_iter: int,
        ngrams: list[int],
        min_count: int,
        max_count: int,
        remove_leaps: bool,
        classifier_type: str,
        scale: bool
):
    """Create and optimize random forest harmony classifier using provided command line arguemnts"""
    logger.info("Creating white box classifier using harmony data!")
    logger.info(f'... using model type {classifier_type}')
    # Get clips from all splits of the dataset
    train_clips, test_clips, validation_clips = rf_utils.get_all_clips(dataset)
    # Get n-grams from both datasets
    train_x_arr, test_x_arr, valid_x_arr, feature_names, train_y, test_y, valid_y = get_harmony_features(
        train_clips, test_clips, validation_clips, ngrams, min_count, max_count, remove_leaps
    )
    # Load the optimized parameter settings (or recreate them, if they don't exist)
    csvpath = os.path.join(
        utils.get_project_root(),
        'references/rf_baselines',
        f'{dataset}_{classifier_type}_harmony.csv'
    )
    # Scale the data if required (never for multinomial naive Bayes as this expects count data)
    if scale and classifier_type != "nb":
        train_x_arr, test_x_arr, valid_x_arr = rf_utils.scale_features(train_x_arr, test_x_arr, valid_x_arr)
    # Optimize the classifier
    _, __, ___ = rf_utils.fit_classifier(
        train_x_arr, test_x_arr, valid_x_arr, train_y, test_y, valid_y, csvpath, n_iter, classifier_type
    )
    logger.info('Done!')


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Create white box classifier for harmony')
    args = rf_utils.parse_arguments(parser)
    # Create the forest
    rf_harmony(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        ngrams=args["ngrams"],
        min_count=args["min_count"],
        max_count=args["max_count"],
        remove_leaps=args["remove_leaps"],
        classifier_type=args["classifier_type"],
        scale=args["scale"]
    )
