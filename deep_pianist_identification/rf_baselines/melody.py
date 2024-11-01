#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to skylined MIDI melody files"""

import os
from collections import defaultdict

from loguru import logger
from pretty_midi import PrettyMIDI

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.extractors import MelodyExtractor, ExtractorError

__all__ = ["extract_melody_ngrams", "get_melody_features"]


def _extract_fn(roll: PrettyMIDI, ngrams: list[int]):
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
        ngs = _extract_fn(roll.output_midi, ngrams)
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


def rf_melody(
        dataset: str,
        n_iter: int,
        ngrams: list[int],
        min_count: int,
        max_count: int,
        remove_leaps: bool,
        classifier_type: str,
        scale: bool
) -> None:
    """Create and optimize random forest melody classifier using provided command line arguemnts"""
    logger.info("Creating white box classifier using melody data!")
    logger.info(f'... using model type {classifier_type}')
    # Get clips from all splits of the dataset
    train_clips, test_clips, validation_clips = rf_utils.get_all_clips(dataset)
    # Get n-grams from both datasets
    train_x_arr, test_x_arr, valid_x_arr, feature_names, train_y, test_y, valid_y = get_melody_features(
        train_clips, test_clips, validation_clips, ngrams, min_count, max_count, remove_leaps
    )
    # Load the optimized parameter settings (or recreate them, if they don't exist)
    csvpath = os.path.join(
        utils.get_project_root(),
        'references/rf_baselines',
        f'{dataset}_{classifier_type}_melody.csv'
    )
    _, __ = rf_utils.fit_classifier(
        train_x_arr, test_x_arr, valid_x_arr, train_y, test_y, valid_y, csvpath, n_iter, classifier_type, scale
    )
    logger.info('Done!')


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Create white box classifier for melody')
    args = rf_utils.parse_arguments(parser)
    # Create the forest
    rf_melody(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        ngrams=args["ngrams"],
        min_count=args["min_count"],
        max_count=args["max_count"],
        remove_leaps=args["remove_leaps"],
        classifier_type=args["classifier_type"],
        scale=args["scale"]
    )
