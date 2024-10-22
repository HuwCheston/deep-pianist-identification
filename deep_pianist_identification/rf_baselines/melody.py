#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to skylined MIDI melody files"""

import os

from loguru import logger
from pretty_midi import PrettyMIDI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.extractors import MelodyExtractor, ExtractorError

__all__ = ["NGRAMS", "VALID_NGRAMS", "extract_melody_ngrams"]

NGRAMS = [3, 4]
VALID_NGRAMS = 10


def _extract_fn(roll: PrettyMIDI, ngrams: list[int]):
    """From a given piano roll, extract required melody n-grams"""
    # Sort the notes by onset start time and calculate the intervals
    melody = sorted(roll.instruments[0].notes, key=lambda x: x.start)
    intervals = [i2.pitch - i1.pitch for i1, i2 in zip(melody[0:], melody[1:])]
    # Iterate through every ngram and corresponding results dictionary
    for n in ngrams:
        # Extract the n-grams from the clip
        yield [intervals[i: i + n] for i in range(len(intervals) - n + 1)]


def extract_melody_ngrams(tpath: str, nclips: int, pianist: str, ngrams: list = None) -> tuple[dict, int]:
    """For a given track path, number of clips, and pianist, extract all melody `ngrams` and return a dictionary"""
    # Use the default ngram value, if not provided
    if ngrams is None:
        ngrams = NGRAMS
    # Create a list of empty dictionaries, one per each value of n we wish to extract (e.g., 3-grams, 4-grams...)
    ngram_res = [dict() for _ in range(len(ngrams))]
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
            # Append each n-gram to the dictionary if not present already and increase the count by one
            for ng in grams:
                ng = str(ng)
                if ng not in di.keys():
                    di[ng] = 1
                else:
                    di[ng] += 1
    # Collapse all the ngram dictionaries to a single dictionary and return alongside the ground truth label
    return {k: v for d in ngram_res for k, v in d.items()}, pianist


def rf_melody(dataset: str, n_iter: int, valid_ngrams_count: int, ngrams: list[int]) -> None:
    """Create and optimize random forest melody classifier using provided command line arguemnts"""
    logger.info("Creating baseline random forest classifier using melody data!")
    # Get clips from all splits of the dataset
    logger.info(f"Getting tracks from all splits for dataset {dataset}...")
    train_clips = list(rf_utils.get_split_clips("train", dataset))
    test_clips = list(rf_utils.get_split_clips("test", dataset))
    validation_clips = list(rf_utils.get_split_clips("validation", dataset))
    logger.info(f"... found {len(train_clips)} train tracks, "
                f"{len(test_clips)} test tracks, "
                f"{len(validation_clips)} validation tracks!")
    # Get n-grams from both datasets
    logger.info("Extracting n-grams from training tracks..")
    temp_x, train_y = rf_utils.extract_ngrams_from_clips(train_clips, extract_melody_ngrams, ngrams)
    logger.info("Extracting n-grams from testing tracks...")
    test_x, test_y = rf_utils.extract_ngrams_from_clips(test_clips, extract_melody_ngrams, ngrams)
    logger.info("Extracting n-grams from validation tracks...")
    valid_x, valid_y = rf_utils.extract_ngrams_from_clips(validation_clips, extract_melody_ngrams, ngrams)
    # Get those n-grams that appear in at least N tracks in the training dataset
    logger.info(f"Extracting n-grams which appear in at least {valid_ngrams_count} training tracks...")
    valid_ngs = rf_utils.get_valid_ngrams(temp_x, min_count=valid_ngrams_count)
    logger.info(f"... found {len(valid_ngs)} n-grams!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting features...")
    train_x = rf_utils.drop_invalid_ngrams(temp_x, valid_ngs)
    test_x = rf_utils.drop_invalid_ngrams(test_x, valid_ngs)
    valid_x = rf_utils.drop_invalid_ngrams(valid_x, valid_ngs)
    # Convert both training and testing features into numpy arrays with the same dimensionality
    train_x_arr, test_x_arr, valid_x_arr = rf_utils.format_features(train_x, test_x, valid_x)
    # Load the optimized parameter settings (or recreate them, if they don't exist)
    logger.info(f"Beginning parameter optimization with {n_iter} iterations...")
    csvpath = os.path.join(utils.get_project_root(), 'references/rf_baselines', f'{dataset}_melody.csv')
    logger.info(f"... optimization results will be saved in {csvpath}")
    optimized_params = rf_utils.optimize_classifier(train_x_arr, train_y, test_x_arr, test_y, csvpath, n_iter=n_iter)
    # Create the optimized random forest model
    logger.info("Optimization finished, fitting optimized model to test and validation set...")
    clf_opt = RandomForestClassifier(**optimized_params, random_state=utils.SEED)
    clf_opt.fit(train_x_arr, train_y)
    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_x_arr)
    test_acc = accuracy_score(test_y, test_y_pred)
    logger.info(f"... test accuracy for melody: {test_acc:.3f}")
    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x_arr)
    valid_acc = accuracy_score(valid_y, valid_y_pred)
    logger.info(f"... validation accuracy for melody: {valid_acc:.3f}")
    logger.info('Done!')


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Create baseline random forest classifier for melody')
    parser.add_argument(
        "-d", "--dataset",
        default="20class_80min",
        type=str,
        help="Name of dataset inside `references/data_split`"
    )
    parser.add_argument(
        "-i", "--n-iter",
        default=rf_utils.N_ITER,
        type=int,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "-v", "--valid-ngrams-count",
        default=VALID_NGRAMS,
        type=int,
        help="Valid n-grams appear in this many tracks"
    )
    parser.add_argument(
        '-l', '--ngrams',
        nargs='+',
        type=int,
        help="Extract these n-grams",
        default=NGRAMS
    )
    # Parse all arguments and create the forest
    args = vars(parser.parse_args())
    rf_melody(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        ngrams=args["ngrams"],
        valid_ngrams_count=args["valid_ngrams_count"]
    )
