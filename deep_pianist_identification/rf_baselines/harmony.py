#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to harmony"""

import os
from collections import defaultdict
from itertools import groupby

from loguru import logger
from pretty_midi import PrettyMIDI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.extractors import HarmonyExtractor, ExtractorError

VALID_CHORDS = 10


def extract_fn(roll: PrettyMIDI, transpose: bool = True):
    chorded = groupby(roll.instruments[0].notes, lambda x: x.start)
    for _, chord in chorded:
        pitches = sorted(i.pitch for i in chord)
        if transpose:
            yield [i - min(pitches) for i in pitches]
        else:
            yield list(pitches)


def extract_chords(tpath, nclips, pianist, remove_highest_pitch: bool):
    track_chords = defaultdict(int)
    for clip_idx in range(nclips):
        clip_str = f'clip_{str(clip_idx).zfill(3)}.mid'
        loaded = PrettyMIDI(os.path.join(utils.get_project_root(), tpath, clip_str))
        try:
            he = HarmonyExtractor(loaded, remove_highest_pitch=remove_highest_pitch, quantize_resolution=0.3)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {clip_idx}, skipping! {e}')
            continue
        for chord in extract_fn(he.output_midi):
            track_chords[str(chord)] += 1
    return track_chords, pianist


def rf_harmony(dataset: str, n_iter: int, valid_chords_count: int, remove_highest_pitch: bool):
    """Create and optimize random forest harmony classifier using provided command line arguemnts"""
    logger.info("Creating baseline random forest classifier using harmony data!")
    # Get clips from all splits of the dataset
    logger.info(f"Getting tracks from all splits for dataset {dataset}...")
    train_clips = list(rf_utils.get_split_clips("train", dataset))
    test_clips = list(rf_utils.get_split_clips("test", dataset))
    validation_clips = list(rf_utils.get_split_clips("validation", dataset))
    logger.info(f"... found {len(train_clips)} train tracks, "
                f"{len(test_clips)} test tracks, "
                f"{len(validation_clips)} validation tracks!")
    # Get n-grams from both datasets
    logger.info("Extracting chords from training tracks..")
    temp_x, train_y = rf_utils.extract_ngrams_from_clips(train_clips, extract_chords, remove_highest_pitch)
    logger.info("Extracting chords from testing tracks...")
    test_x, test_y = rf_utils.extract_ngrams_from_clips(test_clips, extract_chords, remove_highest_pitch)
    logger.info("Extracting chords from validation tracks...")
    valid_x, valid_y = rf_utils.extract_ngrams_from_clips(validation_clips, extract_chords, remove_highest_pitch)
    # Get those n-grams that appear in at least N tracks in the training dataset
    logger.info(f"Extracting chords which appear in at least {valid_chords_count} training tracks...")
    valid_chords = rf_utils.get_valid_ngrams(temp_x, valid_count=valid_chords_count)
    logger.info(f"... found {len(valid_chords)} chords!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting harmony features...")
    train_x = rf_utils.drop_invalid_ngrams(temp_x, valid_chords)
    test_x = rf_utils.drop_invalid_ngrams(test_x, valid_chords)
    valid_x = rf_utils.drop_invalid_ngrams(valid_x, valid_chords)
    train_x_arr, test_x_arr, valid_x_arr = rf_utils.format_features(train_x, test_x, valid_x)
    # Load the optimized parameter settings (or recreate them, if they don't exist)
    logger.info(f"Beginning parameter optimization for harmony with {n_iter} iterations...")
    csvpath = os.path.join(utils.get_project_root(), 'references/rf_baselines', f'{dataset}_harmony.csv')
    logger.info(f"... optimization results will be saved in {csvpath}")
    optimized_params = rf_utils.optimize_classifier(train_x_arr, train_y, test_x_arr, test_y, csvpath, n_iter=n_iter)
    # Create the optimized random forest model
    logger.info("Optimization finished, fitting optimized model to test and validation set...")
    clf_opt = RandomForestClassifier(**optimized_params, random_state=utils.SEED)
    clf_opt.fit(train_x_arr, train_y)
    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_x_arr)
    test_acc = accuracy_score(test_y, test_y_pred)
    logger.info(f"... test accuracy for harmony: {test_acc:.3f}")
    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x_arr)
    valid_acc = accuracy_score(valid_y, valid_y_pred)
    logger.info(f"... validation accuracy for harmony: {valid_acc:.3f}")
    logger.info('Done!')


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Create baseline random forest classifier for harmony')
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
        "-r", "--remove-highest-pitch",
        default=False,
        type=bool,
        help="Remove highest pitch in a chord",
    )
    parser.add_argument(
        "-v", "--valid-chords-count",
        default=VALID_CHORDS,
        type=int,
        help="Valid chords appear in this many tracks"
    )
    # Parse all arguments and create the forest
    args = vars(parser.parse_args())
    rf_harmony(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        valid_chords_count=args["valid_chords_count"],
        remove_highest_pitch=args['remove_highest_pitch']
    )
