#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to harmony & melody"""

import os

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.rf_baselines.harmony import extract_chords
from deep_pianist_identification.rf_baselines.melody import NGRAMS, extract_melody_ngrams

VALID_NUM = 1


def rf_melody_harmony(dataset: str, n_iter: int, valid_count: int, ngrams: list[int], remove_highest_pitch: bool):
    logger.info("Creating baseline random forest classifier using melody and harmony data!")
    # Get clips from all splits of the dataset
    logger.info(f"Getting tracks from all splits for dataset {dataset}...")
    train_clips = list(rf_utils.get_split_clips("train", dataset))
    test_clips = list(rf_utils.get_split_clips("test", dataset))
    validation_clips = list(rf_utils.get_split_clips("validation", dataset))
    logger.info(f"... found {len(train_clips)} train tracks, "
                f"{len(test_clips)} test tracks, "
                f"{len(validation_clips)} validation tracks!")
    # MELODY EXTRACTION
    logger.info('---MELODY---')
    logger.info("Extracting n-grams from training tracks..")
    temp_x_mel, train_y_mel = rf_utils.extract_ngrams_from_clips(train_clips, extract_melody_ngrams, ngrams)
    logger.info("Extracting n-grams from testing tracks...")
    test_x_mel, test_y_mel = rf_utils.extract_ngrams_from_clips(test_clips, extract_melody_ngrams, ngrams)
    logger.info("Extracting n-grams from validation tracks...")
    valid_x_mel, valid_y_mel = rf_utils.extract_ngrams_from_clips(validation_clips, extract_melody_ngrams, ngrams)
    # Get those n-grams that appear in at least N tracks in the training dataset
    logger.info(f"Extracting n-grams which appear in at least {valid_count} training tracks...")
    valid_ngs = rf_utils.get_valid_ngrams(temp_x_mel, valid_count=valid_count)
    logger.info(f"... found {len(valid_ngs)} n-grams!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting features...")
    train_x_mel = rf_utils.drop_invalid_ngrams(temp_x_mel, valid_ngs)
    test_x_mel = rf_utils.drop_invalid_ngrams(test_x_mel, valid_ngs)
    valid_x_mel = rf_utils.drop_invalid_ngrams(valid_x_mel, valid_ngs)
    train_x_arr_mel, test_x_arr_mel, valid_x_arr_mel = rf_utils.format_features(train_x_mel, test_x_mel, valid_x_mel)
    # HARMONY EXTRACTION
    logger.info('---HARMONY---')
    logger.info("Extracting chords from training tracks..")
    temp_x_har, train_y_har = rf_utils.extract_ngrams_from_clips(train_clips, extract_chords, remove_highest_pitch)
    logger.info("Extracting chords from testing tracks...")
    test_x_har, test_y_har = rf_utils.extract_ngrams_from_clips(test_clips, extract_chords, remove_highest_pitch)
    logger.info("Extracting chords from validation tracks...")
    valid_x_har, valid_y_har = rf_utils.extract_ngrams_from_clips(validation_clips, extract_chords,
                                                                  remove_highest_pitch)
    # Get those n-grams that appear in at least N tracks in the training dataset
    logger.info(f"Extracting chords which appear in at least {valid_count} training tracks...")
    valid_chords = rf_utils.get_valid_ngrams(temp_x_har, valid_count=valid_count)
    logger.info(f"... found {len(valid_chords)} chords!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting harmony features...")
    train_x_har = rf_utils.drop_invalid_ngrams(temp_x_har, valid_chords)
    test_x_har = rf_utils.drop_invalid_ngrams(test_x_har, valid_chords)
    valid_x_har = rf_utils.drop_invalid_ngrams(valid_x_har, valid_chords)
    train_x_arr_har, test_x_arr_har, valid_x_arr_har = rf_utils.format_features(train_x_har, test_x_har, valid_x_har)
    # Combine corresponding arrays column-wise
    logger.info('---COMBINING ARRAYS---')
    train_x_arr = np.concatenate((train_x_arr_mel, train_x_arr_har), axis=1)
    test_x_arr = np.concatenate((test_x_arr_mel, test_x_arr_har), axis=1)
    valid_x_arr = np.concatenate((valid_x_arr_mel, valid_x_arr_har), axis=1)
    # Load the optimized parameter settings (or recreate them, if they don't exist)
    logger.info('---FITTING---')
    logger.info(f"Beginning parameter optimization with {n_iter} iterations...")
    csvpath = os.path.join(utils.get_project_root(), 'references/rf_baselines', f'{dataset}_harmony+melody.csv')
    logger.info(f"... optimization results will be saved in {csvpath}")
    # Check targets are identical for both melody and harmony
    assert train_y_mel == train_y_har, "Melody and Harmony train targets are not identical (shouldn't happen!)"
    assert test_y_mel == test_y_har, "Melody and Harmony test targets are not identical (shouldn't happen!)"
    assert valid_y_mel == valid_y_har, "Melody and Harmony validation targets are not identical (shouldn't happen!)"
    optimized_params = rf_utils.optimize_classifier(
        train_x_arr, train_y_mel, test_x_arr, test_y_mel, csvpath, n_iter=n_iter
    )
    # Create the optimized random forest model
    logger.info("Optimization finished, fitting optimized model to test and validation set...")
    clf_opt = RandomForestClassifier(**optimized_params, random_state=utils.SEED)
    clf_opt.fit(train_x_arr, train_y_mel)
    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_x_arr)
    test_acc = accuracy_score(test_y_mel, test_y_pred)
    logger.info(f"... test accuracy for melody: {test_acc:.3f}")
    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x_arr)
    valid_acc = accuracy_score(valid_y_mel, valid_y_pred)
    logger.info(f"... validation accuracy for melody: {valid_acc:.3f}")
    # Computing parameter importance scores
    logger.info('---CONCEPT IMPORTANCE---')
    mel_concept_idxs, harm_concept_idxs = train_x_arr_mel.shape[1], train_x_arr_har.shape[1]
    for idxs, concept in zip([mel_concept_idxs, harm_concept_idxs], ['melody', 'harmony']):
        temp = valid_x_arr.copy()
        temp = np.random.shuffle(temp[:, idxs])
        valid_y_permute_predict = clf_opt.predict(temp)
        valid_permute_acc = accuracy_score(valid_y_mel, valid_y_permute_predict)
        importance = valid_acc - valid_permute_acc
        logger.info(f"... feature importance for {concept}: {importance:.3f}")
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
        '-l', '--ngrams',
        nargs='+',
        type=int,
        help="Extract these n-grams",
        default=NGRAMS
    )
    parser.add_argument(
        "-v", "--valid-count",
        default=VALID_NUM,
        type=int,
        help="Valid chords or n-grams appear in this many tracks"
    )
    # Parse all arguments and create the forest
    args = vars(parser.parse_args())
    rf_melody_harmony(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        valid_count=args["valid_count"],
        remove_highest_pitch=args['remove_highest_pitch'],
        ngrams=args['ngrams']
    )
