#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to harmony & melody"""

import os

import numpy as np
from loguru import logger

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.rf_baselines.harmony import get_harmony_features
from deep_pianist_identification.rf_baselines.melody import get_melody_features


def rf_melody_harmony(
        dataset: str,
        n_iter: int,
        min_count: int,
        max_count: int,
        ngrams: list[int],
        remove_leaps: bool,
        classifier_type: str = "rf"
):
    """Fit the random forest to both melody and harmony features"""
    logger.info("Creating white box classifier using melody and harmony data!")
    logger.info(f'... using model type {classifier_type}')
    logger.info(f"... using ngrams {ngrams}, with remove_leaps {remove_leaps}")
    train_clips, test_clips, validation_clips = rf_utils.get_all_clips(dataset)
    # Melody extraction
    logger.info('---MELODY---')
    train_x_arr_mel, test_x_arr_mel, valid_x_arr_mel, mel_features, train_y_mel, test_y_mel, valid_y_mel = get_melody_features(
        train_clips=train_clips,
        test_clips=test_clips,
        validation_clips=validation_clips,
        min_count=min_count,
        max_count=max_count,
        ngrams=ngrams,
        remove_leaps=remove_leaps
    )
    # HARMONY EXTRACTION
    logger.info('---HARMONY---')
    train_x_arr_har, test_x_arr_har, valid_x_arr_har, har_features, train_y_har, test_y_har, valid_y_har = get_harmony_features(
        train_clips=train_clips,
        test_clips=test_clips,
        validation_clips=validation_clips,
        min_count=min_count,
        max_count=max_count,
        ngrams=ngrams,
        remove_leaps=remove_leaps
    )
    # Check targets are identical for both melody and harmony
    assert train_y_mel == train_y_har, "Melody and Harmony train targets are not identical (shouldn't happen!)"
    assert test_y_mel == test_y_har, "Melody and Harmony test targets are not identical (shouldn't happen!)"
    assert valid_y_mel == valid_y_har, "Melody and Harmony validation targets are not identical (shouldn't happen!)"
    # Combine corresponding arrays column-wise
    logger.info('---COMBINING ARRAYS---')
    train_x_arr = np.concatenate((train_x_arr_mel, train_x_arr_har), axis=1)
    test_x_arr = np.concatenate((test_x_arr_mel, test_x_arr_har), axis=1)
    valid_x_arr = np.concatenate((valid_x_arr_mel, valid_x_arr_har), axis=1)
    # Load the optimized parameter settings (or recreate them, if they don't exist)
    logger.info('---FITTING---')
    csvpath = os.path.join(
        utils.get_project_root(),
        'references/rf_baselines',
        f'{dataset}_{classifier_type}_harmony+melody.csv'
    )
    clf_opt, valid_acc = rf_utils.fit_classifier(
        train_x_arr, test_x_arr, valid_x_arr, train_y_mel, test_y_mel, valid_y_mel, csvpath, n_iter, classifier_type
    )
    # Computing parameter importance scores
    # logger.info('---CONCEPT IMPORTANCE---')
    # mel_concept_idxs, harm_concept_idxs = train_x_arr_mel.shape[1], train_x_arr_har.shape[1]
    # for idxs, concept in zip([mel_concept_idxs, harm_concept_idxs], ['melody', 'harmony']):
    #     temp = valid_x_arr.copy()
    #     temp = np.random.shuffle(temp[:, idxs])
    #     valid_y_permute_predict = clf_opt.predict(temp)
    #     valid_permute_acc = accuracy_score(valid_y_mel, valid_y_permute_predict)
    #     importance = valid_acc - valid_permute_acc
    #     logger.info(f"... feature importance for {concept}: {importance:.3f}")
    logger.info('Done!')


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Create white box classifier for melody + harmony')
    args = rf_utils.parse_arguments(parser)
    # Create the forest
    rf_melody_harmony(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        min_count=args["min_count"],
        ngrams=args['ngrams'],
        remove_leaps=args['remove_leaps'],
        max_count=args["max_count"],
        classifier_type=args["classifier_type"]
    )
