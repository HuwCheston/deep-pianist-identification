#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to harmony & melody"""

import os

import numpy as np
from loguru import logger

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.plotting import StripplotTopKFeatures
from deep_pianist_identification.rf_baselines.harmony import get_harmony_features
from deep_pianist_identification.rf_baselines.melody import get_melody_features


def rf_melody_harmony(
        dataset: str,
        n_iter: int,
        min_count: int,
        max_count: int,
        ngrams: list[int],
        remove_leaps: bool,
        classifier_type: str = "rf",
        scale: bool = True
):
    """Fit the random forest to both melody and harmony features"""
    logger.info("Creating white box classifier using melody and harmony data!")
    logger.info(f'... using model type {classifier_type}')
    logger.info(f"... using ngrams {ngrams}, with remove_leaps {remove_leaps}")
    # Get the class mapping dictionary from the dataset
    class_mapping = utils.get_class_mapping(dataset)
    # Get all clips from the given dataset
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
    # Scale the data if required (never for multinomial naive Bayes as this expects count data)
    if scale and classifier_type != "nb":
        train_x_arr, test_x_arr, valid_x_arr = rf_utils.scale_features(train_x_arr, test_x_arr, valid_x_arr)
    # Optimize the classifier
    clf_opt, valid_acc, best_params = rf_utils.fit_classifier(
        train_x_arr, test_x_arr, valid_x_arr, train_y_mel, test_y_mel, valid_y_mel,
        csvpath, n_iter, classifier_type
    )
    # Get feature importance scores
    logger.info('---OVERALL FEATURE IMPORTANCE---')
    rf_utils.get_harmony_feature_importance(
        harmony_features=valid_x_arr_har,
        melody_features=valid_x_arr_mel,
        classifier=clf_opt,
        initial_acc=valid_acc,
        y_actual=valid_y_mel
    )
    rf_utils.get_melody_feature_importance(
        harmony_features=valid_x_arr_har,
        melody_features=valid_x_arr_mel,
        classifier=clf_opt,
        initial_acc=valid_acc,
        y_actual=valid_y_mel
    )
    # Get feature importance scores for individual performers
    if classifier_type == "lr" or classifier_type == "svm":
        logger.info('---PERFORMER FEATURE IMPORTANCE---')
        # Get array of all features, with identifying string at the start
        all_features = np.array(['M_' + f for f in mel_features] + ['H_' + f for f in har_features])
        # Combine train, test and validation datasets together
        full_x = np.vstack([train_x_arr, test_x_arr, valid_x_arr])
        full_y = np.hstack([train_y_mel, test_y_mel, valid_y_mel])
        # Scale the data (if it requires this)
        if scale:
            full_x = rf_utils.scale_feature(full_x)
        # Get the classifier weights from the model
        logger.info('... getting classifier weights with bootstrapping (this may take a while)')
        weights, boot_weights = rf_utils.get_classifier_weights(
            full_x, full_y, all_features, best_params, classifier_type
        )
        # Create the dictionary of top (and bottom) K features for both harmony and melody
        topk_dict = rf_utils.get_topk_features(weights, boot_weights, all_features, class_mapping)
        # Iterate through all performers and concepts
        for perf in topk_dict.keys():
            for concept in ['melody', 'harmony']:
                # Create the plot and save in the default (reports/figures/whitebox) directory
                sp = StripplotTopKFeatures(topk_dict[perf][concept], pianist_name=perf, concept_name=concept)
                sp.create_plot()
                sp.save_fig()
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
        classifier_type=args["classifier_type"],
        scale=args["scale"]
    )
