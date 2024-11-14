#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create whitebox classifiers with command line interface"""

import os

import numpy as np
from loguru import logger

from deep_pianist_identification import utils
from deep_pianist_identification.whitebox import wb_utils
from deep_pianist_identification.whitebox.classifiers import fit_classifier
from deep_pianist_identification.whitebox.explainers import (
    LRWeightExplainer, DatabaseExplainer, PermutationExplainer
)
from deep_pianist_identification.whitebox.features import get_harmony_features, get_melody_features


def create_classifier(
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
    train_clips, test_clips, validation_clips = wb_utils.get_all_clips(dataset)
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
        'references/whitebox',
        f'{dataset}_{classifier_type}_harmony+melody.csv'
    )
    # Scale the data if required (never for multinomial naive Bayes as this expects count data)
    if scale and classifier_type != "nb":
        train_x_arr, test_x_arr, valid_x_arr = wb_utils.scale_features(train_x_arr, test_x_arr, valid_x_arr)
    # Optimize the classifier
    clf_opt, valid_acc, best_params = fit_classifier(
        train_x_arr,
        test_x_arr,
        valid_x_arr,
        train_y_mel,
        test_y_mel,
        valid_y_mel,
        csvpath,
        n_iter,
        classifier_type
    )
    # Permutation feature importance: this class will do all analysis and create plots/outputs
    logger.info('---PERMUTATION FEATURE IMPORTANCE---')
    permute_explainer = PermutationExplainer(
        harmony_features=valid_x_arr_har,
        melody_features=valid_x_arr_mel,
        y=valid_y_mel,
        classifier=clf_opt,
        init_acc=valid_acc,
        scale=scale,
        n_iter=n_iter,
        n_boot_features=n_iter // 10
    )
    permute_explainer.explain()
    permute_explainer.create_outputs()
    # Correlation between top-k coefficients from the full model for individual database models
    logger.info('---DATASET FEATURE CORRELATIONS---')
    database_explainer = DatabaseExplainer(
        x=np.vstack([train_x_arr, test_x_arr, valid_x_arr]),
        y=np.hstack([train_y_mel, test_y_mel, valid_y_mel]),
        dataset_idxs=wb_utils.get_database_mapping(train_clips, test_clips, validation_clips),
        feature_names=np.array(['M_' + f for f in mel_features] + ['H_' + f for f in har_features]),
        class_mapping=class_mapping,
        classifier_params=best_params,
        classifier_type=classifier_type,
    )
    database_explainer.explain()
    database_explainer.create_outputs()
    # Weights for top k features for each performer across melody/harmony features
    lr_exp = LRWeightExplainer(
        x=np.vstack([train_x_arr, test_x_arr, valid_x_arr]),
        y=np.hstack([train_y_mel, test_y_mel, valid_y_mel]),
        feature_names=np.array(['M_' + f for f in mel_features] + ['H_' + f for f in har_features]),
        class_mapping=class_mapping,
        classifier_type=classifier_type,
        classifier_params=best_params,
        use_odds_ratios=True,
        n_iter=n_iter,
    )
    lr_exp.explain()
    lr_exp.create_outputs()
    logger.info('Done!')


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Create white box classifier')
    args = wb_utils.parse_arguments(parser)
    # Create the forest
    create_classifier(
        dataset=args['dataset'],
        n_iter=args["n_iter"],
        min_count=args["min_count"],
        ngrams=args['ngrams'],
        remove_leaps=args['remove_leaps'],
        max_count=args["max_count"],
        classifier_type=args["classifier_type"],
        scale=args["scale"]
    )
