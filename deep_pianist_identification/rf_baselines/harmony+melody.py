#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to harmony & melody"""

import os

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.plotting import StripplotTopKFeatures, BarPlotWhiteboxDatabaseCoeficients
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
        y_actual=valid_y_mel,
        n_iter=n_iter
    )
    rf_utils.get_melody_feature_importance(
        harmony_features=valid_x_arr_har,
        melody_features=valid_x_arr_mel,
        classifier=clf_opt,
        initial_acc=valid_acc,
        y_actual=valid_y_mel,
        n_iter=n_iter
    )
    # Get feature importance scores for individual performers
    if classifier_type == "lr" or classifier_type == "svm":
        # Get array of all features, with identifying string at the start
        all_features = np.array(['M_' + f for f in mel_features] + ['H_' + f for f in har_features])
        # Combine train, test and validation datasets together
        full_x = np.vstack([train_x_arr, test_x_arr, valid_x_arr])
        full_y = np.hstack([train_y_mel, test_y_mel, valid_y_mel])

        logger.info('---DATASET FEATURE CORRELATIONS---')
        dataset_mapping = rf_utils.get_database_mapping(train_clips, test_clips, validation_clips)
        full_x_mel = np.vstack([train_x_arr_mel, test_x_arr_mel, valid_x_arr_mel])
        full_x_har = np.vstack([train_x_arr_har, test_x_arr_har, valid_x_arr_har])
        # Get the model coefficients for both melody and harmony
        jtd_mel_coef, pijama_mel_coef = rf_utils.get_database_coefs(
            full_x_mel,
            full_y,
            dataset_mapping,
            classifier_type=classifier_type,
            classifier_params=best_params,
            scale=scale
        )
        jtd_har_coef, pijama_har_coef = rf_utils.get_database_coefs(
            full_x_har,
            full_y,
            dataset_mapping,
            classifier_type=classifier_type,
            classifier_params=best_params,
            scale=scale
        )
        # Combine these to get the melody and harmony correlations (one value per performer)
        mel_corr = rf_utils.get_database_coef_corrs(jtd_mel_coef, pijama_mel_coef, class_mapping)
        har_corr = rf_utils.get_database_coef_corrs(jtd_har_coef, pijama_har_coef, class_mapping)
        # Set an indexing value and concatenate row-wise
        mel_corr['feature'] = 'Melody'
        har_corr['feature'] = 'Harmony'
        all_corrs = pd.concat([mel_corr, har_corr], axis=0).reset_index(drop=True)
        # Create the bar plot
        bp = BarPlotWhiteboxDatabaseCoeficients(all_corrs)
        bp.create_plot()
        # Save the plot
        outpath = os.path.join(utils.get_project_root(), 'reports/figures/whitebox/barplot_database_correlations.png')
        bp.save_fig(outpath)

        logger.info('---PERFORMER FEATURE IMPORTANCE---')
        # Scale the data (if it requires this)
        if scale:
            full_x = rf_utils.scale_feature(full_x)
        # Get the classifier weights from the model
        logger.info('... getting classifier weights with bootstrapping (this may take a while)')
        weights, boot_weights = rf_utils.get_classifier_weights(
            full_x, full_y, all_features, best_params, classifier_type, n_boot=n_iter
        )
        # Create the dictionary of top (and bottom) K features for both harmony and melody
        logger.info('... getting top-/bottom-K features for all performers')
        topk_dict = rf_utils.get_topk_features(weights, boot_weights, all_features, class_mapping)
        logger.info('... creating performer strip plots')
        # Iterate through all performers and concepts
        for perf in topk_dict.keys():
            for concept in ['melody', 'harmony']:
                # Create the plot and save in the default (reports/figures/whitebox) directory
                sp = StripplotTopKFeatures(topk_dict[perf][concept], pianist_name=perf, concept_name=concept)
                sp.create_plot()
                sp.save_fig()
        logger.info('... creating piano rolls + MIDI for clips matching top-K melody ngrams')
        # Iterate through all performers again
        all_clips = train_clips + test_clips + validation_clips
        for perf_idx, perf_name in tqdm(class_mapping.items()):
            # Get tracks by this performer
            perf_tracks = [c for c in all_clips if c[2] == perf_idx]
            # Get data for this performer
            perf_data = topk_dict[perf_name]['melody']
            # Iterate through all tracks by the performer
            for track_path, n_clips, _ in perf_tracks:
                # Iterate through all clips for the track
                for clip_idx in range(n_clips):
                    # Get the filepath
                    clip_idx = 'clip_' + str(clip_idx).zfill(3) + '.mid'
                    clip_path = os.path.join(track_path, clip_idx)
                    # Iterate through all n-grams by the performer
                    for ng in perf_data['top_names']:
                        manager = rf_utils.MIDIOutputManager(clip_path, perf_name, ng)
                        manager.run()
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
