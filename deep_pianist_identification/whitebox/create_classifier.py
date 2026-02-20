#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create whitebox classifiers with command line interface"""

import os
from copy import deepcopy

import numpy as np
from loguru import logger

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.whitebox import wb_utils
from deep_pianist_identification.whitebox.classifiers import fit_classifier, fit_with_optimization
from deep_pianist_identification.whitebox.explainers import (
    LRWeightExplainer, DomainExplainer, DatabasePermutationExplainer, PCAFeatureCountsExplainer
)
from deep_pianist_identification.whitebox.features import (
    get_harmony_features, get_melody_features, drop_invalid_features
)


def create_classifier(
        dataset: str,
        n_iter: int,
        min_count: int,
        max_count: int,
        feature_sizes: list[int],
        database_k_coefs: int,
        classifier_type: str = "rf",
        scale: bool = True,
        optimize: bool = False,
        domain_boot_proportional: bool = False,
        subsume_ngrams: bool = False
):
    """Fit the random forest to both melody and harmony features"""
    logger.info("Creating white box classifier using melody and harmony data!")
    logger.info(f'... using model type {classifier_type}')
    logger.info(f"... using feature sizes {feature_sizes}")
    logger.info(f"... subsuming n-grams {subsume_ngrams}")

    # Get the class mapping dictionary from the dataset
    class_mapping = utils.get_class_mapping(dataset)

    # Get all clips from the given dataset
    train_clips, test_clips, validation_clips = wb_utils.get_all_clips(dataset)
    # train_clips = train_clips[:50]
    # test_clips = test_clips[:50]
    # validation_clips = validation_clips[:50]

    # Melody extraction
    logger.info('---MELODY---')
    train_x_full_mel, test_x_full_mel, valid_x_full_mel, train_y_mel, test_y_mel, valid_y_mel = get_melody_features(
        train_clips=train_clips,
        test_clips=test_clips,
        validation_clips=validation_clips,
        feature_sizes=feature_sizes,
    )

    # plots of unique track appearances / counts for 4-grams
    bp = plotting.BarPlotWhiteboxFeatureUniqueTracks(
        train_x_full_mel + test_x_full_mel + valid_x_full_mel
    )
    bp.create_plot()
    bp.save_fig()
    bp = plotting.BarPlotWhiteboxNGramFeatureCounts(
        train_x_full_mel + test_x_full_mel + valid_x_full_mel
    )
    bp.create_plot()
    bp.save_fig()
    train_x_arr_mel, test_x_arr_mel, valid_x_arr_mel, mel_features = drop_invalid_features(
        train_x_full_mel, test_x_full_mel, valid_x_full_mel, min_count, max_count, subsume_ngrams=subsume_ngrams
    )

    # HARMONY EXTRACTION
    logger.info('---HARMONY---')
    train_x_full_har, test_x_full_har, valid_x_full_har, train_y_har, test_y_har, valid_y_har = get_harmony_features(
        train_clips=train_clips,
        test_clips=test_clips,
        validation_clips=validation_clips,
        feature_sizes=feature_sizes,
    )
    # never subsume harmony n-grams
    train_x_arr_har, test_x_arr_har, valid_x_arr_har, har_features = drop_invalid_features(
        train_x_full_har, test_x_full_har, valid_x_full_har, min_count, max_count, subsume_ngrams=False
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

    # Create a plot of the feature counts
    bp = plotting.BarPlotWhiteboxFeatureCounts(
        np.vstack([train_x_arr, test_x_arr, valid_x_arr]), mel_features, har_features
    )
    bp.create_plot()
    bp.save_fig()

    # Load the optimized parameter settings (or recreate them, if they don't exist)
    logger.info('---FITTING---')
    csvpath = os.path.join(
        utils.get_project_root(),
        'references/whitebox',
        f'{dataset}_{classifier_type}_harmony+melody_{"".join([str(i) for i in feature_sizes])}_min{min_count}_max{max_count}.csv'
    )

    # Scale the data if required (never for multinomial naive Bayes as this expects count data)
    train_x_raw, test_x_raw, valid_x_raw = deepcopy(train_x_arr), deepcopy(test_x_arr), deepcopy(valid_x_arr)
    if scale and classifier_type != "nb":
        train_x_arr, test_x_arr, valid_x_arr = wb_utils.scale_features(train_x_arr, test_x_arr, valid_x_arr)

    # Decompose feature counts using PCA: first melody, then harmony
    logger.info('---EXPLAINING: DECOMPOSING FEATURE COUNTS---')
    all_xs_raw = np.vstack([train_x_raw, test_x_raw, valid_x_raw])
    all_ys_raw = np.hstack([train_y_mel, test_y_mel, valid_y_mel])
    for feat_type, feat_names, xs in zip(
            ['melody', 'harmony'],
            [mel_features, har_features],
            [all_xs_raw[:, :len(mel_features)], all_xs_raw[:, len(mel_features):]]
    ):
        pc = PCAFeatureCountsExplainer(
            feature_counts=xs,  # these will be z-transformed as part of the class
            targets=all_ys_raw,
            feature_names=feat_names,
            class_mapping=class_mapping,
            feature_size=3,  # plotting 4-grams
            n_components=4,
            feature_type=feat_type
        )
        logger.info(f'... shape of features into PCA: {pc.counts.shape}, n_components: {pc.n_components}')
        pc.explain()
        pc.create_outputs()

    # Optimize the classifier
    if not optimize:
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
    else:
        clf_opt, valid_acc, best_params = fit_with_optimization(
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

    # Domain/feature importance: this class will do all analysis and create plots/outputs
    logger.info('---EXPLAINING: DOMAIN IMPORTANCE---')
    logger.info(f'... n_iter {n_iter}, initial accuracy {valid_acc}')
    if domain_boot_proportional:
        n_boot_features_har = int(len(har_features) * 0.1)
        n_boot_features_mel = int(len(mel_features) * 0.1)
        logger.info(f"... using proportional values of K: "
                    f"melody {n_boot_features_mel} / {len(mel_features)}, "
                    f"harmony {n_boot_features_har} / {len(har_features)}")
    else:
        n_boot_features_har = None
        n_boot_features_mel = None

    permute_explainer = DomainExplainer(
        harmony_features=valid_x_arr[:, valid_x_arr_mel.shape[1]:],  # we can subset to just get the harmony features
        melody_features=valid_x_arr[:, :valid_x_arr_mel.shape[1]],  # same for the melody features
        y=valid_y_mel,
        classifier=clf_opt,
        init_acc=valid_acc,
        n_iter=n_iter,
        n_boot_features_har=n_boot_features_har,
        n_boot_features_mel=n_boot_features_mel
    )
    permute_explainer.explain()
    permute_explainer.create_outputs(classifier_type)
    logger.info(
        f'all melody: {permute_explainer.df.iloc[2]["mean"]}, SD {permute_explainer.df.iloc[2]["std"]}\n'
        f'all harmony: {permute_explainer.df.iloc[0]["mean"]}, SD {permute_explainer.df.iloc[0]["std"]}\n'
        f'bootstrap melody: {permute_explainer.df.iloc[3]["mean"]}, SD {permute_explainer.df.iloc[3]["std"]}\n'
        f'bootstrap harmony: {permute_explainer.df.iloc[1]["mean"]}, SD {permute_explainer.df.iloc[1]["std"]}'
    )

    # Don't create the other outputs if the classifier isn't a logistic regression
    if classifier_type != "lr":
        return

    # Concatenate all features and feature names
    all_xs = np.vstack([train_x_arr, test_x_arr, valid_x_arr])
    all_ys = np.hstack([train_y_mel, test_y_mel, valid_y_mel])
    feature_names = np.array(['M_' + f for f in mel_features] + ['H_' + f for f in har_features])
    dataset_idxs = wb_utils.get_database_mapping(train_clips, test_clips, validation_clips)

    # Correlation between top-k coefficients from the full model for individual database models
    logger.info('---EXPLAINING: DATASET FEATURE CORRELATIONS---')
    logger.info(f'...  k: {database_k_coefs}, x_shape: {all_xs.shape}, y_shape: {all_ys.shape},')
    database_explainer = DatabasePermutationExplainer(
        x=all_xs,
        y=all_ys,
        k=database_k_coefs,
        dataset_idxs=dataset_idxs,
        feature_names=feature_names,
        class_mapping=class_mapping,
        classifier_params=best_params,
        classifier_type=classifier_type,
        n_iter=n_iter
    )
    # TODO: needs fixing
    # database_explainer.explain()
    # database_explainer.create_outputs()

    # Log the mean and SD coefficients for melody and harmony to the console
    logger.info(
        f'mean melody r {database_explainer.mel_coefs.mean()}, SD {database_explainer.mel_coefs.std()}\n'
        f'mean harmony r {database_explainer.har_coefs.mean()}, SD {database_explainer.har_coefs.std()}'
    )

    # Weights for top k features for each performer across melody/harmony features
    logger.info('---EXPLAINING: MODEL WEIGHTS (LOCAL)---')
    logger.info(f'... shapes: x {all_xs.shape}, y {all_ys.shape}, feature names {feature_names.shape}')
    logger.info(f'... n_iter {n_iter}')
    lr_exp = LRWeightExplainer(
        x=all_xs,
        y=np.hstack([train_y_mel, test_y_mel, valid_y_mel]),
        feature_names=feature_names,
        class_mapping=class_mapping,
        classifier_type=classifier_type,
        classifier_params=best_params,
        use_odds_ratios=False,
        n_iter=n_iter,
    )
    # TODO: needs fixing
    # lr_exp.explain()
    # lr_exp.create_outputs()


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
        feature_sizes=args['feature_sizes'],
        max_count=args["max_count"],
        classifier_type=args["classifier_type"],
        scale=args["scale"],
        database_k_coefs=args["database_k_coefs"],
        optimize=args["optimize"],
        domain_boot_proportional=args["domain_boot_proportional"],
        subsume_ngrams=args["subsume_ngrams"],
    )
    logger.info('Done!')
