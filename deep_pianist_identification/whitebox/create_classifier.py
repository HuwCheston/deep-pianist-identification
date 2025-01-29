#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create whitebox classifiers with command line interface"""

import os

import numpy as np
from loguru import logger

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.whitebox import wb_utils
from deep_pianist_identification.whitebox.classifiers import fit_classifier
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
):
    """Fit the random forest to both melody and harmony features"""
    logger.info("Creating white box classifier using melody and harmony data!")
    logger.info(f'... using model type {classifier_type}')
    logger.info(f"... using feature sizes {feature_sizes}")
    # Get the class mapping dictionary from the dataset
    class_mapping = utils.get_class_mapping(dataset)
    # Get all clips from the given dataset
    train_clips, test_clips, validation_clips = wb_utils.get_all_clips(dataset)
    # Melody extraction
    logger.info('---MELODY---')
    train_x_full_mel, test_x_full_mel, valid_x_full_mel, train_y_mel, test_y_mel, valid_y_mel = get_melody_features(
        train_clips=train_clips,
        test_clips=test_clips,
        validation_clips=validation_clips,
        feature_sizes=feature_sizes,
    )
    train_x_arr_mel, test_x_arr_mel, valid_x_arr_mel, mel_features = drop_invalid_features(
        train_x_full_mel, test_x_full_mel, valid_x_full_mel, min_count, max_count
    )
    # HARMONY EXTRACTION
    logger.info('---HARMONY---')
    train_x_full_har, test_x_full_har, valid_x_full_har, train_y_har, test_y_har, valid_y_har = get_harmony_features(
        train_clips=train_clips,
        test_clips=test_clips,
        validation_clips=validation_clips,
        feature_sizes=feature_sizes,
    )
    train_x_arr_har, test_x_arr_har, valid_x_arr_har, har_features = drop_invalid_features(
        train_x_full_har, test_x_full_har, valid_x_full_har, min_count, max_count
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
        f'{dataset}_{classifier_type}_harmony+melody_{"".join([str(i) for i in feature_sizes])}.csv'
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
    # Domain/feature importance: this class will do all analysis and create plots/outputs
    logger.info('---EXPLAINING: DOMAIN IMPORTANCE---')
    logger.info(f'... n_iter {n_iter}, initial accuracy {valid_acc}')
    permute_explainer = DomainExplainer(
        harmony_features=valid_x_arr[:, valid_x_arr_mel.shape[1]:],  # we can subset to just get the harmony features
        melody_features=valid_x_arr[:, :valid_x_arr_mel.shape[1]],  # same for the melody features
        y=valid_y_mel,
        classifier=clf_opt,
        init_acc=valid_acc,
        n_iter=n_iter,
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
    dataset_idxs: np.array = wb_utils.get_database_mapping(train_clips, test_clips, validation_clips)
    # Decompose feature counts using PCA
    logger.info('---EXPLAINING: DECOMPOSING FEATURE COUNTS---')
    pc = PCAFeatureCountsExplainer(
        feature_counts=all_xs[:, :len(mel_features)],  # first decomposing melody features
        targets=all_ys,
        feature_names=mel_features,
        class_mapping=class_mapping,
        feature_size=3,  # plotting 4-grams
        n_components=4,
        feature_type='melody'
    )
    pc.explain()
    pc.create_outputs()
    pc = PCAFeatureCountsExplainer(
        feature_counts=all_xs[:, len(mel_features):],  # second decomposing harmony features
        targets=all_ys,
        feature_names=har_features,
        class_mapping=class_mapping,
        feature_size=3,  # plotting 4-grams
        n_components=4,
        feature_type='harmony'
    )
    pc.explain()
    pc.create_outputs()
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
    database_explainer.explain()
    database_explainer.create_outputs()
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
    lr_exp.explain()
    lr_exp.create_outputs()


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
        database_k_coefs=args["database_k_coefs"]
    )
    logger.info('Done!')

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from deep_pianist_identification.preprocessing.m21_utils import intervals_to_pitches
# from math import isclose
#
# fig, axs = plt.subplots(1, 2, figsize=(plotting.WIDTH, plotting.WIDTH // 2), sharex=True, sharey=True)
# plt.rcParams.update({'font.size': 18})
#
# train_x_arr_norm, test_x_arr_norm, valid_x_arr_norm = wb_utils.scale_features(train_x_arr, test_x_arr, valid_x_arr)
# all_features = np.vstack([train_x_arr, test_x_arr, valid_x_arr])
# all_targets = np.concatenate([train_y_mel, test_y_mel, valid_y_mel])
#
# all_features.shape
# gram4_idxs = [n for n, i in enumerate(mel_features) if len(eval(i)) == 3]
# all_gram4 = all_features[:, gram4_idxs]
# pca = PCA(n_components=32, random_state=utils.SEED)
# x_new = pca.fit_transform(all_gram4)
#
# for ax, pca1_idx, pca2_idx in zip(axs.flatten(), [0, 2, 4, 6], [2, 4, 6, 8]):
#

# class BigramplotPCAFeatureCount(plotting.BasePlot):
#     def __init__(self, df: pd.DataFrame, n_components_to_plot: int = 4, **kwargs):
#         super().__init__(**kwargs)
#         self.n_components_to_plot = n_components_to_plot
#         self.df = self._format_df(df)
#         self.fig, self.ax = plt.subplots(2, n_components_to_plot // 2, figsize=(plotting.WIDTH, plotting.WIDTH),
#                                          sharex=True, sharey=True)
#
#     def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
#         return df[df['n_component'] < self.n_components_to_plot].reset_index(drop=True)
#
#     @staticmethod
#     def _scale_loading(loading: pd.Series) -> pd.Series:
#         mm = MinMaxScaler(feature_range=(-1, 1))
#         return pd.Series(mm.fit_transform(loading.to_numpy().reshape(-1, 1)).flatten())
#
#     def _create_plot(self):
#         comps = sorted(self.df['n_component'].unique().tolist())
#
#         for ax, comp1, comp2 in zip(self.ax, comps, comps[1:]):
#             sub = self.df[self.df['n_component'].isin([comp1, comp2])]
#             sub['loading'] = self._scale_loading(sub['loading'])
#             perf_xy = np.column_stack([
#                 sub[(sub['n_component'] == comp1) & (sub['type'] == 'performer')]['loading'].to_numpy(),
#                 sub[(sub['n_component'] == comp2) & (sub['type'] == 'performer')]['loading'].to_numpy()
#             ])
#
#             break

#     print(pca1_idx, pca2_idx)
#     data = np.column_stack([x_new[:, pca1_idx:pca2_idx], all_targets])
#     # Extract the unique groups from column 3
#     unique_groups = np.unique(data[:, 2])
#
#     # Calculate the average for each group
#     result = []
#     for group in unique_groups:
#         group_data = data[data[:, 2] == group]  # Filter rows for this group
#         group_mean = group_data.mean(axis=0)  # Average across rows
#         result.append(group_mean)
#
#     result = np.array(result)
#     coeff = np.transpose(pca.components_[pca1_idx:pca2_idx, :])
#     labels = np.array(mel_features)[gram4_idxs]
#
#     xs = result[:, 0]
#     ys = result[:, 1]
#     n = coeff.shape[0]
#     scalex = 1.0 / (xs.max() - xs.min())
#     scaley = 1.0 / (ys.max() - ys.min())
#
#     xs *= scalex
#     ys *= scaley
#
#     loads_x = np.argsort(np.abs(xs))[::-1][:3]
#     loads_y = np.argsort(np.abs(ys))[::-1][:3]
#     loads_total = np.concatenate([loads_x, loads_y])
#     for load_idx in loads_total:
#         perf = class_mapping[load_idx]
#         mark = f'${perf.split(" ")[0][0]}. {perf.split(" ")[1][:5]}$'
#         ax.annotate(mark, (xs[load_idx], ys[load_idx]), label=class_mapping[load_idx])
#
#     loads_x = np.argsort(np.abs(coeff[:, 0]))[::-1][:3]
#     loads_y = np.argsort(np.abs(coeff[:, 1]))[::-1][:3]
#     loads_total = np.concatenate([loads_x, loads_y])
#
#     seen_x, seen_y = [], []
#     for load_idx in loads_total:
#         x, y = coeff[load_idx, 0], coeff[load_idx, 1]
#         ax.arrow(0, 0, x, y, color='r', alpha=0.5)
#         iscl = all([[isclose(x, x_) for x_ in seen_x], [isclose(y, y_) for y_ in seen_y]])
#         if iscl:
#             for i in range(5):
#                 x *= 1.05
#                 y *= 1.05
#                 if not all([[isclose(x, x_) for x_ in seen_x], [isclose(y, y_) for y_ in seen_y]]):
#                     break
#
#         ax.text(x, y, intervals_to_pitches(labels[load_idx]), color='g', ha='center', va='center')
#         seen_y.append(y)
#         seen_x.append(x)
#
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_xlabel("PC{}".format(pca1_idx + 1))
#     ax.set_ylabel("PC{}".format(pca2_idx))
#     circle1 = plt.Circle((0, 0), 1., color='black', fill=False)
#     ax.add_patch(circle1)
#     ax.grid()
# fig.tight_layout()
# plt.show()
