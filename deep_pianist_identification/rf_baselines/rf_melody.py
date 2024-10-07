#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random forest baseline applied to skylined MIDI melody files"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pretty_midi import PrettyMIDI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler

import deep_pianist_identification.rf_baselines.rf_utils as rf_utils
from deep_pianist_identification import utils
from deep_pianist_identification.extractors import MelodyExtractor, ExtractorError


def _get_split_clips(split_path: str):
    test_clips = pd.read_csv(split_path, index_col=0, delimiter=',')
    for idx, track in test_clips.iterrows():
        track_path = os.path.join(utils.get_project_root(), 'data/clips', track['track'])
        track_clips = len(os.listdir(track_path))
        yield track_path, track_clips, track['pianist']


def _extract_ngrams(tpath: str, nclips: int, pianist: str) -> tuple[dict, int]:
    track_3gs, track_4gs = {}, {}
    for idx in range(nclips):
        # Load the clip in as a PrettyMIDI object
        clip_path = os.path.join(utils.get_project_root(), tpath, f'clip_{str(idx).zfill(3)}.mid')
        loaded = PrettyMIDI(clip_path)
        # Try and create the piano roll, but skip if we end up with zero notes
        try:
            roll = MelodyExtractor(loaded, clip_start=0)
        except ExtractorError as e:
            logger.warning(f'Errored for {tpath}, clip {idx}, skipp! {e}')
            continue
        # Sort the notes by onset start time and calculate the intervals
        melody = sorted(roll.output_midi.instruments[0].notes, key=lambda x: x.start)
        intervals = [i2.pitch - i1.pitch for i1, i2 in zip(melody[0:], melody[1:])]
        # TODO: fix this to work with different values of NGRAMS
        for n, di in zip(rf_utils.NGRAMS, [track_3gs, track_4gs]):
            grams = [intervals[i: i + n] for i in range(len(intervals) - n + 1)]
            for ng in grams:
                ng = str(ng)
                if ng not in di.keys():
                    di[ng] = 1
                else:
                    di[ng] += 1
    return track_3gs | track_4gs, pianist


def extract_ngrams_for_split(split_name: str) -> tuple[list[dict], list[int]]:
    assert split_name in ["test", "train"], "`split_name` must be either 'test' or 'train'"
    # Get n-grams from training data set
    split_path = os.path.join(utils.get_project_root(), 'references/data_splits', f'{split_name}_split.csv')
    split = _get_split_clips(split_path)
    with Parallel(n_jobs=1, verbose=5) as par:
        res = par(delayed(_extract_ngrams)(p, n, pi) for p, n, pi in list(split))
    return zip(*res)


def get_valid_ngrams(list_of_ngram_dicts: list[dict], valid_count: int = rf_utils.VALID_NGRAMS) -> list[str]:
    ngram_counts = defaultdict(int)
    for d in list_of_ngram_dicts:
        for key in d:
            ngram_counts[key] += 1
    return [key for key, count in ngram_counts.items() if count >= valid_count]


def format_features(train_features: list[dict], test_features: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    # Convert training + test features to dataframe and fill NA values
    train_df = pd.DataFrame(train_features).fillna(0)
    test_df = pd.DataFrame(test_features).fillna(0)
    # Set identifying columns
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    # Concatenate both dataframes row-wise; this ensures that we have the correct number of columns for both
    conc = pd.concat([train_df, test_df], axis=0)
    # Return a tuple of arrays comprising training features and testing features
    return (conc[conc['split'] == sp].drop(columns='split').to_numpy() for sp in ["test", "train"])


def _optimize_step(train_features, train_targets, test_features, test_targets, parameters, iteration) -> dict:
    try:
        cached_results = rf_utils.safe_load_csv(
            os.path.join(utils.get_project_root(), 'references/rf_baseline_optimizations', 'melody.csv')
        )
    except FileNotFoundError:
        cached_results = []

    try:
        cached_res = [o for o in cached_results if o['iteration'] == iteration][0]
        # assert all(k in cached_res and parameters[k] == cached_res[k] for k in to_check)
    except (IndexError, AssertionError):
        pass
    else:
        return cached_res

    # Create the forest model
    forest = RandomForestClassifier(**parameters, random_state=utils.SEED)
    # Fit the model to the training data and get the test data accuracy
    forest.fit(train_features, train_targets)
    acc = accuracy_score(test_targets, forest.predict(test_features))
    # Create the results dictionary and save
    results_dict = {'accuracy': acc, 'iteration': iteration, **parameters}
    rf_utils.safe_save_csv(
        results_dict,
        os.path.join(utils.get_project_root(), "references/rf_baseline_optimizations"),
        "melody.csv"
    )
    # Return the results for this optimization step
    return results_dict


def optimize_forest(
        train_features: np.ndarray, train_targets: list, test_features: np.ndarray, test_targets: np.ndarray
) -> dict:
    # Create the parameter sampling instance with the required parameters, number of iterations, and random state
    sampler = ParameterSampler(rf_utils.OPTIMIZE_PARAMS, n_iter=rf_utils.N_ITER, random_state=utils.SEED)
    # Use lazy parallelization to create the forest and fit to the data
    with Parallel(n_jobs=-1, verbose=5) as par:
        fit = par(delayed(_optimize_step)(train_features, train_targets, test_features, test_targets, params, num)
                  for num, params in enumerate(sampler))
    # Get the parameter combination that yielded the best test set accuracy
    best_params = max(fit, key=lambda x: x["accuracy"])
    return {k: v for k, v in best_params.items() if k not in ["accuracy", "iteration"]}


if __name__ == "__main__":
    from loguru import logger

    logger.info("Creating baseline random forest classifier using melody data!")
    # Get n-grams from both datasets
    logger.info("Extracting n-grams from training data...")
    temp_X, train_y = extract_ngrams_for_split("train")
    logger.info("Extracting n-grams from testing data...")
    test_X, test_y = extract_ngrams_for_split("test")
    # Get those n-grams that appear in at least N tracks in the training dataset
    logger.info(f"Extracting n-grams which appear in at least {rf_utils.VALID_NGRAMS} training tracks...")
    valid_ngs = get_valid_ngrams(temp_X)
    logger.info(f"... found {len(valid_ngs)} n-grams!")
    # Subset both datasets to ensure we only keep valid ngrams
    logger.info("Formatting features...")
    train_X = [{k: v for k, v in track_ngs.items() if k in valid_ngs} for track_ngs in temp_X]
    test_X = [{k: v for k, v in track_ngs.items() if k in valid_ngs} for track_ngs in test_X]
    # Convert both training and testing features into numpy arrays with the same dimensionality
    train_X_arr, test_X_arr = format_features(train_X, test_X)
    # Load the optimized parameter settings (or recreate these, if they don't exist)
    logger.info(f"Beginning parameter optimization with {rf_utils.N_ITER} iterations...")
    optimized_params = optimize_forest(train_X_arr, train_y, test_X_arr, test_y)
    # Create the optimized random forest model
    clf_opt = RandomForestClassifier(**optimized_params, random_state=utils.SEED)
    clf_opt.fit(train_X_arr, train_y)
    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_X_arr)
    test_acc = accuracy_score(test_y, test_y_pred)
    logger.info(f"Test accuracy for melody: {test_acc:.2f}")
    # TODO: compute validation accuracy using optimized settings
