#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions and variables for random forest baselines"""

import csv
import json
import os
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler
from tenacity import retry, retry_if_exception_type

from deep_pianist_identification import utils

# These are the parameters we'll sample from when optimizing
OPTIMIZE_PARAMS = dict(
    # The number of trees to grow in the forest
    n_estimators=[i for i in range(10, 1001, 1)],
    # Max number of features considered for splitting a node
    max_features=[None, 'sqrt', 'log2'],
    # Max number of levels in each tree
    max_depth=[None, *[i for i in range(1, 101, 1)]],
    # Minimum number of samples required to split a node
    min_samples_split=[i for i in range(2, 11)],
    # Minimum number of samples required at each leaf node
    min_samples_leaf=[i for i in range(1, 11)],
    # Whether to sample data points with our without replacement
    bootstrap=[True, False],
)

N_ITER = 10000  # default number of iterations for optimization process
N_JOBS = -1  # number of parallel processing cpu cores. -1 means use all cores.


@retry(retry=retry_if_exception_type(json.JSONDecodeError))
def threadsafe_load_csv(csv_path: str) -> dict:
    """Simple wrapper around `csv.load` that catches errors when working on the same file in multiple threads"""

    def eval_(i):
        try:
            return literal_eval(i)
        except (ValueError, SyntaxError) as _:
            return str(i)

    with open(csv_path, "r+") as in_file:
        return [{k: eval_(v) for k, v in dict(row).items()} for row in csv.DictReader(in_file, skipinitialspace=True)]


def threadsafe_save_csv(obj: list | dict, fpath: str) -> None:
    """Simple wrapper around csv.DictWriter with protections to assist in multithreaded access"""
    # Get the base directory and filename from the provided path
    _path = Path(fpath)
    fdir, fpath = str(_path.parent), str(_path.name)
    # If we have an existing file with the same name, load it in and extend it with our new data
    try:
        existing_file = threadsafe_load_csv(os.path.join(fdir, fpath))
    except FileNotFoundError:
        pass
    else:
        if isinstance(obj, dict):
            obj = [obj]
        obj: list = existing_file + obj

    # Create a new temporary file, in append mode
    temp_file = NamedTemporaryFile(mode='a', newline='', dir=fdir, delete=False, suffix='.csv')
    # Get our CSV header from the keys of the first dictionary, if we've passed in a list of dictionaries
    if isinstance(obj, list):
        keys = obj[0].keys()
    # Otherwise, if we've just passed in a dictionary, get the keys from it directly
    else:
        keys = obj.keys()
    # Open the temporary file and create a new dictionary writer with our given columns
    with temp_file as out_file:
        dict_writer = csv.DictWriter(out_file, keys)
        dict_writer.writeheader()
        # Write all the rows, if we've passed in a list
        if isinstance(obj, list):
            dict_writer.writerows(obj)
        # Alternatively, write a single row, if we've passed in a dictionary
        else:
            dict_writer.writerow(obj)

    @retry(retry=retry_if_exception_type(PermissionError))
    def replacer():
        os.replace(temp_file.name, os.path.join(fdir, fpath))

    replacer()


def get_split_clips(split_name: str) -> tuple[str, int, str]:
    """For a given CSV, yields tuples of track path, N track clips, track target"""
    # Quick check to make sure the split name is valid
    assert split_name in ["test", "train", "validation"], "`split_name` must be one of 'test', 'train', or 'validation"
    # Load the dataframe
    split_path = os.path.join(utils.get_project_root(), 'references/data_splits', f'{split_name}_split.csv')
    test_clips = pd.read_csv(split_path, index_col=0, delimiter=',')
    # Iterate through rows (= tracks)
    for idx, track in test_clips.iterrows():
        # Get the location of the clip folder
        track_path = os.path.join(utils.get_project_root(), 'data/clips', track['track'])
        # Count the number of clips for this track
        track_clips = len(os.listdir(track_path))
        # Return a tuple for each track
        yield track_path, track_clips, track['pianist']


def extract_ngrams_from_clips(split_clips: Iterable, extract_func: Callable, *args) -> tuple[list[dict], list[int]]:
    """For clips from a given split, applies an n-gram `extract_func` in parallel"""
    # Iterate over each track and apply our extraction function
    with Parallel(n_jobs=N_JOBS, verbose=5) as par:
        res = par(delayed(extract_func)(p, n, pi, *args) for p, n, pi in list(split_clips))
    # Returns a tuple of ngrams (dict) and targets (int) for every *clip* in the dataset
    return zip(*res)


def get_valid_ngrams(list_of_ngram_dicts: list[dict], valid_count: int) -> list[str]:
    """From a list of n-gram dictionaries (one dictionary per track), get n-grams that appear in at least N tracks"""
    ngram_counts = defaultdict(int)
    # Iterate through each dictionary (= one track)
    for d in list_of_ngram_dicts:
        # Iterate through each key (= one n-gram in the track)
        for key in d:
            # Increase the global count for this n-gram
            ngram_counts[key] += 1
    # Return a list of n-grams that appear in at least N tracks
    return [key for key, count in ngram_counts.items() if count >= valid_count]


def drop_invalid_ngrams(list_of_ngram_dicts: list[dict], valid_ngrams: list[str]) -> list[dict]:
    """From a list of n-gram dictionaries (one dictionary per track), remove n-grams that do not appear in N tracks"""
    # Define the removal function
    rm = lambda feat: {k: v for k, v in feat.items() if k in valid_ngrams}
    # In parallel, remove non-valid n-grams from all dictionaries
    with Parallel(n_jobs=N_JOBS, verbose=5) as par:
        return par(delayed(rm)(f) for f in list_of_ngram_dicts)


def format_features(*features: list[dict]) -> tuple:
    """For an arbitrary number of ngram dictionaries, format these as NumPy arrays with the same number of columns"""
    fmt = []
    for split_count, feature in enumerate(features):
        # Convert features to a dataframe
        df = pd.DataFrame(feature)
        # Set ID column
        df['split'] = str(split_count)
        fmt.append(df)
    # Combine all dataframes row-wise, which ensures the correct number of columns in all, and fill NaNs with 0
    conc = pd.concat(fmt, axis=0).fillna(0)
    # Return a tuple of numpy arrays, one for each split initially provided
    return (conc[conc['split'] == str(sp)].drop(columns='split').to_numpy() for sp in range(len(features)))


def optimize_classifier(
        train_features: np.ndarray, train_targets: list, test_features: np.ndarray, test_targets: np.ndarray,
        out: str, classifier: Callable = RandomForestClassifier, n_iter: int = N_ITER, use_cache: bool = True
) -> dict:
    """With given training and test features/targets, optimize random forest for test accuracy and save CSV to `out`"""

    def _step(iteration: int, parameters: dict) -> dict:
        """Inner optimization function"""
        # Try and load in the CSV file for this particular task
        if use_cache:
            try:
                cached_results = threadsafe_load_csv(out)
                # If we have it, just return the cached results for this particular iteration
                return [o for o in cached_results if o['iteration'] == iteration][0]
            # Don't worry if we haven't got the CSV file or results for this interation yet
            except (FileNotFoundError, IndexError):
                pass
        # Create the forest model with the given parameter settings
        forest = classifier(**parameters, random_state=utils.SEED)
        # Fit the model to the training data and get the test data accuracy
        forest.fit(train_features, train_targets)
        acc = accuracy_score(test_targets, forest.predict(test_features))
        # Create the results dictionary and save
        results_dict = {'accuracy': acc, 'iteration': iteration, **parameters}
        threadsafe_save_csv(results_dict, out)
        # Return the results for this optimization step
        return results_dict

    # Create the parameter sampling instance with the required parameters, number of iterations, and random state
    sampler = ParameterSampler(OPTIMIZE_PARAMS, n_iter=n_iter, random_state=utils.SEED)
    # Use lazy parallelization to create the forest and fit to the data
    with Parallel(n_jobs=1, verbose=5) as p:
        fit = p(delayed(_step)(num, params) for num, params in enumerate(sampler))
    # Get the parameter combination that yielded the best test set accuracy
    best_params = max(fit, key=lambda x: x["accuracy"])
    return {k: v for k, v in best_params.items() if k not in ["accuracy", "iteration"]}
