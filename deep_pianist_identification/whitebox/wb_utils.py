#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions and variables for white-box models"""

import csv
import json
import os
from ast import literal_eval
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt

from deep_pianist_identification import utils

# These are the parameters we'll sample from when optimizing different types of model
RF_OPTIMIZE_PARAMS = dict(
    # The number of trees to grow in the forest
    n_estimators=[i for i in range(10, 401, 1)],
    # Max number of features considered for splitting a node
    # max_features=[None, 'sqrt', 'log2'],
    max_features=['sqrt', 'log2', *np.linspace(1e-4, 1.0, 401)],
    # Max number of levels in each tree
    max_depth=[None, *[i for i in range(1, 41, 1)]],
    # Minimum number of samples required to split a node
    min_samples_split=[i for i in range(2, 11)],
    # Minimum number of samples required at each leaf node
    min_samples_leaf=[i for i in range(1, 11)],
    # Whether to sample data points with our without replacement
    bootstrap=[True, False],
    random_state=[utils.SEED]
)
XGB_OPTIMIZE_PARAMS = dict(
    # 1 = no shrinkage
    learning_rate=np.linspace(1e-3, 1., 401),
    # Will grow max_iter * num_classes trees per iteration
    max_iter=[i for i in range(1, 101, 1)],
    # Number of leaves for each tree; if None, there is no upper limit
    max_leaf_nodes=[None, *[i for i in range(2, 51, 1)]],
    # Max number of levels in each tree
    max_depth=[None, *range(1, 51, 1)],
    # Minimum number of samples required at each leaf node
    min_samples_leaf=[i for i in range(1, 11)],
    # Max number of features considered for splitting a node
    max_features=np.linspace(1e-4, 1., 401),
    random_state=[utils.SEED]
)
SVM_OPTIMIZE_PARAMS = dict(
    C=np.logspace(-3, 3, num=5001),
    kernel=["linear"],
    shrinking=[True, False],
    class_weight=[None, 'balanced'],
    decision_function_shape=['ovr'],
    max_iter=[10000],
    random_state=[utils.SEED]
)
LR_OPTIMIZE_PARAMS = dict(
    C=np.logspace(-3, 3, num=5001),
    penalty=[None, 'l2'],
    class_weight=[None, 'balanced'],
    solver=["lbfgs"],
    random_state=[utils.SEED],
    max_iter=[10000]
)
NB_OPTIMIZE_PARAMS = dict(
    alpha=np.logspace(-3, 3, num=5001),
    fit_prior=[True, False]
)
GNB_OPTIMIZE_PARAMS = dict(
    var_smoothing=np.logspace(-9, 0, num=10001)
)

N_ITER = 10000  # default number of iterations for optimization process
N_JOBS = -1  # number of parallel processing cpu cores. -1 means use all cores.
N_BOOT_FEATURES = 2000  # number of features to sample when bootstrapping permutation importance scores

MIN_COUNT = 10
MAX_COUNT = 1000

MAX_LEAP = 15  # If we leap by more than this number of semitones (once in an n-gram, twice in a chord), drop it
NGRAMS = [2, 3]  # The ngrams to consider when extracting melody (horizontal) or chords (vertical)


@retry(
    retry=retry_if_exception_type(json.JSONDecodeError),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(20)
)
def threadsafe_load_csv(csv_path: str) -> list[dict]:
    """Simple wrapper around `csv.load` that catches errors when working on the same file in multiple threads"""

    def eval_(i):
        try:
            return literal_eval(i)
        except (ValueError, SyntaxError) as _:
            return str(i)

    with open(csv_path, "r+") as in_file:
        return [{k: eval_(v) for k, v in dict(row).items()} for row in csv.DictReader(in_file, skipinitialspace=True)]


def threadsafe_save_csv(obje: list | dict, filepath: str) -> None:
    """Simple wrapper around csv.DictWriter with protections to assist in multithreaded access"""
    @retry(
        retry=retry_if_exception_type(PermissionError),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(20)
    )
    def replacer(obj: list | dict, fpath: str):
        # Get the base directory and filename from the provided path
        _path = Path(fpath)
        fdir, fpath = str(_path.parent), str(_path.name)
        # If we have an existing file with the same name, load it in and extend it with our new data
        try:
            existing_file: list = threadsafe_load_csv(os.path.join(fdir, fpath))
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
        # Try to replace the master file with the temporary file
        try:
            os.replace(temp_file.name, os.path.join(fdir, fpath))
        # If another file is already editing the master file
        except PermissionError as e:
            # Remove the temporary file
            os.remove(temp_file.name)
            # Reraise the exception, which tenacity will catch and retry saving with exponential backoff
            raise e

    replacer(obje, filepath)


def get_split_clips(split_name: str, dataset: str = "20class_80min") -> tuple[str, int, str]:
    """For a given CSV and dataset, yields tuples of track path, N track clips, track target"""
    # Quick check to make sure the split name is valid
    assert split_name in ["test", "train", "validation"], "`split_name` must be one of 'test', 'train', or 'validation"
    # Load the dataframe
    split_path = os.path.join(utils.get_project_root(), 'references/data_splits', dataset, f'{split_name}_split.csv')
    test_clips = pd.read_csv(split_path, index_col=0, delimiter=',')
    # Iterate through rows (= tracks)
    for idx, track in test_clips.iterrows():
        # Get the location of the clip folder
        track_path = os.path.join(utils.get_project_root(), 'data/clips', track['track'])
        # Count the number of clips for this track
        track_clips = len(os.listdir(track_path))
        # Return a tuple for each track
        yield track_path, track_clips, track['pianist']


def get_classifier_and_params(classifier_type: str) -> tuple:
    """Return the correct classifier instance and optimized parameter settings"""
    clf = ["rf", "svm", "nb", "lr", "xgb", "gnb"]
    assert classifier_type in clf, "`classifier_type` must be one of " + ", ".join(clf) + f" but got {classifier_type}"
    if classifier_type == "rf":
        return RandomForestClassifier, RF_OPTIMIZE_PARAMS
    elif classifier_type == "svm":
        return SVC, SVM_OPTIMIZE_PARAMS
    elif classifier_type == "nb":
        return MultinomialNB, NB_OPTIMIZE_PARAMS
    elif classifier_type == "lr":
        return LogisticRegression, LR_OPTIMIZE_PARAMS
    elif classifier_type == "xgb":
        return HistGradientBoostingClassifier, XGB_OPTIMIZE_PARAMS
    elif classifier_type == "gnb":
        return GaussianNB, GNB_OPTIMIZE_PARAMS


def scale_features(
        train_x: np.ndarray,
        test_x: np.ndarray,
        valid_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale the data using Z-transformation"""
    logger.info('... data will be scaled using z-transformation!')
    # Create the scaler object
    scaler = StandardScaler()
    # Fit to the training data and use these values to transform others
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    valid_x = scaler.transform(valid_x)
    return train_x, test_x, valid_x


def get_all_clips(dataset: str, n_clips: int = None):
    # Get clips from all splits of the dataset
    logger.info(f"Getting tracks from all splits for dataset {dataset}...")
    train_clips = list(get_split_clips("train", dataset))
    test_clips = list(get_split_clips("test", dataset))
    validation_clips = list(get_split_clips("validation", dataset))
    # Truncate the number of clips if required
    if n_clips is not None and isinstance(n_clips, int):
        train_clips = train_clips[:n_clips]
        test_clips = test_clips[:n_clips]
        validation_clips = validation_clips[:n_clips]
    logger.info(f"... found {len(train_clips)} train tracks, "
                f"{len(test_clips)} test tracks, "
                f"{len(validation_clips)} validation tracks!")
    return train_clips, test_clips, validation_clips


def parse_arguments(parser) -> dict:
    """Takes in a parser object and adds all required arguments, then returns these arguments"""
    parser.add_argument(
        "-d", "--dataset",
        default="20class_80min",
        type=str,
        help="Name of dataset inside `references/data_split`"
    )
    parser.add_argument(
        "-i", "--n-iter",
        default=N_ITER,
        type=int,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "-r", "--remove-leaps",
        default=True,
        type=bool,
        help="Remove leaps of more than +/- 15 semitones",
    )
    parser.add_argument(
        '-l', '--ngrams',
        nargs='+',
        type=int,
        help="Extract these n-grams",
        default=NGRAMS
    )
    parser.add_argument(
        "-s", "--min-count",
        default=MIN_COUNT,
        type=int,
        help="Minimum number of tracks an n-gram can appear in"
    )
    parser.add_argument(
        "-b", "--max-count",
        default=MAX_COUNT,
        type=int,
        help="Maximum number of tracks an n-gram can appear in"
    )
    parser.add_argument(
        "-c", "--classifier-type",
        default="rf",
        type=str,
        help="Classifier type to use. "
             "Either 'rf' (random forest)', 'svm' (support vector machine), "
             "'nb' (naive Bayes), or 'lr' (logistic regression)"
    )
    parser.add_argument(
        "-z", "--scale",
        default=True,
        type=bool,
        help="Whether to scale data using z-transformation or not"
    )
    parser.add_argument(
        "-k", "--database-k-coefs",
        default=1000,
        type=int,
        help="Number of k coefficients to extract when computing correlation between different datasets."
             "-1 uses all coefficients."
    )
    # Parse all arguments and return
    args = vars(parser.parse_args())
    if args['database_k_coefs'] == -1:
        args['database_k_coefs'] = None
    return args


def get_database_mapping(
        train_clips: list[tuple],
        test_clips: list[tuple],
        valid_clips: list[tuple]
) -> np.array:
    """Creates an array with 1 for recordings in JTD and 0 for PiJAMA"""
    # Create arrays for each individual clip
    train_datasets = np.array([1 if 'jtd' in i[0] else 0 for i in train_clips])
    test_datasets = np.array([1 if 'jtd' in i[0] else 0 for i in test_clips])
    valid_datasets = np.array([1 if 'jtd' in i[0] else 0 for i in valid_clips])
    # Concatenate and return
    return np.hstack([train_datasets, test_datasets, valid_datasets])
