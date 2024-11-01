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
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

from deep_pianist_identification import utils

# These are the parameters we'll sample from when optimizing different types of model
RF_OPTIMIZE_PARAMS = dict(
    # The number of trees to grow in the forest
    n_estimators=[i for i in range(10, 401, 1)],
    # Max number of features considered for splitting a node
    # max_features=[None, 'sqrt', 'log2'],
    max_features=['sqrt', 'log2', 1/1000, 1/100, 1/50, 1/10],
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
    learning_rate=np.linspace(0, 1., 401),
    # Will grow max_iter * num_classes trees per iteration
    max_iter=[i for i in range(10, 401, 1)],
    # Number of leaves for each tree; if None, there is no upper limit
    max_leaf_nodes=[None, *[i for i in range(1, 101, 1)]],
    # Max number of levels in each tree
    max_depth=[None, 1 / 1000, 1 / 100, 1 / 50, 1 / 10],
    # Minimum number of samples required at each leaf node
    min_samples_leaf=[i for i in range(1, 11)],
    # Max number of features considered for splitting a node
    max_features=np.linspace(0, 1., 401),
    random_state=[utils.SEED]
)
SVM_OPTIMIZE_PARAMS = dict(
    C=np.logspace(-2, 2, num=401),
    penalty=['l2', 'l1'],
    class_weight=[None, 'balanced'],
    intercept_scaling=np.logspace(-2, 2, num=401),
    max_iter=[1000],
    random_state=[utils.SEED]
)
LR_OPTIMIZE_PARAMS = dict(
    C=np.logspace(-2, 2, num=401),
    penalty=[None, 'l2', 'l1'],
    class_weight=[None, 'balanced'],
    solver=["saga"],
    random_state=[utils.SEED]
)
NB_OPTIMIZE_PARAMS = dict(
    alpha=np.logspace(-2, 2, num=401),
    fit_prior=[True, False]
)

N_ITER = 10000  # default number of iterations for optimization process
N_JOBS = -1  # number of parallel processing cpu cores. -1 means use all cores.

MIN_COUNT = 10
MAX_COUNT = 1000

MAX_LEAP = 15  # If we leap by more than this number of semitones (once in a n-gram, twice in a chord), drop it
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


def threadsafe_save_csv(obj: list | dict, fpath: str) -> None:
    """Simple wrapper around csv.DictWriter with protections to assist in multithreaded access"""
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

    @retry(
        retry=retry_if_exception_type(PermissionError),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(20)
    )
    def replacer():
        os.replace(temp_file.name, os.path.join(fdir, fpath))

    replacer()


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


def extract_ngrams_from_clips(split_clips: Iterable, extract_func: Callable, *args) -> tuple[list[dict], list[int]]:
    """For clips from a given split, applies an n-gram `extract_func` in parallel"""
    # Iterate over each track and apply our extraction function
    with Parallel(n_jobs=N_JOBS, verbose=5) as par:
        res = par(delayed(extract_func)(p, n, pi, *args) for p, n, pi in list(split_clips))
    # Returns a tuple of ngrams (dict) and targets (int) for every *clip* in the dataset
    return zip(*res)


def get_valid_ngrams(list_of_ngram_dicts: list[dict], min_count: int, max_count: int = 10000) -> list[str]:
    """From a list of n-gram dictionaries (one dictionary per track), get n-grams that appear in at least N tracks"""
    ngram_counts = defaultdict(int)
    # Iterate through each dictionary (= one track)
    for d in list_of_ngram_dicts:
        # Iterate through each key (= one n-gram in the track)
        for key in d:
            # Increase the global count for this n-gram
            ngram_counts[key] += 1
    # Return a list of n-grams that appear in at least N tracks
    return [key for key, count in ngram_counts.items() if min_count <= count <= max_count]


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
    # Return a tuple of numpy arrays, one for each split initially provided, PLUS the names of the features themselves
    return (
        *(conc[conc['split'] == str(sp)].drop(columns='split').to_numpy() for sp in range(len(features))),
        conc.drop(columns='split').columns.tolist()
    )


def _optimize_classifier(
        train_features: np.ndarray,
        train_targets: list,
        test_features: np.ndarray,
        test_targets: np.ndarray,
        out: str,
        n_iter: int = N_ITER,
        use_cache: bool = True,
        classifier_type: str = "rf"
) -> dict:
    """With given training and test features/targets, optimize random forest for test accuracy and save CSV to `out`"""

    def __step(iteration: int, parameters: dict) -> dict:
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
        forest = classifier(**parameters)
        # Fit the model to the training data and get the test data accuracy
        forest.fit(train_features, train_targets)
        acc = accuracy_score(test_targets, forest.predict(test_features))
        # Create the results dictionary and save
        results_dict = {'accuracy': acc, 'iteration': iteration, **parameters}
        threadsafe_save_csv(results_dict, out)
        # Return the results for this optimization step
        return results_dict

    classifier, params = get_classifier_and_params(classifier_type)
    # Create the parameter sampling instance with the required parameters, number of iterations, and random state
    sampler = ParameterSampler(params, n_iter=n_iter, random_state=utils.SEED)
    # Use lazy parallelization to create the forest and fit to the data
    # fit = [_step(num, params) for num, params in tqdm(enumerate(sampler), total=n_iter, desc="Fitting...")]
    with Parallel(n_jobs=-1, verbose=5) as p:
        fit = p(
            delayed(__step)(num, params) for num, params in tqdm(enumerate(sampler), total=n_iter, desc="Fitting...")
        )
    # Get the parameter combination that yielded the best test set accuracy
    best_params = max(fit, key=lambda x: x["accuracy"])
    return {k: v for k, v in best_params.items() if k not in ["accuracy", "iteration"]}


def get_classifier_and_params(classifier_type: str) -> tuple:
    """Return the correct classifier instance and optimized parameter settings"""
    clf = ["rf", "svm", "nb", "lr", "xgb"]
    assert classifier_type in clf, "`classifier_type` must be one of " + ", ".join(clf) + f" but got {classifier_type}"
    if classifier_type == "rf":
        return RandomForestClassifier, RF_OPTIMIZE_PARAMS
    elif classifier_type == "svm":
        return LinearSVC, SVM_OPTIMIZE_PARAMS
    elif classifier_type == "nb":
        return MultinomialNB, NB_OPTIMIZE_PARAMS
    elif classifier_type == "lr":
        return LogisticRegression, LR_OPTIMIZE_PARAMS
    elif classifier_type == "xgb":
        return HistGradientBoostingClassifier, XGB_OPTIMIZE_PARAMS


def fit_classifier(
        train_x: np.ndarray,
        test_x: np.ndarray,
        valid_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
        valid_y: np.ndarray,
        csvpath: str,
        n_iter: int,
        classifier_type: str = "rf",
        scale: bool = True
) -> tuple[RandomForestClassifier, float]:
    """Runs optimization, fits optimized settings to validation set, and returns accuracy + fitted model"""
    logger.info(f"Beginning parameter optimization with {n_iter} iterations and classifier {classifier_type}...")
    logger.info(f"... optimization results will be saved in {csvpath}")
    logger.info(f"... shape of training features: {train_x.shape}")
    logger.info(f"... shape of testing features: {test_x.shape}")
    classifier, _ = get_classifier_and_params(classifier_type)
    optimized_params = _optimize_classifier(
        train_x, train_y, test_x, test_y, csvpath, n_iter=n_iter, classifier_type=classifier_type
    )
    # Scale the data if required (not for naive Bayes as data must be non-negative)
    if scale and classifier_type != "nb":
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        valid_x = scaler.transform(valid_x)
    # Create the optimized random forest model
    logger.info("Optimization finished, fitting optimized model to test and validation set...")
    clf_opt = classifier(**optimized_params)
    clf_opt.fit(train_x, train_y)

    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_x)
    test_acc = accuracy_score(test_y, test_y_pred)
    logger.info(f"... test accuracy: {test_acc:.3f}")
    # Compute top-k test accuracy
    # TODO: fix this
    # test_y_proba = clf_opt.predict_proba(test_x)
    # for k in [2, 5, 10]:
    #     test_k_acc = top_k_accuracy_score(test_y, test_y_proba, k=k)
    #     logger.info(f"... test accuracy @ k = {k}: {test_k_acc:.3f}")

    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x)
    valid_acc = accuracy_score(valid_y, valid_y_pred)
    logger.info(f"... validation accuracy: {valid_acc:.3f}")
    # Compute top-k validation accuracy
    # valid_y_proba = clf_opt.predict_proba(valid_x)
    # for k in [2, 5, 10]:
    #     valid_k_acc = top_k_accuracy_score(valid_y, valid_y_proba, k=k)
    #     logger.info(f"... validation accuracy @ k = {k}: {valid_k_acc:.3f}")
    # TODO: permutation importance can go here
    return clf_opt, valid_acc


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
    # Parse all arguments and return
    args = vars(parser.parse_args())
    return args
