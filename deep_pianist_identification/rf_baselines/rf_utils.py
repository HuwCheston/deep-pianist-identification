#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions and variables for white-box models"""

import csv
import json
import os
import warnings
from ast import literal_eval
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from multiprocessing import Manager, Process
from operator import itemgetter
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from pretty_midi import PrettyMIDI, Instrument, Note
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import RollExtractor, quantize, ExtractorError
from deep_pianist_identification.plotting import HeatmapWhiteboxFeaturePianoRoll

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
K = 5  # When plotting, we'll get this number of features per performer (bottom and top)

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
        n_iter: int = 100,
        use_cache: bool = True,
        classifier_type: str = "rf"
) -> dict:
    """With given training and test features/targets, optimize random forest for test accuracy and save CSV to `out`"""

    def save_to_file(q):
        """Threadsafe writing to file"""
        while True:
            val = q.get()
            if val is None:
                break
            with open(out, 'a') as o:
                o.write(val + '\n')

    def load_from_file() -> list[dict]:
        """Tries to load cached results and returns a list of dictionaries"""
        try:
            cr = pd.read_csv(out).to_dict(orient='records')
            logger.info(f'... loaded {len(cr)} results from {out}')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            cr = []
            q.put('accuracy,iteration,time,' + ','.join(i for i in next(iter(sampler)).keys()))
        return cr

    def __step(iteration: int, parameters: dict) -> dict:
        """Inner optimization function"""
        start = time()
        # Try and load in the CSV file for this particular task
        if use_cache:
            try:
                # If we have it, just return the cached results for this particular iteration
                res = [o for o in cached_results if o['iteration'] == iteration][0]
                return res
            # Don't worry if we haven't got the CSV file or results for this interation yet
            except IndexError:
                pass
        # Create the forest model with the given parameter settings
        forest = classifier(**parameters)
        # Fit the model to the training data and get the test data accuracy
        forest.fit(train_features, train_targets)
        acc = accuracy_score(test_targets, forest.predict(test_features))
        # Create the results dictionary and save
        end = time() - start
        # line = str(acc) + ',' + str(iteration) + ',' + str(end) + ',' + ','.join(str(i) for i in parameters.values())
        results_dict = {'accuracy': acc, 'iteration': iteration, 'time': end, **parameters}
        # threadsafe_save_csv(results_dict, out)
        q.put(','.join(str(i) for i in results_dict.values()))
        # Return the results for this optimization step
        return results_dict

    classifier, params = get_classifier_and_params(classifier_type)
    # Create the parameter sampling instance with the required parameters, number of iterations, and random state
    sampler = ParameterSampler(params, n_iter=n_iter, random_state=utils.SEED)
    # Define multiprocessing instance used to write in threadsafe manner
    m = Manager()
    q = m.Queue()
    p = Process(target=save_to_file, args=(q,))
    p.start()
    # Get cached results
    cached_results = load_from_file()
    # Use lazy parallelization to create the forest and fit to the data
    warnings.filterwarnings("ignore")
    with Parallel(n_jobs=-1, verbose=5) as p:
        fit = p(
            delayed(__step)(num, params) for num, params in tqdm(enumerate(sampler), total=n_iter, desc="Fitting...")
        )
    warnings.filterwarnings("default")
    # Adding a None at this point kills the multiprocessing manager instance
    q.put(None)
    # Get the best parameter combination using pandas
    best_params = (
        pd.DataFrame(fit)
        .replace({np.nan: None})
        .convert_dtypes()  # Needed to convert some int columns from float
        .sort_values(by=['accuracy', 'iteration'], ascending=False)
        .head(1)
        .to_dict(orient='records')[0]
    )
    # TODO: maybe this needs to become a function?
    if classifier_type == 'rf':
        try:
            best_params["max_features"] = float(best_params["max_features"])
        except ValueError:
            pass
    logger.info(f'... best parameters: {best_params}')
    return {k: v for k, v in best_params.items() if k not in ["accuracy", "iteration", "time"]}


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


def get_harmony_feature_importance(
        harmony_features: np.ndarray,
        melody_features: np.ndarray,
        classifier,
        y_actual: np.ndarray,
        initial_acc: float,
        scale: bool = True,
        n_iter: int = N_ITER
):
    """Calculates harmony feature importance, both for all features and with bootstrapped subsamples of features"""
    scaler = StandardScaler()

    def _shuffler(idxs: np.ndarray = None, seed: int = utils.SEED):
        rng = np.random.default_rng(seed=seed)  # Used to permute arrays
        # Make a copy of the data and shuffle
        toshuffle = deepcopy(harmony_features)
        # If we're using all features, i.e., we're not bootstrapping
        if idxs is None:
            shuffled = rng.permuted(toshuffle, axis=0)
            # Combine the arrays together as before
            combined = np.concatenate([melody_features, shuffled], axis=1)
        # If we're only using some features, i.e., we're bootstrapping
        else:
            cols = toshuffle[:, idxs]
            # Permute the column
            permuted = rng.permutation(cols, axis=0)
            # Set the column back
            toshuffle[:, idxs] = permuted
            # Combine the arrays together as before
            combined = np.concatenate([melody_features, toshuffle], axis=1)
        # Z-transform the features if required
        if scale:
            combined = scaler.fit_transform(combined)
        # Make predictions and take accuracy
        valid_y_pred_permute = classifier.predict(combined)
        return initial_acc - accuracy_score(y_actual, valid_y_pred_permute)

    def _shuffler_boot(seed):
        # Get an array of bootstrapping indexes to use
        boot_idxs = np.random.choice(harmony_features.shape[1], N_BOOT_FEATURES)
        return _shuffler(boot_idxs, seed)

    with Parallel(n_jobs=-1, verbose=5) as par:
        # Permute all features
        logger.info('Permuting harmony features...')
        har_acc = par(delayed(_shuffler)(None, seed) for seed in range(n_iter))
        logger.info(f"... all harmony feature importance {np.mean(har_acc)}, "
                    f"SD {np.std(har_acc)}, "
                    f"CI: [{np.percentile(har_acc, 2.5)}, {np.percentile(har_acc, 97.5)}]")
        # Permute bootstrapped subsamples of features
        logger.info('Permuting harmony features with bootstrapping...')
        har_acc_boot = par(delayed(_shuffler_boot)(seed) for seed in range(n_iter))
        logger.info(f"... bootstrapped harmony feature importance {np.mean(har_acc_boot)}, "
                    f"SD {np.std(har_acc_boot)}, "
                    f"CI: [{np.percentile(har_acc_boot, 2.5)}, {np.percentile(har_acc_boot, 97.5)}]")


def get_melody_feature_importance(
        harmony_features: np.ndarray,
        melody_features: np.ndarray,
        classifier,
        y_actual: np.ndarray,
        initial_acc: float,
        scale: bool = True,
        n_iter: int = N_ITER
):
    """Calculates melody feature importance, both for all features and with bootstrapped subsamples of features"""
    scaler = StandardScaler()

    def _shuffler(idxs: np.ndarray = None, seed: int = utils.SEED):
        rng = np.random.default_rng(seed=seed)
        # Make a copy of the data and shuffle
        toshuffle = deepcopy(melody_features)
        # If we're using all features, i.e., we're not bootstrapping
        if idxs is None:
            shuffled = rng.permuted(toshuffle, axis=0)
            # Combine the arrays together as before
            combined = np.concatenate([shuffled, harmony_features], axis=1)
        # If we're only using some features, i.e., we're bootstrapping
        else:
            cols = toshuffle[:, idxs]
            # Permute the column
            permuted = rng.permutation(cols, axis=0)
            # Set the column back
            toshuffle[:, idxs] = permuted
            # Combine the arrays together as before
            combined = np.concatenate([toshuffle, harmony_features], axis=1)
        # Z-transform the features if required
        if scale:
            combined = scaler.fit_transform(combined)
        # Make predictions and take accuracy
        valid_y_pred_permute = classifier.predict(combined)
        return initial_acc - accuracy_score(y_actual, valid_y_pred_permute)

    def _shuffler_boot(seed):
        # Get an array of bootstrapping indexes to use
        boot_idxs = np.random.choice(melody_features.shape[1], N_BOOT_FEATURES)
        return _shuffler(boot_idxs, seed=seed)

    with Parallel(n_jobs=-1, verbose=5) as par:
        # Permute all features
        logger.info('Permuting melody features...')
        mel_acc = par(delayed(_shuffler)(None, seed) for seed in range(n_iter))
        logger.info(f"... all melody feature importance {np.mean(mel_acc)}, "
                    f"SD {np.std(mel_acc)}, "
                    f"CI: [{np.percentile(mel_acc, 2.5)}, {np.percentile(mel_acc, 97.5)}]")
        # Permute bootstrapped subsamples of features
        logger.info('Permuting melody features with bootstrapping...')
        mel_acc_boot = par(delayed(_shuffler_boot)(seed) for seed in range(n_iter))
        logger.info(f"... bootstrapped melody feature importance {np.mean(mel_acc_boot)}, "
                    f"SD {np.std(mel_acc_boot)}, "
                    f"CI: [{np.percentile(mel_acc_boot, 2.5)}, {np.percentile(mel_acc_boot, 97.5)}]")


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


def scale_feature(
        feature: np.array
) -> np.array:
    scaler = StandardScaler()
    return scaler.fit_transform(feature)


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
) -> tuple:
    """Runs optimization, fits optimized settings to validation set, and returns accuracy + fitted model"""
    logger.info(f"Beginning parameter optimization with {n_iter} iterations and classifier {classifier_type}...")
    logger.info(f"... optimization results will be saved in {csvpath}")
    logger.info(f"... shape of training features: {train_x.shape}")
    logger.info(f"... shape of testing features: {test_x.shape}")
    # Run the hyperparameter optimization here and get the optimized parameter settings
    optimized_params = _optimize_classifier(
        train_x, train_y, test_x, test_y, csvpath, n_iter=n_iter, classifier_type=classifier_type
    )
    # Create the optimized random forest model
    logger.info("Optimization finished, fitting optimized model to test and validation set...")
    classifier, _ = get_classifier_and_params(classifier_type)
    clf_opt = classifier(**optimized_params)
    clf_opt.fit(train_x, train_y)
    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_x)
    test_acc = accuracy_score(test_y, test_y_pred)
    logger.info(f"... test accuracy: {test_acc:.3f}")
    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x)
    valid_acc = accuracy_score(valid_y, valid_y_pred)
    logger.info(f"... validation accuracy: {valid_acc:.3f}")
    # Return the fitted classifier for e.g., permutation importance testing
    return clf_opt, valid_acc, optimized_params


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


def get_topk_features(
        weights: np.array,
        bootstrapped_weights: list[np.array],
        feature_names: np.array,
        class_mapping: dict,
        k: int = K,
) -> dict:
    """For weights (with bootstrapping), get top and bottom `k` features for each performer"""
    res = {}
    # Iterate over all classes
    for class_idx in tqdm(range(len(class_mapping.keys()))):
        # Get the actual and bootstrapped weights for this performer
        class_ratios = weights[class_idx, :]
        boot_class_ratios = np.array([b[class_idx, :] for b in bootstrapped_weights])
        # Apply functions to the bootstrapped weights to get SD, lower, upper bounds
        stds = np.apply_along_axis(np.std, 0, boot_class_ratios)
        lower_bounds = np.apply_along_axis(lambda x: np.percentile(x, 2.5), 0, boot_class_ratios)
        upper_bounds = np.apply_along_axis(lambda x: np.percentile(x, 97.5), 0, boot_class_ratios)
        # Get sorted idxs in ascending/descending order
        bottom_idxs = np.argsort(class_ratios)
        top_idxs = bottom_idxs[::-1]
        # Iterate through all concepts
        perf_dict = {}
        for concept, starter in zip(['melody', 'harmony'], ['M_', 'H_']):
            # Get top idxs
            top_m_idxs = np.array([i for i in top_idxs if feature_names[i].startswith(starter)])[:k]
            top_m_features = feature_names[top_m_idxs]
            top_m_ors = class_ratios[top_m_idxs]
            top_m_high = upper_bounds[top_m_idxs] - top_m_ors
            top_m_low = top_m_ors - lower_bounds[top_m_idxs]
            top_m_stds = stds[top_m_idxs]
            # Get bottom idxs
            bottom_m_idxs = np.array([i for i in bottom_idxs if feature_names[i].startswith(starter)])[:k]
            bottom_m_features = feature_names[bottom_m_idxs]
            bottom_m_ors = class_ratios[bottom_m_idxs]
            bottom_m_high = upper_bounds[bottom_m_idxs] - bottom_m_ors
            bottom_m_low = bottom_m_ors - lower_bounds[bottom_m_idxs]
            bottom_m_stds = stds[bottom_m_idxs]
            # Format results as a dictionary
            r = dict(
                top_ors=top_m_ors,
                top_names=top_m_features,
                top_high=top_m_high,
                top_low=top_m_low,
                top_std=top_m_stds,
                bottom_ors=bottom_m_ors,
                bottom_names=bottom_m_features,
                bottom_high=bottom_m_high,
                bottom_low=bottom_m_low,
                bottom_std=bottom_m_stds,
            )
            # Append to the performer dictionary
            perf_dict[concept] = r
        # Append to the master dictionary
        res[class_mapping[class_idx]] = perf_dict
    return res


def get_classifier_weights(
        all_x: np.array,
        all_y: np.array,
        feature_names: np.array,
        classifier_params: dict,
        classifier_type: str,
        n_boot: int = N_ITER // 10,  # this will take ages otherwise, so use a smaller number of iterations
        use_odds_ratios: bool = True
) -> tuple:
    """For a given classifier type, get actual weights and bootstrapped versions of these"""

    # Get the correct classifier instance (we don't care about parameters, though)
    classifier, _ = get_classifier_and_params(classifier_type)
    # Quick checks here to make sure that all parameters are as expected
    if classifier_type not in ["lr", "svm"]:
        raise ValueError(f"Getting classifier weights only supports `lr` and `svm` models, but got {classifier_type}!")
    if use_odds_ratios and classifier_type != "lr":
        raise ValueError(f"Getting weights as odds ratios is only supported when `classifier_type == 'lr'`")

    def fitter(x: np.array, y: np.array) -> np.array:
        # Create and fit the classifier and take the weights
        temp_model = classifier(**classifier_params)
        temp_model.fit(x, y)
        coef = temp_model.coef_
        # Take exponential if we want odds ratios
        if use_odds_ratios:
            coef = np.exp(coef)
        return coef

    def booter(x: np.array, y: np.array, i: int = utils.SEED) -> np.array:
        # Sample the features
        x_samp = x.sample(frac=1, replace=True, random_state=i)
        # Use the index to sample the y values
        y_samp = y[x_samp.index]
        # Return the coefficients
        return fitter(x_samp, y_samp)

    # Convert to a dataframe as this makes bootstrapping a lot easier
    full_x_df = pd.DataFrame(all_x, columns=feature_names)
    full_y_df = pd.Series(all_y)
    # Get actual coefficients: (n_classes, n_features)
    ratios = fitter(full_x_df, full_y_df)
    # Bootstrap the coefficients: (n_boots, n_classes, n_features)
    with Parallel(n_jobs=-1, verbose=5) as par:
        boot_ratios = par(delayed(booter)(full_x_df, full_y_df, i) for i in range(n_boot))
    return ratios, boot_ratios


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


class MIDIOutputManager:
    """Takes in a single MIDI clip and performer n-grams, and creates outputs that match every n-gram"""

    def __init__(
            self,
            midi_path: str,
            performer_name: str,
            desired_ngram: str
    ):
        self.midi_path = midi_path
        self.input_midi = PrettyMIDI(midi_path)
        self.performer_name = performer_name
        self.desired_ngram = desired_ngram

    @staticmethod
    def apply_skyline(quantized_notes):
        """For all notes with the same (quantized) start time, keep only the highest pitch note"""
        note_starts = itemgetter(0)
        # Group all the MIDI events by the (quantized) start time
        grouper = groupby(sorted(quantized_notes, key=note_starts), note_starts)
        # Iterate through each group and keep only the note with the highest pitch
        for quantized_note_start, subiter in grouper:
            yield max(subiter, key=itemgetter(2))

    @staticmethod
    def quantize_notes(input_midi: PrettyMIDI):
        quantized_notes = []
        # Iterate through all notes
        for note in input_midi.instruments[0].notes:
            note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
            # If the note is above our lower bound
            if note_pitch >= utils.MIDI_OFFSET:
                # Snap note start and end with the given time resolution
                new_note_start = quantize(note_start, 1 / utils.FPS)
                new_note_end = quantize(note_end, 1 / utils.FPS)
                # Set the velocity to maximum to binarize
                quantized_notes.append((new_note_start, new_note_end, note_pitch, note_velocity, note_start, note_end))
        return quantized_notes

    @staticmethod
    def convert_to_interval_classes(input_midi: list[tuple]):
        interval_classes = []
        for n1, n2 in zip(input_midi, input_midi[1:]):
            ic = n2[2] - n1[2]
            if abs(ic) > MAX_LEAP:
                continue
            # actual start, actual end, interval class
            interval_classes.append((n2[4], n2[5], ic))
        return interval_classes

    @staticmethod
    def get_desired_ngram_phrases(ng: list[int], interval_classes, span_notes: int = 2):
        ng_len = len(ng)
        matches, phrase_spans = [], []
        for ic_start_idx in range(len(interval_classes)):
            ic_end_idx = ic_start_idx + ng_len
            desired_phrase = interval_classes[ic_start_idx:ic_end_idx]
            all_ics = [i[2] for i in desired_phrase]
            if all_ics == ng:
                idx_before = ic_start_idx - span_notes
                idx_after = ic_end_idx + span_notes
                if idx_before < 0:
                    idx_before = 0
                if idx_after >= len(interval_classes):
                    idx_after = len(interval_classes) - 1
                note_before_phrase = interval_classes[idx_before]
                note_after_phrase = interval_classes[idx_after]
                phrase_span = note_before_phrase[0], note_after_phrase[1]
                phrase_spans.append(phrase_span)
        return phrase_spans

    def process_match(self, phrase_start: float, phrase_end: float) -> PrettyMIDI:
        # Get the notes from the *original* MIDI that are within the bounds
        midi_in_bounds = [n for n in self.input_midi.instruments[0].notes if phrase_start <= n.start <= phrase_end]
        # Get first onset and offset time
        earliest_onset = min(midi_in_bounds, key=lambda x: x.start).start
        # Adjust all note onset and offset times to fit within boundaries
        newmidi = PrettyMIDI()
        newinstr = Instrument(self.input_midi.instruments[0].program)
        newinstr.notes.extend(
            [Note(start=n.start - earliest_onset, end=n.end - earliest_onset, pitch=n.pitch, velocity=n.velocity) for
             n in midi_in_bounds])
        newmidi.instruments = [newinstr]
        return newmidi

    def create_outputs(self, processed_midi: PrettyMIDI):
        # Get the performer-ngram folder
        fp = os.path.join(
            utils.get_project_root(),
            'reports/figures/whitebox/clip_outputs',
            f'{self.performer_name.lower().replace(" ", "_")}_{self.desired_ngram[2:]}'
        )
        # Create if it doesn't already exist
        if not os.path.isdir(fp):
            os.makedirs(fp)
        # Create the folder to save outputs for this clip-ngram combination in
        n_already_saved = len(os.listdir(fp))
        newfp = os.path.join(fp, f'phrase_{str(n_already_saved).zfill(3)}')
        assert not os.path.isdir(newfp)
        os.makedirs(newfp)
        # Create the static matplotlib plot
        pl = HeatmapWhiteboxFeaturePianoRoll(
            processed_midi,
            self.performer_name,
            self.desired_ngram,
            os.path.sep.join(self.midi_path.split(os.path.sep)[-2:])
        )
        pl.create_plot()
        pl.save_fig(os.path.join(newfp, 'static.png'))

        # Write the MIDI
        processed_midi.write(os.path.join(newfp, 'midi.mid'))

    def run(self):
        # Validate the input MIDI, and skip any cases where e.g., we don't have notes
        try:
            validated = RollExtractor(self.input_midi).output_midi
        except ExtractorError:
            return
        quant = self.quantize_notes(validated)
        skylined = list(self.apply_skyline(quant))
        ics = self.convert_to_interval_classes(skylined)
        matched_phrases = self.get_desired_ngram_phrases(eval(self.desired_ngram[2:]), ics)
        # Skip over clip/ngram combinations where the ngram doesn't appear
        if len(matched_phrases) == 0:
            return
        # Otherwise, process each match to get the MIDI within the boundaries
        for phrase_start, phrase_end in matched_phrases:
            processed = self.process_match(phrase_start, phrase_end)
            self.create_outputs(processed)
