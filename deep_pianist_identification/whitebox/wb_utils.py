#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions and variables for white-box models"""

import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from deep_pianist_identification import utils
from deep_pianist_identification.whitebox.features import FEATURE_MIN_TRACKS, FEATURE_MAX_TRACKS, DEFAULT_FEATURE_SIZES

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

N_ITER = 1000  # default number of iterations for optimization process
N_JOBS = -1  # number of parallel processing cpu cores. -1 means use all cores.
N_BOOT_FEATURES = 2000  # number of features to sample when bootstrapping permutation importance scores


class TfidfFixed(TfidfTransformer):
    """Wrapper around TfidfTransformer that implements a proper .transform method allowing use in a pipeline"""

    def __init__(self, *, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False):
        super().__init__(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    def fit_transform(self, X, y=None, copy=True):
        self.fit(X, y)
        return self.transform(X, copy)

    def transform(self, X, copy=True):
        # This is the change, otherwise we get a sparse matrix
        return super().transform(X, copy).toarray()


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


def scale_for_pca(
        feature_counts: np.ndarray
) -> np.ndarray:
    pipe = Pipeline([
        ('tfidf', TfidfFixed(norm='l2')),
        ('zscore', StandardScaler()),
        ('scale', MinMaxScaler()),
    ])
    return pipe.fit_transform(feature_counts)


def scale_features(
        train_x: np.ndarray,
        test_x: np.ndarray,
        valid_x: np.ndarray,
        method: str = "tf-idf"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale the data using Z-transformation"""
    if method == "z-score":
        logger.info('... data will be scaled using z-transformation!')
        # Create the scaler object
        scaler = StandardScaler()
        # Fit to the training data and use these values to transform others
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        valid_x = scaler.transform(valid_x)
        return train_x, test_x, valid_x
    elif method == "tf-idf":
        logger.info('... data will be scaled using tf-idf!')
        # Create the scaler object
        scaler = TfidfTransformer(norm='l2')
        # Fit to the training data and use these values to transform others
        train_x = scaler.fit_transform(train_x).toarray()
        test_x = scaler.transform(test_x).toarray()
        valid_x = scaler.transform(valid_x).toarray()
        return train_x, test_x, valid_x
    else:
        raise ValueError(f'method should be either `tf-idf` or `z-score`, but got {method}!')


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
        help="Name of dataset inside `references/data_split`, default is '20class_80min'."
    )
    parser.add_argument(
        "-i", "--n-iter",
        default=N_ITER,
        type=int,
        help="Number of optimization iterations, default is 10000."
    )
    parser.add_argument(
        '-l', '--feature-sizes',
        nargs='+',
        type=int,
        help="Extract these n-grams and harmony features, default is [2, 3] (i.e., triads and tetrads for harmony)",
        default=DEFAULT_FEATURE_SIZES
    )
    parser.add_argument(
        "-s", "--min-count",
        default=FEATURE_MIN_TRACKS,
        type=int,
        help="Minimum number of tracks an n-gram can appear in, default is 10."
    )
    parser.add_argument(
        "-b", "--max-count",
        default=FEATURE_MAX_TRACKS,
        type=int,
        help="Maximum number of tracks an n-gram can appear in, default is 1000."
    )
    parser.add_argument(
        "-c", "--classifier-type",
        default="lr",
        type=str,
        help="Classifier type to use when fitting model. "
             "Either 'rf' (random forest)', 'svm' (support vector machine), or 'lr' (logistic regression, default)"
    )
    parser.add_argument(
        "-z", "--scale",
        default=True,
        type=utils.string_to_bool,
        help="Whether to transform data using or not, default is True."
    )
    parser.add_argument(
        "-o", "--optimize",
        default=False,
        type=utils.string_to_bool,
        help="Whether to formally optimize model with optuna, default is False (random parameter sampling instead)."
    )
    parser.add_argument(
        "-k", "--database-k-coefs",
        default=500,
        type=int,
        help="Number of features to use when computing correlation between different datasets, default is 500. "
             "Either int or float is accepted: float is interpreted as fraction of total features. "
             "-1 is interpreted as using all available features."
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
