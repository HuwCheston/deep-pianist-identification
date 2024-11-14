#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create and optimize different white-box classifier types"""

import warnings
from multiprocessing import Manager, Process
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.whitebox.wb_utils import get_classifier_and_params

ACC_TOP_KS = [1, 2, 3, 5, 10]  # We'll print the model top-k accuracy at these levels


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
    logger.info(f"... test accuracy: {test_acc:.6f}")
    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x)
    valid_acc = accuracy_score(valid_y, valid_y_pred)
    logger.info(f"... validation accuracy: {valid_acc:.6f}")
    # Get the top-k accuracy
    valid_y_proba = clf_opt.predict_proba(valid_x)
    for k in ACC_TOP_KS:
        topk_acc = top_k_accuracy_score(valid_y, valid_y_proba, k=k)
        logger.info(f'... top-{k} validation accuracy: {topk_acc:.6f}')
    # Return the fitted classifier for e.g., permutation importance testing
    return clf_opt, valid_acc, optimized_params


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
    if classifier_type == 'rf':
        try:
            best_params["max_features"] = float(best_params["max_features"])
        except ValueError:
            pass
    logger.info(f'... best parameters: {best_params}')
    return {k: v for k, v in best_params.items() if k not in ["accuracy", "iteration", "time"]}
