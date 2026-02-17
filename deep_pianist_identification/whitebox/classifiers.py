#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create and optimize different white-box classifier types"""

import pickle
import warnings
from functools import wraps
from multiprocessing import Manager, Process
from time import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.whitebox.wb_utils import get_classifier_and_params, N_JOBS

ACC_TOP_KS = [1, 2, 3, 5, 10]  # We'll print the model top-k accuracy at these levels


def log_topk_acc(validation_x: np.ndarray, validation_y: np.ndarray, classifier_optimised) -> None:
    """Tries to log top-k accuracy at various values of k for models that support the `.predict_proba` function"""
    try:
        valid_y_proba = classifier_optimised.predict_proba(validation_x)
    except AttributeError:
        logger.warning(f"... couldn't compute top-k accuracy for this classifier type!")
    else:
        for k in ACC_TOP_KS:
            topk_acc = top_k_accuracy_score(validation_y, valid_y_proba, k=k)
            logger.info(f'... top-{k} validation accuracy: {topk_acc:.6f}')


def save_classifier(outpath: str, classifier) -> None:
    """Saves a fitted sklearn classifier as a pickle file"""
    with open(outpath, "wb") as f:
        pickle.dump(classifier, f, protocol=5)  # protocol=5 recommended in sklearn documentation
    logger.info(f"... classifier instance dumped to {outpath}")


def ignore_user_warning(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            out = func(*args, **kwargs)
        finally:
            warnings.filterwarnings("default", category=UserWarning)
        return out
    return wrapper


@ignore_user_warning
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
    logger.info("Optimization finished!")
    classifier, _ = get_classifier_and_params(classifier_type)
    clf_opt = classifier(**optimized_params)
    clf_opt.fit(train_x, train_y)
    # Dump the classifier
    # save_classifier(csvpath.replace("csv", "p"), clf_opt)
    # Get the optimized test accuracy
    test_y_pred = clf_opt.predict(test_x)
    test_acc = accuracy_score(test_y, test_y_pred)
    logger.info(f"... test accuracy: {test_acc:.6f}")
    # Get the optimized validation accuracy
    valid_y_pred = clf_opt.predict(valid_x)
    valid_acc = accuracy_score(valid_y, valid_y_pred)
    logger.info(f"... validation accuracy: {valid_acc:.6f}")
    # Get the top-k accuracy if we can
    log_topk_acc(valid_x, valid_y, clf_opt)
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
        warnings.filterwarnings("ignore", category=UserWarning)
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
        results_dict = {
            'accuracy': acc,
            'iteration': iteration,
            'time': end,
            # 'clock_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **parameters
        }
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
    with Parallel(n_jobs=N_JOBS, verbose=5) as p:
        fit = p(
            delayed(__step)(num, params) for num, params in tqdm(enumerate(sampler), total=n_iter, desc="Fitting...")
        )
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


@ignore_user_warning
def fit_with_optimization(
    train_x: np.ndarray,
    test_x: np.ndarray,
    valid_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    valid_y: np.ndarray,
    csvpath: str,
    n_iter: int,
    classifier_type: str = "lr",
):
    classifier, dummy_params = get_classifier_and_params(classifier_type)

    def objective(trial):
        if classifier_type == "lr":
            penalty = trial.suggest_categorical("penalty", dummy_params["penalty"])
            class_weight = trial.suggest_categorical("class_weight", dummy_params["class_weight"])
            c = trial.suggest_float("C", min(dummy_params["C"]), max(dummy_params["C"]), log=True)

            # penalty=None makes C irrelevant — prune redundant trials
            if penalty is None:
                c = 1.0

            params = dict(
                C=c,
                penalty=penalty,
                class_weight=class_weight,
                solver=dummy_params["solver"][0],
                random_state=dummy_params["random_state"][0],
                max_iter=dummy_params["max_iter"][0],
            )

        elif classifier_type == "svm":
            c = trial.suggest_float("C", min(dummy_params["C"]), max(dummy_params["C"]), log=True)
            shrinking = trial.suggest_categorical("shrinking", dummy_params["shrinking"])
            class_weight = trial.suggest_categorical("class_weight", dummy_params["class_weight"])

            params = dict(
                C=c,
                shrinking=shrinking,
                class_weight=class_weight,
                kernel=dummy_params["kernel"][0],
                decision_function_shape=dummy_params["decision_function_shape"][0],
                max_iter=dummy_params["max_iter"][0],
                random_state=dummy_params["random_state"][0],
            )

        elif classifier_type == "rf":
            n_estimators = trial.suggest_int(
                "n_estimators", min(dummy_params["n_estimators"]), max(dummy_params["n_estimators"])
            )
            # max_features is mixed: categorical strings + continuous floats
            max_features_choice = trial.suggest_categorical(
                "max_features_type", ["sqrt", "log2", "float"]
            )
            if max_features_choice == "float":
                max_features = trial.suggest_float(
                    "max_features_float",
                    min(f for f in dummy_params["max_features"] if isinstance(f, float)),
                    max(f for f in dummy_params["max_features"] if isinstance(f, float)),
                )
            else:
                max_features = max_features_choice

            # max_depth is mixed: None + integers
            use_max_depth_ = trial.suggest_categorical("use_max_depth", [True, False])
            max_depth = (
                trial.suggest_int(
                    "max_depth",
                    min(d for d in dummy_params["max_depth"] if d is not None),
                    max(d for d in dummy_params["max_depth"] if d is not None),
                )
                if use_max_depth_
                else None
            )

            min_samples_split = trial.suggest_int(
                "min_samples_split",
                min(dummy_params["min_samples_split"]),
                max(dummy_params["min_samples_split"]),
            )
            min_samples_leaf = trial.suggest_int(
                "min_samples_leaf",
                min(dummy_params["min_samples_leaf"]),
                max(dummy_params["min_samples_leaf"]),
            )
            bootstrap = trial.suggest_categorical("bootstrap", dummy_params["bootstrap"])

            params = dict(
                n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                random_state=dummy_params["random_state"][0],
            )

        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type!r}")

        model = classifier(**params)
        model.fit(train_x, train_y)
        preds = model.predict(test_x)
        return accuracy_score(test_y, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=utils.SEED)
    )
    study.optimize(objective, n_trials=n_iter, n_jobs=N_JOBS)

    best_trial = study.best_trial
    test_acc = best_trial.value
    logger.info(f"... test accuracy: {test_acc:.6f}")
    logger.info(f"... best params: {best_trial.params}")

    # Reconstruct best params for final model fit, handling the RF mixed params
    best_params = best_trial.params.copy()
    fixed_params = {}

    if classifier_type == "lr":
        fixed_params = dict(
            solver=dummy_params["solver"][0],
            random_state=dummy_params["random_state"][0],
            max_iter=dummy_params["max_iter"][0],
        )
        # Restore the pruned C if penalty was None
        if best_params.get("penalty") is None:
            best_params["C"] = 1.0

    elif classifier_type == "svm":
        fixed_params = dict(
            kernel=dummy_params["kernel"][0],
            decision_function_shape=dummy_params["decision_function_shape"][0],
            max_iter=dummy_params["max_iter"][0],
            random_state=dummy_params["random_state"][0],
        )

    elif classifier_type == "rf":
        # Collapse the two-stage max_features and max_depth back into single params
        max_features_type = best_params.pop("max_features_type")
        best_params["max_features"] = (
            best_params.pop("max_features_float") if max_features_type == "float"
            else max_features_type
        )
        use_max_depth = best_params.pop("use_max_depth")
        if not use_max_depth:
            best_params.pop("max_depth", None)
            best_params["max_depth"] = None

        fixed_params = dict(random_state=dummy_params["random_state"][0])

    best_model = classifier(**best_params, **fixed_params)
    best_model.fit(train_x, train_y)
    best_preds = best_model.predict(valid_x)
    best_acc = accuracy_score(valid_y, best_preds)
    logger.info(f"... validation accuracy: {best_acc:.6f}")

    log_topk_acc(valid_x, valid_y, best_model)

    return best_model, best_acc, best_params
