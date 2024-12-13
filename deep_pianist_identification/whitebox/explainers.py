#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Explainer classes for whitebox classification models."""

import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.whitebox.wb_utils import get_classifier_and_params, N_ITER, N_JOBS

N_BOOT = 10000
K_COEFS = 2000
N_PERMUTATION_COEFS = 2000


class WhiteBoxExplainer:
    """Primary class for all other explainers to inherit from"""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(
            utils.get_project_root(),
            'reports/figures/whitebox',
            output_dir
        )
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        # We'll create this after running `self.explain`
        self.df = None

    def explain(self) -> pd.DataFrame:
        pass

    def create_outputs(self) -> None:
        if self.df is None:
            raise ValueError('Must call `.explain` before `.create_outputs`')


class LRWeightExplainer(WhiteBoxExplainer):
    """Creates explanations for weights generated by logistic regression model"""

    def __init__(
            self,
            x: np.array,
            y: np.array,
            feature_names: np.array,
            class_mapping: dict,
            classifier_type: str,
            classifier_params: dict,
            use_odds_ratios: bool = True,
            k: int = 5,
            n_iter: int = N_BOOT,
            n_features: int = None
    ):
        super().__init__(output_dir='lr_weights')
        # Unpack arguments to attributes
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.class_mapping = class_mapping
        self.k = k
        self.n_iter = n_iter
        # Quick checks here to make sure that all parameters are as expected
        if classifier_type not in ["lr", "svm"]:
            raise ValueError(
                f"Getting classifier weights only supports `lr` and `svm` models, but got {classifier_type}!")
        if use_odds_ratios and classifier_type != "lr":
            raise ValueError(f"Getting weights as odds ratios is only supported when `classifier_type == 'lr'`")
        # Get the classifier class (don't create the classifier)
        self.classifier_cls, _ = get_classifier_and_params(classifier_type)
        self.classifier_params = classifier_params
        self.use_odds_ratios = use_odds_ratios

        if n_features is not None:
            # Get equal numbers of melody and harmony features
            self.feature_names = np.hstack(
                [self.feature_names[:n_features // 2], self.feature_names[-n_features // 2:]])
            self.x = np.hstack([self.x[:, :n_features // 2], self.x[:, -n_features // 2:]])

    def fitter(self, x: np.array, y: np.array) -> np.array:
        # Create and fit the classifier and take the weights
        temp_model = self.classifier_cls(**self.classifier_params)
        temp_model.fit(x, y)
        coef = temp_model.coef_
        # Take exponential if we want odds ratios
        if self.use_odds_ratios:
            coef = np.exp(coef)
        return coef

    def booter(self, x: pd.DataFrame, y: pd.Series, i: int = utils.SEED) -> np.array:
        # Sample the features
        x_samp = x.sample(frac=1, replace=True, random_state=i)
        # Use the index to sample the y values
        y_samp = y[x_samp.index]
        # Return the coefficients
        return self.fitter(x_samp, y_samp)

    def get_classifier_weights(self) -> tuple:
        # Convert to a dataframe as this makes bootstrapping a lot easier
        full_x_df = pd.DataFrame(self.x, columns=self.feature_names)
        full_y_df = pd.Series(self.y)
        # Get actual coefficients: (n_classes, n_features)
        ratios = self.fitter(full_x_df, full_y_df)
        # Bootstrap the coefficients: (n_boots, n_classes, n_features)
        with Parallel(n_jobs=N_JOBS, verbose=0) as par:
            boot_ratios = par(delayed(self.booter)(
                full_x_df, full_y_df, i) for i in tqdm(range(self.n_iter), desc='Bootstrapping weights...')
                              )
        return ratios, boot_ratios

    def get_topk_features(
            self,
            weights: np.array,
            bootstrapped_weights: list[np.array],
    ) -> dict:
        """For weights (with bootstrapping), get top and bottom `k` features for each performer"""
        res = {}
        # Iterate over all classes
        for class_idx in range(len(self.class_mapping.keys())):
            # Get the actual and bootstrapped weights for this performer
            class_ratios = weights[class_idx, :]
            boot_class_ratios = np.array([b[class_idx, :] for b in bootstrapped_weights])
            # Apply functions to the bootstrapped weights to get SD, lower, upper bounds
            stds = np.apply_along_axis(np.std, 0, boot_class_ratios)
            lower_bounds = np.apply_along_axis(lambda x: np.percentile(x, 2.5), 0, boot_class_ratios)
            upper_bounds = np.apply_along_axis(lambda x: np.percentile(x, 97.5), 0, boot_class_ratios)
            # Get sorted idxs in ascending/descending order
            bottom_idxs = np.argsort(class_ratios)
            # TODO: check that this is working as expected
            top_idxs = bottom_idxs[::-1]
            perf_res = {}
            # Iterate through all concepts
            for concept, starter in zip(['melody', 'harmony'], ['M_', 'H_']):
                # Get top idxs
                top_m_idxs = np.array([i for i in top_idxs if self.feature_names[i].startswith(starter)])[:self.k]
                top_m_features = self.feature_names[top_m_idxs]
                top_m_ors = class_ratios[top_m_idxs]
                top_m_high = upper_bounds[top_m_idxs] - top_m_ors
                top_m_low = top_m_ors - lower_bounds[top_m_idxs]
                top_m_stds = stds[top_m_idxs]
                # Get bottom idxs
                bottom_m_idxs = np.array([i for i in bottom_idxs if self.feature_names[i].startswith(starter)])[:self.k]
                bottom_m_features = self.feature_names[bottom_m_idxs]
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
                perf_res[concept] = r
            res[self.class_mapping[class_idx]] = perf_res
        return res

    def explain(self):
        weights, boot_weights = self.get_classifier_weights()
        self.df = self.get_topk_features(weights, boot_weights)

    def create_outputs(self):
        super().create_outputs()
        # Dump top-k dictionary to a JSON
        # with open(os.path.join(self.output_dir, 'topk.json'), 'w') as fp:
        #     json.dump(self.df, fp)
        # Iterate through all performers and concepts
        for perf in self.df.keys():
            for concept in ['melody', 'harmony']:
                # Create the plot and save in the default (reports/figures/whitebox) directory
                sp = plotting.StripplotTopKFeatures(
                    self.df[perf][concept],
                    pianist_name=perf,
                    concept_name=concept
                )
                sp.create_plot()
                sp.save_fig(self.output_dir)


class PermutationExplainer(WhiteBoxExplainer):
    def __init__(
            self,
            harmony_features: np.array,
            melody_features: np.array,
            y: np.array,
            classifier,
            init_acc: float,
            scale: bool = True,
            n_iter: int = N_BOOT,
            n_boot_features: int = N_PERMUTATION_COEFS,
            n_features: int = None
    ):
        super().__init__(output_dir='permutation')
        self.harmony_features = harmony_features
        self.melody_features = melody_features
        if n_features is not None:
            self.harmony_features = self.harmony_features[:, :n_features // 2]
            self.melody_features = self.melody_features[:, :n_features // 2]
        self.y = y
        self.classifier = classifier
        self.init_acc = init_acc
        self.n_iter = n_iter
        self.n_boot_features = n_boot_features
        # If we're going to be z-transforming features
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()
        # We'll create this when calling `self.explain`
        self.df = None

    def permutation_importance(self, features: np.array) -> float:
        """Compute loss in accuracy vs non-permuted model with given permuted `features`"""
        # Make predictions and take accuracy
        valid_y_pred_permute = self.classifier.predict(features)
        permute_acc = accuracy_score(self.y, valid_y_pred_permute)
        # Compute the loss in accuracy vs the non-permuted model
        return self.init_acc - permute_acc

    def shuffler(
            self,
            features_to_shuffle: np.array,
            features_not_to_shuffle: np.array,
            shuffled_first: bool,
            idxs: np.ndarray = None,
            seed: int = utils.SEED,
    ) -> float:
        """Shuffles one array (or a subset of columns in array) and concatenates"""
        rng = np.random.default_rng(seed=seed)  # Used to permute arrays
        # Make a copy of the data and shuffle
        shuff = deepcopy(features_to_shuffle)
        # If we're using all features, i.e., we're not bootstrapping
        if idxs is None:
            shuff = rng.permuted(shuff, axis=0)
        # If we're only using some features, i.e., we're bootstrapping
        else:
            # Take the corresponding columns
            cols = shuff[:, idxs]
            # Permute these column
            permuted = rng.permutation(cols, axis=0)
            # Set the columns back correctly
            shuff[:, idxs] = permuted
        # If the shuffled features should be "left-most" in the array
        #  this is the case for melody features
        if shuffled_first:
            combined = np.concatenate([shuff, features_not_to_shuffle], axis=1)
        # Otherwise, the shuffled features will be "right-most"
        #  this is the case for harmony features
        else:
            combined = np.concatenate([features_not_to_shuffle, shuff], axis=1)
        # Z-transform data if required
        if self.scale:
            combined = self.scaler.fit_transform(combined)
        # Return loss in accuracy with permuted feature set
        return self.permutation_importance(combined)

    def _get_harmony_feature_importance(self) -> tuple[np.array, np.array]:
        """Calculates harmony feature importance, both for all features and with bootstrapped subsamples of features"""

        def _shuffler_boot(seed):
            # Get an array of bootstrapping indexes to use
            boot_idxs = np.random.choice(self.harmony_features.shape[1], self.n_boot_features)
            return self.shuffler(self.harmony_features, self.melody_features, False, boot_idxs, seed)

        with Parallel(n_jobs=1, verbose=0) as par:
            # Permute all features
            har_acc = par(delayed(self.shuffler)(
                self.harmony_features, self.melody_features, False, None, seed
            ) for seed in tqdm(range(self.n_iter), desc="Getting harmony importance..."))
            # Permute bootstrapped subsamples of features
            boot_har_accs = par(delayed(_shuffler_boot)(
                seed) for seed in tqdm(range(self.n_iter), desc="Bootstrapping harmony importance..."))
        return np.array(har_acc), np.array(boot_har_accs)

    def _get_melody_feature_importance(self) -> tuple[np.array, np.array]:
        """Calculates melody feature importance, both for all features and with bootstrapped subsamples of features"""

        def _shuffler_boot(seed):
            # Get an array of bootstrapping indexes to use
            boot_idxs = np.random.choice(self.melody_features.shape[1], self.n_boot_features)
            return self.shuffler(self.melody_features, self.harmony_features, True, boot_idxs, seed)

        with Parallel(n_jobs=1, verbose=0) as par:
            # Permute all features
            mel_acc = par(delayed(self.shuffler)(
                self.melody_features, self.harmony_features, True, None, seed
            ) for seed in tqdm(range(self.n_iter), desc="Getting melody importance..."))
            # Permute bootstrapped subsamples of features
            boot_mel_accs = par(delayed(_shuffler_boot)(
                seed) for seed in tqdm(range(self.n_iter), desc="Bootstrapping melody importance..."))
        return np.array(mel_acc), np.array(boot_mel_accs)

    @staticmethod
    def format_dict(accs: np.array, name: str) -> dict:
        return dict(
            feature=name,
            mean=np.mean(accs),
            std=np.std(accs),
            high=np.percentile(accs, 2.5),
            low=np.percentile(accs, 97.5),
        )

    def explain(self) -> pd.DataFrame:
        # Compute harmony feature importance
        har_acc, har_boot = self._get_harmony_feature_importance()
        # Format harmony dictionaries
        har_dict = self.format_dict(har_acc, "Harmony")
        har_boot_dict = self.format_dict(har_boot, "Harmony (Bootstrapped)")
        # Compute melody feature importance
        mel_acc, mel_boot = self._get_melody_feature_importance()
        mel_dict = self.format_dict(mel_acc, "Melody")
        mel_boot_dict = self.format_dict(mel_boot, "Melody (Bootstrapped)")
        # Concatenate as a single dataframe
        self.df = pd.DataFrame([har_dict, har_boot_dict, mel_dict, mel_boot_dict])
        return self.df

    def create_outputs(self):
        super().create_outputs()
        # Save the dataframe
        self.df.to_csv(os.path.join(self.output_dir, 'out.csv'))


class DatabasePermutationExplainer(WhiteBoxExplainer):
    """Creates explanations for correlation between top-k melody/harmony features from both source datasets"""

    def __init__(
            self,
            x: np.array,
            y: np.array,
            dataset_idxs: np.array,
            feature_names: np.array,
            class_mapping: dict,
            classifier_params: dict,
            classifier_type: str,
            bonferroni: bool = True,
            n_iter: int = N_ITER,
            n_features: int = None
    ):
        super().__init__(output_dir='database')
        # Set all passed in variables as attributes
        self.full_x = x
        self.full_y = y
        self.dataset_idxs = dataset_idxs
        self.classifier_params = classifier_params
        self.class_mapping = class_mapping
        self.n_iter = n_iter
        self.bonferroni = bonferroni
        # Get the classifier type from the input string
        self.classifier, _ = get_classifier_and_params(classifier_type)
        # Create arrays of indexes for each feature category: melody and harmony
        self.mel_idxs = np.array([i for i, f in enumerate(feature_names) if f.startswith('M_')])
        self.har_idxs = np.array([i for i, f in enumerate(feature_names) if f.startswith('H_')])
        # We'll fill these variables later
        self.mel_coefs, self.mel_ps, self.mel_null_dists = None, None, None
        self.har_coefs, self.har_ps, self.har_null_dists = None, None, None
        # If we want to truncate the number of features (for low-memory systems)
        if n_features is not None:
            # Take only the first N melody and harmony indexes and reduce the feature space
            reduced_x_idxs = [*self.mel_idxs[:n_features // 2], *self.har_idxs[:n_features // 2]]
            self.full_x = self.full_x[:, reduced_x_idxs]
            # Set the idxs properly so we can still subset the reduced feature sapce
            self.mel_idxs = np.arange(0, n_features // 2)
            self.har_idxs = np.arange(n_features // 2, n_features)
        # Sort everything so we have y-values in ascending order (all class 0 tracks, all class 1 tracks, ...)
        sorters = np.argsort(self.full_y)
        self.full_x = self.full_x[sorters, :]
        self.full_y = self.full_y[sorters]
        self.dataset_idxs = self.dataset_idxs[sorters]

    def get_weights(self, all_dataset_idxs, desired_dataset_idx, feature_idxs):
        # Get indexes for rows (tracks) for this dataset
        row_idxs = np.argwhere(all_dataset_idxs == desired_dataset_idx).flatten()
        # Subset feature and target variables
        current_xs, current_ys = self.full_x[np.ix_(row_idxs, feature_idxs)], self.full_y[row_idxs]
        # Create the LR model and fit
        full_model = self.classifier(**self.classifier_params)
        full_model.fit(current_xs, current_ys)
        # Return the coefficients and stderrs
        return full_model.coef_

    def get_permutation_pval(self, test_statistic: float, null_distribution: np.array) -> float:
        # Center both distributions on 0 by subtracting the mean of the null distribution
        shifted_null_distribution = null_distribution - np.mean(null_distribution)
        shifted_obs_stat = test_statistic - np.mean(null_distribution)
        # Count how many values in null distribution are as extreme or more extreme as observed
        # We use both directions here for a two-sided test
        extreme_count = np.sum(
            (shifted_null_distribution <= -abs(shifted_obs_stat)) |
            (shifted_null_distribution >= abs(shifted_obs_stat))
        )
        # Compute the p-value: adding 1 to the denominator ensures we never have p == 0
        pval = extreme_count / (len(null_distribution) + 1)
        # Apply multiple correction testing if required
        if self.bonferroni:
            # number of tests = [harmony, melody] = 2
            n_tests = 2
            # Clamping to 1 in case we increase beyond this
            pval = min(pval * n_tests, 1.)
        return pval

    def get_null_distributions(self, col_idxs):
        n_performers = len(np.unique(self.full_y))

        def booter(bidx):
            rng = np.random.default_rng(seed=bidx)
            # Shuffle the idx of dataset values for each performer's recordings and then stack together
            # I.e., for Keith Jarrett, we don't want to end up with more unaccompanied recordings than we originally had
            # So we're shuffling the indices for each performer separately and then stacking them together at the end
            permuted_dataset_idxs = np.concatenate(
                [rng.permutation(self.dataset_idxs[self.full_y == i]) for i in range(n_performers)]
            )
            # Get weights for the shuffled "pijama" tracks
            null_pijama = self.get_weights(permuted_dataset_idxs, 0, col_idxs)
            # Get weights for the shuffled "jtd" tracks
            null_jtd = self.get_weights(permuted_dataset_idxs, 1, col_idxs)
            # Correlate across both datasets for each performer
            null_coefs = np.array([np.corrcoef(null_pijama[i, :], null_jtd[i, :])[0, 1] for i in range(n_performers)])
            return null_coefs

        with Parallel(n_jobs=-1, verbose=0) as parallel:
            boots = parallel(delayed(booter)(boot_idx) for boot_idx in tqdm(range(self.n_iter)))
        return np.stack(list(boots))

    @staticmethod
    def pval_to_asterisk(pval):
        critical_values = [0.05, 0.01, 0.001]
        asterisks = ['*', '**', '***']
        # Iterate over all critical values and update value in array
        set_asterisk = ''
        for thresh, asterisk in zip(critical_values, asterisks):
            if pval <= thresh:
                set_asterisk = asterisk
        return set_asterisk

    def get_correlation_coefficients(self, col_idxs: np.array):
        # Get model weights for both pijama and jtd
        pijama_coef = self.get_weights(self.dataset_idxs, 0, col_idxs)
        jtd_coef = self.get_weights(self.dataset_idxs, 1, col_idxs)
        # Shape (N_performers, N_features)
        assert pijama_coef.shape == jtd_coef.shape
        # Compute the correlation coefficients: shape (N_performers)
        return np.array(
            [np.corrcoef(pijama_coef[i, :], jtd_coef[i, :])[0, 1] for i in range(pijama_coef.shape[0])]
        )

    def get_permutation_test_results(self, col_idxs: np.array, corrcoefs: np.array):
        # Get null distributions: shape (n_iter, n_performers)
        null_dists = self.get_null_distributions(col_idxs)
        # Get p-values as proportion of values that are as extreme as observed value
        p_vals = np.array(
            [self.get_permutation_pval(corrcoefs[pidx], null_dists[:, pidx]) for pidx in range(corrcoefs.shape[0])]
        )
        return p_vals, null_dists

    def format_df(self, mel_coefs, mel_ps, har_coefs, har_ps, ):
        tmp_res = []
        for pianist, m_coef, m_p, h_coef, h_p in zip(self.class_mapping.values(), mel_coefs, mel_ps, har_coefs, har_ps):
            # Append multiple dictionaries
            tmp_res.append(dict(pianist=pianist, corr=m_coef, p=m_p, sig=self.pval_to_asterisk(m_p), feature="melody"))
            tmp_res.append(dict(pianist=pianist, corr=h_coef, p=h_p, sig=self.pval_to_asterisk(h_p), feature="harmony"))
        return pd.DataFrame(tmp_res)

    def explain(self):
        self.mel_coefs = self.get_correlation_coefficients(self.mel_idxs)
        self.har_coefs = self.get_correlation_coefficients(self.har_idxs)
        self.mel_ps, self.mel_null_dists = self.get_permutation_test_results(self.mel_idxs, self.mel_coefs)
        self.har_ps, self.har_null_dists = self.get_permutation_test_results(self.har_idxs, self.har_coefs)
        self.df = self.format_df(self.mel_coefs, self.mel_ps, self.har_coefs, self.har_ps)

    def create_outputs(self):
        # Save the dataframe
        self.df.to_csv(os.path.join(self.output_dir, 'database_correlations.csv'))
        # Create the barplot
        bp = plotting.BarPlotWhiteboxDatabaseCoefficients(self.df)
        bp.create_plot()
        bp.save_fig(os.path.join(self.output_dir, 'barplot_database_correlations.png'))
        # Create the histplot
        hp = plotting.HistPlotDatabaseNullDistributionCoefficients(
            self.mel_null_dists, self.har_null_dists,
            self.mel_coefs, self.har_coefs,
            self.mel_ps, self.har_ps,
            self.class_mapping
        )
        hp.create_plot()
        hp.save_fig(os.path.join(self.output_dir, 'histplot_nulldistribution_coefficients.png'))
        # Dump self to a pickle file
        with open(os.path.join(self.output_dir, 'database_correlations.p'), 'wb') as out:
            pickle.dump(self, out)


class DatabaseTopKExplainer(WhiteBoxExplainer):
    """Creates explanations for correlation between top-k melody/harmony features from both source datasets"""

    def __init__(
            self,
            x: np.array,
            y: np.array,
            dataset_idxs: np.array,
            feature_names: np.array,
            class_mapping: dict,
            classifier_params: dict,
            classifier_type: str,
            top_k: float,
            n_features: int = None
    ):
        super().__init__(output_dir='database_topk')
        # Set all passed in variables as attributes
        self.full_x = x
        self.full_y = y
        self.classifier_params = classifier_params
        self.class_mapping = class_mapping
        self.top_k = top_k
        assert top_k <= 1.
        # Get indices of tracks for each dataset
        self.dataset_idxs = dataset_idxs
        self.jtd_idxs = np.argwhere(dataset_idxs == 1).flatten()
        self.pijama_idxs = np.argwhere(dataset_idxs == 0).flatten()
        # Get the classifier type from the input string
        self.classifier, _ = get_classifier_and_params(classifier_type)
        # Create arrays of indexes for each feature category: melody and harmony
        self.mel_idxs = np.array([i for i, f in enumerate(feature_names) if f.startswith('M_')])
        self.har_idxs = np.array([i for i, f in enumerate(feature_names) if f.startswith('H_')])
        # If we want to truncate the number of features (for low-memory systems)
        if n_features is not None:
            # Take only the first N melody and harmony indexes and reduce the feature space
            reduced_x_idxs = [*self.mel_idxs[:n_features // 2], *self.har_idxs[:n_features // 2]]
            self.full_x = self.full_x[:, reduced_x_idxs]
            # Set the idxs properly so we can still subset the reduced feature sapce
            self.mel_idxs = np.arange(0, n_features // 2)
            self.har_idxs = np.arange(n_features // 2, n_features)

    def _get_weights(self, x: np.array, y: np.array) -> np.array:
        # Create model and fit to corresponding features
        md = self.classifier(**self.classifier_params)
        md.fit(x, y)
        return md.coef_

    def get_topk_weight_idxs(self, feature_idxs: np.array) -> np.array:
        # Fit the full model to all data using all the features
        full_weights = self._get_weights(self.full_x[:, feature_idxs], self.full_y)
        # Get the maximum weight for each feature: shape (n_features)
        max_weights = full_weights.max(axis=0)
        assert len(max_weights) == len(feature_idxs)
        # Get the total number of features to use
        k_int = int(len(feature_idxs) * self.top_k)
        # Get the indices for these top-k features
        topk_idxs = np.argsort(max_weights)[::-1][:k_int]
        return feature_idxs[topk_idxs]

    def get_performer_correlations(self, topk_idxs: np.array) -> np.array:
        # Get weights for pijama tracks: (n_performers, top_k_features)
        pijama_coef = self._get_weights(
            self.full_x[np.ix_(self.pijama_idxs, topk_idxs)],  # subset feature set to just use top-k features
            self.full_y[self.pijama_idxs]
        )
        # Get weights for JTD tracks: (n_performers, top_k_features)
        jtd_coef = self._get_weights(
            self.full_x[np.ix_(self.jtd_idxs, topk_idxs)],  # subset feature set to just use top-k features
            self.full_y[self.jtd_idxs]
        )
        # Correlate both: (n_performers)
        return np.array([
            np.corrcoef(pijama_coef[i, :], jtd_coef[i, :])[0, 1] for i in range(pijama_coef.shape[0])
        ])

    def format_df(self, corrs: np.array, feature_name: str) -> pd.DataFrame:
        tmp_res = []
        for pianist, corr in zip(self.class_mapping.values(), corrs):
            tmp_res.append(dict(pianist=pianist, corr=corr, p=np.nan, feature=feature_name.title(), low=0., high=0.))
        return pd.DataFrame(tmp_res)

    def explain(self):
        # Melody
        mel_topk_idxs = self.get_topk_weight_idxs(self.mel_idxs)
        mel_corrs = self.get_performer_correlations(mel_topk_idxs)
        mel_df = self.format_df(mel_corrs, "melody")
        # Harmony
        har_topk_idxs = self.get_topk_weight_idxs(self.har_idxs)
        har_corrs = self.get_performer_correlations(har_topk_idxs)
        har_df = self.format_df(har_corrs, "harmony")
        # Stack dataframes
        self.df = pd.concat([mel_df, har_df], axis=0)

    def create_outputs(self):
        # Create the barplot
        bp = plotting.BarPlotWhiteboxDatabaseCoefficients(self.df)
        bp.create_plot()
        bp.fig.suptitle(f'$k$ = {self.top_k}')
        bp.save_fig(os.path.join(self.output_dir, f'barplot_database_correlations_topk_{self.top_k}.png'))
        # Save the dataframe
        self.df.to_csv(os.path.join(self.output_dir, f'out_{self.top_k}.csv'))
