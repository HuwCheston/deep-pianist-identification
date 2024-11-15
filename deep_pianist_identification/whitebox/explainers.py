#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Explainer classes for whitebox classification models."""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.whitebox.wb_utils import get_classifier_and_params

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
        with Parallel(n_jobs=-1, verbose=0) as par:
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


class DatabaseExplainer(WhiteBoxExplainer):
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
            topk: int = K_COEFS,
            n_features: int = None
    ):
        super().__init__(output_dir='database')
        # Set all passed in variables as attributes
        self.full_x = x
        self.full_y = y
        self.topk = topk
        self.dataset_idxs = dataset_idxs
        self.classifier_params = classifier_params
        self.class_mapping = class_mapping
        # Get the classifier type from the input string
        self.classifier, _ = get_classifier_and_params(classifier_type)
        # Then, create three classifier objects for 1) all data, 2) jtd only, 3) pijama only
        #  We'll fit these models later on, when calling `self.explain`
        self.full_md = self.classifier(**self.classifier_params)
        self.jtd_md = self.classifier(**self.classifier_params)
        self.pijama_md = self.classifier(**self.classifier_params)
        # Create arrays of indexes for each feature category: melody and harmony
        self.mel_idxs = np.array([i for i, f in enumerate(feature_names) if f.startswith('M_')])
        self.har_idxs = np.array([i for i, f in enumerate(feature_names) if f.startswith('H_')])
        # If we want to truncate the number of features (for low-memory systems)
        if n_features is not None:
            # Take only the first N melody and harmony indexes
            self.mel_idxs = self.mel_idxs[:n_features // 2]
            self.har_idxs = self.har_idxs[:n_features // 2]
            # Reduce the feature space
            self.full_x = self.full_x[:, [*self.mel_idxs, *self.har_idxs]]

    def get_topk_coefs_per_concept(self, coefs: np.array):
        """Gets idxs for the top-k melody and harmony coefficients across all performers"""
        all_topks_mel = []
        all_topks_har = []
        # Iterate through "rows" (performers)
        for perf in coefs:
            # Sort the indexes in descending order
            idxs = np.argsort(perf)[::-1]
            mel_idxs, har_idxs = [], []
            # Iterate through the indexes
            for i in idxs:
                # Append melody indexes to melody list
                if i in self.mel_idxs:
                    mel_idxs.append(i)
                # Append other indexes to other list
                else:
                    har_idxs.append(i)
            # Convert lists of idxs to arrays
            mel_arr = np.array(mel_idxs)
            har_arr = np.array(har_idxs)
            # Truncate to top-k weights if required
            if self.topk is not None:
                mel_arr = mel_arr[:self.topk]
                har_arr = har_arr[:self.topk]
            # Append to master list
            all_topks_mel.append(mel_arr)
            all_topks_har.append(har_arr)
        # Convert to arrays of shape [n_performers, top_k]
        return np.array(all_topks_mel), np.array(all_topks_har)

    def fit_database_models(self) -> tuple[np.array, np.array]:
        """Fits models only to the data from each source database"""
        # Get values for JTD
        jtd_x = np.array([i for num, i in enumerate(self.full_x) if self.dataset_idxs[num] == 1])
        jtd_y = np.array([i for num, i in enumerate(self.full_y) if self.dataset_idxs[num] == 1])
        # Get values for pijama
        pijama_x = np.array([i for num, i in enumerate(self.full_x) if self.dataset_idxs[num] != 1])
        pijama_y = np.array([i for num, i in enumerate(self.full_y) if self.dataset_idxs[num] != 1])
        # Check that the total number of rows adds up to the initial number
        assert jtd_x.shape[0] + pijama_x.shape[0] == self.full_x.shape[0]
        # Fit the models
        self.jtd_md.fit(jtd_x, jtd_y)
        self.pijama_md.fit(pijama_x, pijama_y)
        # Return the coefficients
        return self.jtd_md.coef_, self.pijama_md.coef_

    def get_database_coef_corrs(
            self,
            topk_idxs: np.array,
            n_boot: int = N_BOOT
    ) -> pd.DataFrame:
        """Gets performer-wise correlation between databases using top-k coefficients from full model"""

        def _booter(j, p):
            # Get the indexes for this iteration
            bi = np.random.choice(np.arange(len(j)), len(j), replace=True)
            # Get the correlation with these bootstrap indexes
            return np.corrcoef(j[bi], p[bi])[0, 1]

        # Get the coefficients from each database model
        jtd_coefs, pijama_coefs = self.jtd_md.coef_, self.pijama_md.coef_
        # List to store correlation coefficients
        coef_res = []
        # Iterate through each performer
        for i in tqdm(range(jtd_coefs.shape[0]), desc='Getting correlations...'):
            # Get the index of the top-k coefficients for this performer
            topk_perf_idxs = topk_idxs[i, :]
            # Get the top-k coefficients for this performer for both datasets
            jtd_perf_coefs = jtd_coefs[i, topk_perf_idxs]
            pijama_perf_coefs = pijama_coefs[i, topk_perf_idxs]
            # Get the actual correlation
            act_pcorr = np.corrcoef(jtd_perf_coefs, pijama_perf_coefs)[0, 1]
            # Get the bootstrapped correlations with multiprocessing
            with Parallel(n_jobs=-1, verbose=0) as par:
                boot_corrs = par(delayed(_booter)(jtd_perf_coefs, pijama_perf_coefs) for _ in range(n_boot))
            # Get the upper and lower bounds from the bootstrapped array
            boot_arr = np.array(boot_corrs)
            high, low = np.percentile(boot_arr, 97.5), np.percentile(boot_arr, 2.5)
            # Create a dictionary with everything in and append
            perf_res = dict(
                perf_name=self.class_mapping[i],
                corr=act_pcorr,
                high=high - act_pcorr,
                low=act_pcorr - low
            )
            coef_res.append(perf_res)
        # Return a dataframe sorted by correlation
        return (
            pd.DataFrame(coef_res)
            .sort_values(by='corr', ascending=False)
            .reset_index(drop=True)
        )

    def explain(self) -> pd.DataFrame:
        """Creates the explanation as a dataframe of correlation coefficients"""
        # Fit the full model to all the data
        self.full_md.fit(self.full_x, self.full_y)
        # Get idxs of the top-k melody and harmony features for each performer
        full_mel_topks, full_har_topks = self.get_topk_coefs_per_concept(self.full_md.coef_)
        # Fit the individual models to both JTD and PiJAMA data
        self.fit_database_models()
        # Get correlation coefficients for both melody and harmony as individual dataframes
        mel_df = self.get_database_coef_corrs(full_mel_topks)
        har_df = self.get_database_coef_corrs(full_har_topks)
        # Set an indexing value
        mel_df['feature'] = 'Melody'
        har_df['feature'] = 'Harmony'
        # Concatenate row-wise and return
        self.df = pd.concat([mel_df, har_df], axis=0).reset_index(drop=True)
        return self.df

    def create_outputs(self):
        super().create_outputs()
        # Create the barplot
        bp = plotting.BarPlotWhiteboxDatabaseCoeficients(self.df)
        bp.create_plot()
        bp.save_fig(os.path.join(self.output_dir, 'barplot_database_correlations.png'))
        # Save the dataframe
        self.df.to_csv(os.path.join(self.output_dir, 'out.csv'))
