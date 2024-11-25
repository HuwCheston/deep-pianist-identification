#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility modules (dataloaders, etc.) for generating CAVs using MIDI from textbooks"""

import os
from itertools import groupby
from random import shuffle
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
from joblib import Parallel, delayed
from loguru import logger
from pretty_midi import PrettyMIDI
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import get_piano_roll

DEFAULT_MODEL = ("disentangle-resnet-channel/"
                 "disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc")
# Array of names for all CAVs
_mapping_loc = os.path.join(utils.get_project_root(), 'references/_unsupervised_resources/voicings/cav_mapping.csv')
CAV_MAPPING = np.array([row.iloc[0] for i, row in pd.read_csv(_mapping_loc, index_col=1).iterrows()])
# Array of pianist birth years (not sorted)
BIRTH_YEARS = np.genfromtxt(
    os.path.join(utils.get_project_root(), 'references/data_splits/20class_80min/birth_years.txt'), dtype=int
)
TEST_SIZE = 1 / 3  # same as initial TCAV implementation
SCALER = StandardScaler()
BATCH_SIZE = 10
TRANSPOSE_RANGE = 6  # Transpose exercise MIDI files +/- 6 semitones (up and down a tri-tone)
N_JOBS = 1  # seems to be faster than parallel processing?
N_EXPERIMENTS = 100
N_RANDOM_CLIPS = 250


def parse_arguments(argparser) -> dict:
    """Adds required CLI arguments to an `ArgumentParser` object and parses them to a dictionary"""
    argparser.add_argument(
        '-m', '--model',
        default=DEFAULT_MODEL,
        type=str,
        help="Name of a trained model with saved checkpoints"
    )
    argparser.add_argument(
        '-a', '--attribution-fn',
        default='gradient_x_activation',
        type=str,
        help='Function to use to compute layer attributions (see captum.TCAV)'
    )
    argparser.add_argument(
        '-i', '--multiply-by-inputs',
        default=True,
        type=bool,
        help='Multiply layer activations by input (see captum.TCAV)'
    )
    argparser.add_argument(
        '-e', '--n-experiments',
        default=N_EXPERIMENTS,
        type=int,
        help='Number of experiments to run (i.e., number of CAVs to create per concept)'
    )
    argparser.add_argument(
        '-r', '--n-random-clips',
        default=N_RANDOM_CLIPS,
        type=int,
        help='Number of clips to use when creating random CAV'
    )
    argparser.add_argument(
        '-c', '--n-cavs',
        default=len(CAV_MAPPING),
        type=int,
        help='Number of CAVs to create, defaults to 20'
    )
    argparser.add_argument(
        '-b', '--batch-size',
        default=BATCH_SIZE,
        type=int,
        help='Size of batches to use for GPU processing'
    )
    return vars(argparser.parse_args())


def get_pvals(
        concept_signcounts: np.array,
        random_signcounts: np.array,
) -> np.array:
    """Performs t-test for sign counts from one concept"""
    assert concept_signcounts.shape == random_signcounts.shape
    # shape: (n_performers)
    return np.array([
        stats.ttest_ind(random, concept)[1] for concept, random
        in zip(concept_signcounts, random_signcounts)
    ])


def get_pval_significance(
        pvals: np.array,
        critical_values: list = None,
        bonferroni: bool = True,
) -> np.array:
    # Use default critical values if not provided
    if critical_values is None:
        critical_values = [0.05, 0.01, 0.001]
    # Apply Bonferroni correction to all critical values if required
    if bonferroni:
        num_experiments = pvals.shape[0]
        critical_values = [c / num_experiments for c in critical_values]
    # Sort critical values in order from largest to smallest
    critical_values = np.sort(np.array(critical_values))[::-1]

    all_sigs = []
    # Iterate over the metrics for each CAV (shape = n_performers)
    for experiment in pvals.T:
        exp_sigs = []
        # Iterate over each individual p-value
        for p in experiment:
            ast = ''
            # Set the required asterisk vaule
            for crit_num, crit_value in enumerate(critical_values, 1):
                if p < crit_value:
                    ast = '*' * crit_num
            exp_sigs.append(ast)
        all_sigs.append(np.array(exp_sigs))
    fmt = np.column_stack(all_sigs)
    # Expected shape of this array is (n_cavs, n_performers)
    assert fmt.shape == pvals.shape
    return fmt


class CAV:
    DL_KWARGS = dict(shuffle=True, drop_last=False)

    def __init__(
            self,
            cav_idx: int,
            model: torch.nn.Module,
            layer: torch.nn.Module,
            layer_attribution: str = "gradient_x_activation",
            multiply_by_inputs: bool = True,
            standardize: bool = False,
            batch_size: int = BATCH_SIZE
    ):
        self.cav_idx = cav_idx
        self.batch_size = batch_size
        # Create one concept dataset, used in every experiment
        self.concept_dataset = self.initialise_concept_dataloader()
        # Set model and layer to attributes
        self.model = model
        self.layer = layer
        # Get layer attribution function
        attr_fn = self.get_attribution_fn(layer_attribution)
        self.layer_attribution = attr_fn(model, layer, multiply_by_inputs=multiply_by_inputs)
        # Variables for linear classifier separating concept and random activations
        self.classifier = LogisticRegression
        self.standardize = standardize
        # All of these variables will be updated when creating the CAV
        self.acc = []  # Accuracy scores of linear models separating concept and random datasets
        self.cav = []
        self.magnitudes = []
        self.sign_counts = []

    def initialise_concept_dataloader(self):
        # Create one concept dataset, which we'll use in every experiment
        return DataLoader(
            # TODO: do we want to set a hard cap on the number of clips to create here?
            #  Probably only important if we're creating multiple concept datasets, which we're not currently
            VoicingLoaderReal(
                self.cav_idx,
                n_clips=None,  # use all possible clips for the concept
                transpose_range=TRANSPOSE_RANGE
            ),
            batch_size=self.batch_size,
            **self.DL_KWARGS
        )

    def initialise_random_dataloader(self, n_clips: int):
        return DataLoader(
            VoicingLoaderFake(
                self.cav_idx,
                n_clips=n_clips,  # Use the size of the concept loader dataset
                transpose_range=TRANSPOSE_RANGE
            ),
            batch_size=self.batch_size,
            **self.DL_KWARGS
        )

    @staticmethod
    def get_attribution_fn(attr_fn_str: str) -> Callable:
        """Returns correct layer attribution function from `captum` depending on input `attr_fn_str`"""
        accept = ['integrated_gradients', 'gradient_x_activation']
        assert attr_fn_str in accept, f'{attr_fn_str} not in {", ".join(accept)}'
        if attr_fn_str == 'integrated_gradients':
            return LayerIntegratedGradients
        else:
            return LayerGradientXActivation

    def get_activations(self, loader: DataLoader) -> np.array:
        """Gets activations for `model`.`layer` using data contained in `loader`"""
        acts = {}

        def hooky(_, __, out: torch.tensor):
            acts['values'] = out

        # Register the handle to get the activations from the desired layer
        handle = self.layer.register_forward_hook(hooky)
        embeds = []
        for roll, _ in loader:
            # roll = (B, 4, W, H)
            # Create both sets of embeddings
            with torch.no_grad():
                # Forward pass through the model
                _ = self.model(roll.to(utils.DEVICE))
            # Get the activations from the hook
            roll_acts = acts['values']
            # Channel dimension is either at 1 or 0 depending on if input is batched or not
            start_dim = 1 if len(roll_acts.size()) == 4 else 0
            # Captum uses the unpooled output, flattened (i.e., C * W * H), so we do too
            embed = roll_acts.flatten(start_dim=start_dim)
            # Append to the list
            embeds.append(embed.cpu().numpy())
            acts = {}  # for safety, clear this dictionary
        # Remove the hook
        handle.remove()
        # Concatenate the embeddings
        return np.concatenate(embeds)

    @ignore_warnings(category=ConvergenceWarning)
    def get_cav(
            self,
            real_acts: np.array,
            rand_acts: np.array,
    ) -> tuple[torch.tensor, float]:
        """Given `real` and `random` activations, return coefficients for the classifier"""
        # Create arrays of targets with same shape as embeddings
        real_targets, rand_targets = np.ones(real_acts.shape[0]), np.zeros(rand_acts.shape[0])
        # Combine both real/fake features and targets into single matrix
        x = np.vstack([real_acts, rand_acts])
        y = np.hstack([real_targets, rand_targets])
        # Scale the features if required; not used in initial TCAV implementation
        if self.standardize:
            x = SCALER.fit_transform(x)
        # Split into train-test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, stratify=y, random_state=utils.SEED
        )
        # # Log dimensionality of all inputs
        # logger.info(f'Shape of training inputs: X = {x_train.shape}, y = {y_train.shape}')
        # logger.info(f'Shape of testing inputs: X = {x_test.shape}, y = {y_test.shape}')
        # Initialise the classifier class
        classifier = self.classifier(random_state=utils.SEED)
        # Fit the model and predict the test set
        classifier.fit(x_train, y_train)  # will probably raise convergence warnings
        y_pred = classifier.predict(x_test)
        # Get test set accuracy and log
        acc = accuracy_score(y_test, y_pred)
        # Extract coefficients - these are our CAVs
        # TODO: still unsure whether something needs to happen for `orthogonal` here
        return torch.tensor(classifier.coef_.astype(np.float32)).flatten(), acc

    def compute_cav_sensitivity(self, feat: torch.tensor, targ: torch.tensor, cav: torch.tensor) -> float:
        """Given input `features` and `targets`, compute sensitivitiy to provided `cav`"""
        # Copy the tensor and move to GPU
        inputs = feat.to(utils.DEVICE)
        targ = targ.to(utils.DEVICE)
        # Apply masks to every roll other than the harmony roll
        for mask_idx in [0, 2, 3]:
            inputs[:, mask_idx, :, :] = torch.zeros_like(inputs[:, mask_idx, :, :])
        # We don't seem to require setting inputs.requires_grad_ = True
        # Compute layer activations for final convolutional layer of harmony concept WRT target
        # We set target=targ to get LayerIntegratedGradients to work as this expects a 2nd positional arg
        acts = self.layer_attribution.attribute(inputs, target=targ)
        # Flatten to get C * W * H (captum does this)
        # We can probably detach at this point to save memory
        flatted = acts.flatten(start_dim=1).detach()
        # Compute dot product against the target CAV
        return torch.matmul(flatted, cav.to(utils.DEVICE)).cpu()

    def compute_cav_sensitivities(self, features: torch.tensor, targets: torch.tensor) -> torch.tensor:
        # Create dataloader which returns piano roll and target for each item in batches
        tensor_dataloader = DataLoader(
            TensorDataset(features, targets),
            batch_size=self.batch_size,
            shuffle=False,  # No real need for this
            drop_last=False  # we want to keep all elements
        )
        # For every CAV we've created for this concept
        sensitivities = []
        for cav_num, cav in tqdm(enumerate(self.cav, 1), total=len(self.cav), desc='Computing sensitivities'):
            # Compute the sensitivity of all features and targets
            cav_sensitivities = [self.compute_cav_sensitivity(f, t, cav) for f, t in tensor_dataloader]
            # Concatenate the list: shape is now (n_clips)
            sensitivities.append(torch.cat(cav_sensitivities))
        # Stack to an array of shape (n_experiments, n_clips)
        return torch.stack(sensitivities)

    @staticmethod
    def get_magnitude(tcav: torch.tensor) -> float:
        """Get magnitude of values in provided `tcav` vector (an array of sensitivities, one per clip)"""
        return (torch.sum(torch.abs(tcav * (tcav > 0.0).float()), dim=0) / torch.sum(torch.abs(tcav), dim=0)).item()

    @staticmethod
    def get_sign_count(tcav: torch.tensor) -> float:
        """Get sign count of values in provided `tcav` vector (an array of sensitivities, one per clip)"""
        return (torch.sum(tcav > 0.0, dim=0).float() / tcav.shape[0]).item()

    def fit(self, n_experiments: int = N_EXPERIMENTS) -> None:
        """Fits the classifier to concept and random datasets and gets the CAVs and accuracy scores"""
        # Get activations from the concept dataset
        real_acts = self.get_activations(self.concept_dataset)
        # Iterate through all random datasets
        for _ in tqdm(range(n_experiments), desc='Fitting classifiers'):
            # Get a random dataloader
            rand_dataset = self.initialise_random_dataloader(n_clips=len(self.concept_dataset.dataset))
            # Get activations from this dataloader
            rand_acts = self.get_activations(rand_dataset)
            # Get the CAV and accuracy scores
            cav, acc = self.get_cav(real_acts, rand_acts)
            # Append everything to a list
            self.acc.append(acc)
            self.cav.append(cav)
        # Log the accuracy to the console
        logger.info(f'Mean test accuracy for CAV: {np.mean(self.acc):.5f}, SD {np.std(self.acc):.5f}')

    def interpret(self, features: torch.tensor, targets: torch.tensor, class_mapping: list) -> None:
        # Get sensitivity for all features/targets to all CAVs
        # Shape is (n_experiments, n_clips)
        sensitivities = self.compute_cav_sensitivities(features, targets)
        # Iterate over all class indexes
        for class_idx in class_mapping:
            # Get indices of clips by this performer
            clip_idxs = torch.argwhere(targets == class_idx)
            # Get class sign counts and magnitudes for sensitivity to all CAVs
            class_sign_counts = np.array([self.get_sign_count(re_r[clip_idxs]) for re_r in sensitivities])
            class_magnitudes = np.array([self.get_magnitude(re_r[clip_idxs]) for re_r in sensitivities])
            # Append the sign count and magnitude for all CAVs to the list
            self.sign_counts.append(class_sign_counts)
            self.magnitudes.append(class_magnitudes)
        # Stack everything: shape (n_performers, n_experiments)
        self.sign_counts = np.vstack(self.sign_counts)
        self.magnitudes = np.vstack(self.magnitudes)


class RandomCAV(CAV):
    """Creates CAVs using TWO random datasets, for NHST. Same API as `CAV`."""

    def __init__(
            self,
            n_clips: int,
            model: torch.nn.Module,
            layer: torch.nn.Module,
            layer_attribution: str = "gradient_x_activation",
            multiply_by_inputs: bool = True,
            standardize: bool = False,
            batch_size: int = BATCH_SIZE
    ):
        self.n_clips = n_clips
        super().__init__(
            cav_idx=None,  # means that we'll use any MIDI file to create the dataset
            model=model,
            layer=layer,
            layer_attribution=layer_attribution,
            multiply_by_inputs=multiply_by_inputs,
            standardize=standardize,
            batch_size=batch_size
        )

    def initialise_concept_dataloader(self) -> None:
        """Overrides parent method to avoid creating a concept dataloader unnecessarily"""
        return

    def fit(self, n_experiments: int = N_EXPERIMENTS) -> None:
        """Fits the classifier to two random datasets and gets the CAVs and accuracy scores"""
        # Iterate through all random datasets
        for _ in tqdm(range(n_experiments), desc='Fitting classifiers'):
            # Get one random dataset and compute activations
            rand_dataset_1 = self.initialise_random_dataloader(n_clips=self.n_clips)
            rand_acts_1 = self.get_activations(rand_dataset_1)
            # Get a different random dataset and compute activations
            rand_dataset_2 = self.initialise_random_dataloader(n_clips=len(rand_dataset_1.dataset))
            rand_acts_2 = self.get_activations(rand_dataset_2)
            # Get the CAV and accuracy score for this combination of random datasets
            cav, acc = self.get_cav(rand_acts_1, rand_acts_2)
            # Append everything to a list
            self.acc.append(acc)
            self.cav.append(cav)
        # Log the accuracy to the console, should be about 50%
        logger.info(f'Mean test accuracy for random CAV: {np.mean(self.acc):.5f}, SD {np.std(self.acc):.5f}')


class VoicingLoaderReal(Dataset):
    """Loads voicing MIDIs from unsupervised_resources/voicings/midi_final"""
    SPLIT_PATH = os.path.join(utils.get_project_root(), 'references/_unsupervised_resources/voicings/midi_final')

    def __init__(
            self,
            cav_number: int,
            n_clips: int = None,
            transpose_range: int = TRANSPOSE_RANGE,
            use_rootless: bool = True
    ):
        super().__init__()
        self.transpose_range = transpose_range
        self.use_rootless = use_rootless
        self.n_clips = n_clips
        self.cav_midis = self.get_midi_paths(cav_number)
        self.midis = self.create_midi()

    def get_midi_paths(self, cav_number):
        cav_path = os.path.join(self.SPLIT_PATH, f"{cav_number}_cav")
        assert os.path.isdir(cav_path)
        cavs = [os.path.join(cav_path, i) for i in os.listdir(cav_path) if i.endswith('mid')]
        return cavs

    @staticmethod
    def combine_hands(lh: list[tuple], rh: list[tuple]):
        def sorter(arr: list[tuple]):
            return sorted(arr, key=lambda x: (x[0], x[2]))

        newlh = []
        for key, group in groupby(lh, lambda x: x[0]):
            group = list(group)
            # Sort by pitches
            sort = sorted(group, key=lambda x: x[2])
            # Add everything but the lowest pitch to the right hand
            if len(sort) > 1:
                rh.extend(sort[1:])
            # Keep only the lowest pitch in the left hand
            newlh.append(sort[0])
        # Sort both arrays
        return sorter(newlh), sorter(rh)

    @staticmethod
    def get_inversions(rh: list[tuple]):
        rhand_grp = [list(i[1]) for i in groupby(rh, lambda x: x[0])]
        inverted_rhands = {i: [] for i in range(len(rhand_grp[0]) - 1)}
        for inversion in inverted_rhands.keys():
            for grp in rhand_grp:
                to_invert = sorted(grp, key=lambda x: x[2])[:inversion + 1]
                inverted = [(i[0], i[1], i[2] + 12, i[3]) for i in to_invert]
                combined = inverted + grp[inversion + 1:]
                inverted_rhands[inversion].extend(sorted(combined, key=lambda x: x[2]))
        return [rh, *list(inverted_rhands.values())]

    @staticmethod
    def adjust_durations_and_transpose(arr: list[tuple], transpose_value: int):
        onset_grps = [list(i[1]) for i in groupby(arr, lambda x: x[0])]
        spacer = np.linspace(0, utils.CLIP_LENGTH, len(onset_grps) + 1)
        fmt = []
        for start, end, grp in zip(spacer, spacer[1:], onset_grps):
            fmt_grp = [(start, end, note[2] + transpose_value, note[3]) for note in grp]
            fmt.extend(fmt_grp)
        return fmt

    def _create_midi(self, cav_midi_path: str) -> list[np.array]:
        """Creates MIDI by getting all inversions / transpositions / lh-rh combinations"""
        # Load as pretty MIDI object
        pm = PrettyMIDI(cav_midi_path)
        # Replacing velocity with 1.0
        lh = [(i.start, i.end, i.pitch, 1.0) for i in pm.instruments[1].notes]
        rh = [(i.start, i.end, i.pitch, 1.0) for i in pm.instruments[0].notes]
        # Add upper left hand (non-bass) notes to right hand
        lhand, rhand = self.combine_hands(lh, rh)
        # Get all inversions for right hand chords
        all_rhands = self.get_inversions(rhand)
        # Whether
        all_rolls = []
        # Iterating over all inversions
        for rhands in all_rhands:
            # Iterate over all transposition values
            for transp in range(-self.transpose_range, self.transpose_range + 1):
                # Iterate over all combinations of left-hand/right-hand
                rootless_combs = [True, False] if self.use_rootless else [True]
                for include_lh in rootless_combs:
                    # If we're using the left hand, add both hands together
                    if include_lh:
                        comb = sorted(rhands + lhand, key=lambda x: x[0])
                    # Otherwise, just use the right hand
                    else:
                        comb = sorted(rhands, key=lambda x: x[0])
                    # Adjust the durations to fill the clip
                    adjusted = self.adjust_durations_and_transpose(comb, transp)
                    # Create the piano roll and append
                    all_rolls.append(get_piano_roll(adjusted))
        return all_rolls

    def create_midi(self):
        # Parallel processing doesn't seem to be any faster? But we leave this in so `N_JOBS` can be changed if needed
        with Parallel(n_jobs=N_JOBS, verbose=0) as par:
            midis = par(delayed(self._create_midi)(mid) for mid in self.cav_midis)
            midis = [m for ms in midis for m in ms]
        # midis = [list(self._create_midi(midi)) for midi in self.cav_midis]
        if self.n_clips is not None:
            shuffle(midis)
            midis = midis[:self.n_clips]
        return midis

    def __len__(self):
        return len(self.midis)

    @staticmethod
    def add_masks(midi):
        # Create a mask of zeros with same dimensionality
        masked = np.zeros_like(midi)
        # Stack with masks: M (masked), H (not masked), R (masked), D (masked)
        return np.stack([masked, midi, masked, masked])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        # Get target midi
        targ_midi = self.midis[idx]
        # Add masks for remaining (non-harmony) concepts
        # Return with the class index (i.e., True)
        return self.add_masks(targ_midi), 1


class VoicingLoaderFake(VoicingLoaderReal):
    def __init__(
            self,
            avoid_cav_number: int,
            n_clips: int,
            transpose_range: int = TRANSPOSE_RANGE,
            use_rootless: bool = True
    ):
        super().__init__(
            cav_number=avoid_cav_number,
            n_clips=n_clips,
            transpose_range=transpose_range,
            use_rootless=use_rootless
        )

    def get_midi_paths(self, cav_number):
        # Get folders for all concepts
        # When cav_number is None, we can use any concept to create this random dataset
        if cav_number is None:
            fmt_cav_paths = [os.path.join(self.SPLIT_PATH, i) for i in os.listdir(self.SPLIT_PATH)]
        # When cav_number is an int, we'll avoid using the target concept to create the random dataset
        else:
            fmt_cav_paths = [
                os.path.join(self.SPLIT_PATH, i) for i in os.listdir(self.SPLIT_PATH) if i != f"{cav_number}_cav"
            ]
        # Get midi filepaths for all concepts
        all_cavs = []
        for path in fmt_cav_paths:
            all_cavs.extend([os.path.join(self.SPLIT_PATH, path, i) for i in os.listdir(path) if i.endswith('.mid')])
        # Shuffle all midi filepaths
        shuffle(all_cavs)
        # Only return the first N paths as this will mean we don't have to process all the data
        return all_cavs[:self.n_clips]

    def __len__(self):
        return self.n_clips

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        # Get target midi
        targ_midi = self.midis[idx]
        # Add masks for remaining (non-harmony) concepts
        # Return with the class index (i.e., False)
        return self.add_masks(targ_midi), 0
