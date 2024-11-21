#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility modules (dataloaders, etc.) for generating CAVs using MIDI from textbooks"""

import os
from itertools import groupby
from random import shuffle
from typing import Callable

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
from joblib import Parallel, delayed
from loguru import logger
from pretty_midi import PrettyMIDI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import get_piano_roll

__all__ = ["VoicingLoaderReal", "VoicingLoaderFake", "ConceptExplainer"]

SCALER = StandardScaler()
BATCH_SIZE = 10
TRANSPOSE_RANGE = 6  # Transpose exercise MIDI files +/- 6 semitones (up and down a tri-tone)


def get_attribution_fn(attr_fn_str: str) -> Callable:
    accept = ['integrated_gradients', 'gradient_x_activation']
    assert attr_fn_str in accept, f'{attr_fn_str} not in {", ".join(accept)}'
    if attr_fn_str == 'integrated_gradients':
        return LayerIntegratedGradients
    else:
        return LayerGradientXActivation


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
        with Parallel(n_jobs=-1, verbose=1) as par:
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
        fmt_cav_paths = [
            os.path.join(self.SPLIT_PATH, i) for i in os.listdir(self.SPLIT_PATH) if i != f"{cav_number}_cav"
        ]
        all_cavs = []
        for path in fmt_cav_paths:
            all_cavs.extend([os.path.join(self.SPLIT_PATH, path, i) for i in os.listdir(path) if i.endswith('.mid')])
        shuffle(all_cavs)
        return all_cavs

    def __len__(self):
        return self.n_clips

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        # Get target midi
        targ_midi = self.midis[idx]
        # Add masks for remaining (non-harmony) concepts
        # Return with the class index (i.e., True)
        return self.add_masks(targ_midi), 1


class ConceptExplainer:
    TEST_SIZE = 1 / 3  # same as initial TCAV implementation

    def __init__(
            self,
            cav_idx: int,
            model: torch.nn.Module,
            layer: torch.nn.Module,
            layer_attribution: str = "gradient_x_activation",
            multiply_by_inputs: bool = True,
            n_clips: int = None,
            transpose_range: int = TRANSPOSE_RANGE,
            standardize: bool = False
    ):
        # Create the dataloaders for the desired CAV
        self.cav_idx = cav_idx
        self.real = DataLoader(
            VoicingLoaderReal(
                self.cav_idx,
                n_clips=n_clips,
                transpose_range=transpose_range
            ),
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=False
        )
        self.fake = DataLoader(
            VoicingLoaderFake(
                self.cav_idx,
                n_clips=len(self.real.dataset),
                transpose_range=transpose_range
            ),
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=False
        )
        assert len(self.real.dataset) == len(self.fake.dataset)
        self.layer = layer
        self.model = model
        attr_fn = get_attribution_fn(layer_attribution)
        self.layer_attribution = attr_fn(self.model, self.layer, multiply_by_inputs=multiply_by_inputs)
        self.predictor = LogisticRegression(random_state=utils.SEED, max_iter=10000)
        self.standardize = standardize
        # All of these variables will be updated when creating the CAV
        self.cav = None
        self.sens = None
        self.magnitudes = {}
        self.sign_counts = {}

    def get_embeds(self):
        acts = {}

        def hooky(_, __, out):
            acts['values'] = out

        # Register the handle to get the activations from the desired layer
        handle = self.layer.register_forward_hook(hooky)
        real_embeds, fake_embeds = [], []
        for (real_batch, _), (fake_batch, _) in zip(self.real, self.fake):
            # real_batch = (B, 4, W, H)
            # Create both sets of embeddings
            with torch.no_grad():
                # Through the model
                _ = self.model(real_batch.to(utils.DEVICE))
                # Get the activations from the hook
                # Captum uses the unpooled output, flattened (i.e., C * W * H), so we do too
                real_embed = acts['values'].flatten(start_dim=1)
                acts = {}  # for safety, clear this dictionary
                # Append to the list
                real_embeds.append(real_embed.cpu().numpy())

                # Repeat for the 'fake' (random) dataset
                _ = self.model(fake_batch.to(utils.DEVICE))
                fake_embed = acts['values'].flatten(start_dim=1)
                fake_embeds.append(fake_embed.cpu().numpy())
                acts = {}  # for safety, clear this dictionary
        # Remove the hook
        handle.remove()
        # Concatenate the embeddings
        return np.concatenate(real_embeds), np.concatenate(fake_embeds)

    def fit(self):
        # Get embeddings from the model
        real_embeds, fake_embeds = self.get_embeds()
        # Create arrays of targets with same shape as embeddings
        real_targets, fake_targets = np.ones(real_embeds.shape[0]), np.zeros(fake_embeds.shape[0])
        # Combine both real/fake features and targets into single matrix
        x = np.vstack([real_embeds, fake_embeds])
        y = np.hstack([real_targets, fake_targets])
        # Scale the features if required; not used in initial TCAV implementation
        if self.standardize:
            x = SCALER.fit_transform(x)
        # Split into train-test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.TEST_SIZE, stratify=y)
        # Log dimensionality of all inputs
        logger.info(f'Shape of training inputs: X = {x_train.shape}, y = {y_train.shape}')
        logger.info(f'Shape of testing inputs: X = {x_test.shape}, y = {y_test.shape}')
        # Fit the model and predict the test set
        self.predictor.fit(x_train, y_train)
        y_pred = self.predictor.predict(x_test)
        # Get test set accuracy and log
        acc = accuracy_score(y_test, y_pred)
        logger.info(f'Test accuracy for CAV {self.cav_idx}: {acc:.5f}')
        # Extract coefficients - these are our CAVs
        self.cav = torch.tensor(self.predictor.coef_.astype(np.float32)).flatten()

    def compute_sensitivity(self, feat: np.array, targ: np.array) -> float:
        # Copy the tensor, move to GPU, and add batch dimension in
        inputs = feat.detach().clone().to(utils.DEVICE).unsqueeze(0)
        # Apply masks to every roll other than the harmony roll
        for mask_idx in [0, 2, 3]:
            inputs[:, mask_idx, :, :] = torch.zeros_like(inputs[:, mask_idx, :, :])
        # We don't seem to require setting inputs.requires_grad_ = True
        # Compute layer activations for final convolutional layer of harmony concept WRT target
        acts = self.layer_attribution.attribute(inputs, targ.item())
        # Flatten to get C * W * H (captum does this)
        flatted = acts.flatten()  # only using one element batches, so no need for start_dim
        # Compute dot product against the target CAV
        dotted = torch.dot(flatted.cpu(), self.cav).item()
        return dotted

    @staticmethod
    def get_magnitude(tcav: torch.tensor) -> float:
        return torch.sum(torch.abs(tcav * (tcav > 0.0).float()), dim=0) / torch.sum(torch.abs(tcav), dim=0).item()

    @staticmethod
    def get_sign_count(tcav: torch.tensor) -> float:
        return (torch.sum(tcav > 0.0, dim=0).float() / tcav.shape[0]).item()

    def interpret(self, features: np.array, targets: np.array, class_mapping: dict):
        # Compute sensitivity for all clips: shape (N_clips, N_performers)
        self.sens = np.array([
            self.compute_sensitivity(f, t)
            for f, t in tqdm(
                zip(features, targets),
                total=len(targets),
                desc=f'Computing sensitivity for CAV {self.cav_idx}'
            )
        ])

        all_signs, all_mags = [], []
        # Iterate over each class idx
        for class_idx in class_mapping:
            # Get idxs of clips by this performer
            clip_idxs = torch.argwhere(targets == class_idx)
            # Get the CAV sensitivity values for the corresponding clips
            tcav_score = torch.tensor(self.sens[clip_idxs].flatten())
            # Get the sign count and magnitude for this performer and CAV combination
            all_signs.append(self.get_sign_count(tcav_score))
            all_mags.append(self.get_magnitude(tcav_score))
        # Convert all sign counts and magnitudes to a 1D vector, shape (N_performers)
        self.sign_counts = np.array(all_signs)
        self.magnitudes = np.array(all_mags)
