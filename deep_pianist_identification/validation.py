#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Validation module. Accepts the same YAML configurations as in `training.py`."""

import os
from itertools import combinations
from time import time

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.training import parse_config_yaml, DEFAULT_CONFIG, get_model, groupby_tracks
from deep_pianist_identification.utils import seed_everything, SEED, DEVICE, N_CLASSES, get_project_root

MASKS = ["melody", "harmony", "rhythm", "dynamics"]
MASK_COMBINATIONS = []
for L in range(len(MASKS) + 1):
    for subset in combinations(range(len(MASKS)), L):
        MASK_COMBINATIONS.append(subset)


class ValidateModule:
    def __init__(self, **kwargs):
        # Set all keyword arguments to class parameters
        self.params = kwargs
        for key, val in self.params.items():
            setattr(self, key, val)
        # MODEL
        logger.debug('Initialising model...')
        logger.debug(f"Using encoder {self.encoder_module} with parameters {self.model_cfg}")
        self.model = get_model(self.encoder_module)(**self.model_cfg).to(DEVICE)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(f'Model parameters: {n_params}')
        # Create the validation dataset loader
        # TODO: replace "split" with "validation" once we have this set
        logger.debug(f'Initialising validation dataloader with batch size {self.batch_size} '
                     f'and parameters {self.test_dataset_cfg}')
        self.validation_dataloader = DataLoader(
            MIDILoader(
                'test',
                **self.test_dataset_cfg  # Use the test dataset config here again
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        # LOSS AND VALIDATION METRICS
        logger.debug('Initialising validation metrics...')
        if self.test_dataset_cfg.get("multichannel", False):
            self.validation_fn = self.validate_multichannel
        else:
            self.validation_fn = self.validate_singlechannel
        self.acc_fn = Accuracy(task="multiclass", num_classes=N_CLASSES).to(DEVICE)
        # CHECKPOINTS
        self.checkpoint_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run)
        self.load_checkpoint()  # Will raise errors when checkpoints cannot be found!

    def load_checkpoint(self) -> None:
        """Load the latest checkpoint for the current experiment and run. Raises errors when no checkpoints found!"""
        # If we haven't created a checkpoint for this run, raise an error: we need a checkpoint to validate
        if not os.path.exists(self.checkpoint_folder):
            raise FileNotFoundError(f'Checkpoint folder {self.checkpoint_folder} does not exist!')
        # Get all the checkpoints for the current experiment/run combination
        checkpoints = [i for i in os.listdir(self.checkpoint_folder) if i.endswith(".pth")]
        if len(checkpoints) == 0:
            raise FileNotFoundError(f'Could not find any checkpoints for this run in {self.checkpoint_folder}!')
        # Sort the checkpoints and load the latest one
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(self.checkpoint_folder, latest_checkpoint)
        # This will raise a warning about possible ACE exploits, but we don't care
        loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        # Set state dictionary for all torch objects
        self.model.load_state_dict(loaded["model_state_dict"], strict=True)
        logger.debug(f'Loaded the checkpoint at {checkpoint_path}!')

    def run_validation(self):
        # Model should always be in evaluation mode during validation
        self.model.eval()
        start = time()
        # Calculate the validation accuracy using the required function
        accuracies = self.validation_fn()
        logger.debug(f'Took {time() - start:.2f} seconds to calculate validation accuracy')
        # Convert validation accuracy (list of dictionaries) to a dataframe
        accuracy_df = pd.DataFrame(accuracies)
        # Save the dataframe inside the checkpoint folder for this run and log time
        save_path = os.path.join(self.checkpoint_folder, "validation.csv")
        accuracy_df.to_csv(save_path, index=False)
        logger.debug(f"Saved validation metrics to {save_path}")

    def validate_multichannel(self) -> list[dict]:
        """Validation for multichannel models: compute accuracy for all combinations of masked concepts"""

        def step(r: torch.tensor, idx_to_mask: list[int]):
            # These are the features for each individual concept
            embeddings = self.model.forward_features(r)
            # Apply the required masks to each embedding and concatenate
            for to_mask in idx_to_mask:
                embeddings[to_mask] = embeddings[to_mask] * torch.zeros_like(embeddings[to_mask])
            features = torch.cat(embeddings, dim=1)
            # Calculate the raw class logits and then apply the softmax
            logits = self.model.forward_pooled(features)
            return torch.nn.functional.softmax(logits, dim=1)

        all_accuracies = []
        with torch.no_grad():
            # Iterate through all indexes to mask: i.e., [0, 1, 2, 3], [0, 1, 2], [1, 2, 3]...
            for combo in MASK_COMBINATIONS:
                # Get the names of the concepts that ARE NOT masked this iteration
                cmasks = '+'.join(mask for i, mask in enumerate(MASKS) if i not in combo)
                combo_clip_accs, combo_track_names, combo_track_preds, combo_track_targets = [], [], [], []
                # Iterate through all clips in the validation dataset
                for roll, targets, tracks in tqdm(self.validation_dataloader, desc=f"Predicting using {cmasks}: "):
                    # Set devices correctly for all tensors
                    roll = roll.to(DEVICE)
                    targets = targets.to(DEVICE)
                    # Step forward with the model, get the class probabilities
                    soft = step(roll, combo)
                    # Get clip level accuracy
                    clip_preds = torch.argmax(soft, dim=1)
                    clip_accuracy = self.acc_fn(clip_preds, targets).item()
                    # Append everything to our list for running totals across the batch
                    combo_clip_accs.append(clip_accuracy)
                    combo_track_names.extend(tracks)
                    combo_track_preds.append(soft)
                    combo_track_targets.append(targets)
                # TODO: create a confusion matrix here
                # TODO: compute decrease in accuracy for individual performers based on masks
                # Group tracks by the name and get the average predictions and target classes
                predictions, targets = groupby_tracks(combo_track_names, combo_track_preds, combo_track_targets)
                # Calculate track-level accuracy and mean clip-level accuracy
                track_acc = self.acc_fn(predictions, targets).item()
                clip_acc = np.mean(combo_clip_accs)
                # Append all the results as a dictionary and log
                all_accuracies.append(dict(model=self.run, concepts=cmasks, clip_acc=clip_acc, track_acc=track_acc))
                logger.info(f"Results for {cmasks}: clip accuracy {clip_acc:.2f}, track accuracy {track_acc:.2f}")
        return all_accuracies

    def validate_singlechannel(self) -> list[dict]:
        """Validation function for single channel (i.e., baseline) models, with no concept of masking"""
        # Lists to track metrics across all batches
        # TODO: why is this not working for old checkpoints? (multichannel is OK)
        clip_accs = []
        track_names, track_preds, track_targets = [], [], []
        with torch.no_grad():
            for roll, targets, tracks in tqdm(self.validation_dataloader, desc=f"Single-channel prediction: "):
                # Move all tensors onto correct device
                roll = roll.to(DEVICE)
                targets = targets.to(DEVICE)
                # Forward in the model, get predicted class and calculate accuracy
                logits = self.model(roll)
                softmaxed = torch.nn.functional.softmax(logits, dim=1)
                predicted = torch.argmax(softmaxed, dim=1)
                clip_accuracy = self.acc_fn(predicted, targets).item()
                # Append everything to our list for running totals across the batch
                clip_accs.append(clip_accuracy)
                track_names.extend(tracks)
                track_preds.append(softmaxed)
                track_targets.append(targets)
        # Group tracks by the name and get the average predictions and target classes
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        # Calculate track-level accuracy and mean clip-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        clip_acc = np.mean(clip_accs)
        # Append all the results as a dictionary and log
        logger.info(f"Results for single channel: clip accuracy {clip_acc:.2f}, track accuracy {track_acc:.2f}")
        return [dict(model=self.run, concepts="singlechannel", clip_acc=clip_acc, track_acc=track_acc)]


if __name__ == "__main__":
    import argparse

    seed_everything(SEED)

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Run model training')
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config file")
    # Adding all arguments from our default configuration, allowing these to be updated
    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    # Parse all arguments from the provided YAML file
    args = vars(parser.parse_args())
    if args['config'] is not None:
        parse_config_yaml(args, raise_on_not_found=True)
        logger.debug(f'Starting validation using config file {args["config"]}')
    else:
        raise FileNotFoundError("Must pass in a valid config file for validation!")
    # Create the ValidationModule and run validation
    vm = ValidateModule(**args)
    vm.run_validation()
