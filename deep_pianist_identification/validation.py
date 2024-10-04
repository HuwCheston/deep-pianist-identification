#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Validation module. Accepts the same YAML configurations as in `training.py`"""

import os
import warnings
from copy import deepcopy
from itertools import combinations
from time import time

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
from tqdm import tqdm

import deep_pianist_identification.plotting as plotting
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
        self.confusion_matrix_fn = ConfusionMatrix(
            task="multiclass",
            num_classes=N_CLASSES,
            normalize='true'
        ).to(DEVICE)
        # CHECKPOINTS
        self.checkpoint_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run)
        self.load_checkpoint()  # Will raise errors when checkpoints cannot be found!

    def load_checkpoint(self) -> None:
        """Load the latest checkpoint for the current experiment and run. Raises errors when no checkpoints found"""
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

    def run_validation(self) -> None:
        """Runs the required validation function and saves outputs (plots, csv files)"""
        # Create the folder inside the checkpoints directory to save the output in
        save_path = os.path.join(self.checkpoint_folder, "validation")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # Model should always be in evaluation mode during validation
        self.model.eval()
        start = time()
        # Calculate the validation accuracy using the required function
        accuracies = self.validation_fn()
        logger.debug(f'Took {time() - start:.2f} seconds to calculate validation accuracy')
        # Convert validation accuracy (list of dictionaries) to a dataframe
        accuracy_df = pd.DataFrame(accuracies)
        # Save the dataframe inside the checkpoint folder for this run and log time
        accuracy_df.to_csv(os.path.join(save_path, "validation.csv"), index=False)
        logger.debug(f"Saved validation metrics to {save_path}")
        # Save masked concept accuracy barplot
        if self.validation_fn == self.validate_multichannel:
            bp = plotting.BarPlotMaskedConceptsAccuracy(accuracy_df)
            bp.create_plot()
            bp.fig.savefig(os.path.join(save_path, "barplot_concepts_accuracy.png"), **plotting.SAVE_KWS)

    def create_confusion_matrix(self, predictions, targets, concept: str = ""):
        """Creates a confusion matrix for (track-wise) predictions and targets"""
        # Create the confusion matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = self.confusion_matrix_fn(predictions, targets) * 100
        # Convert to a numpy array
        mat = mat.cpu().numpy()
        # Create the plot
        cm = plotting.HeatmapConfusionMatrix(mat)
        cm.create_plot()
        # Set the title to the concepts we've used, nicely formatted
        cm.ax.set_title(', '.join(sorted([i.title() for i in concept.split('+')])))
        # Save inside the checkpoints directory
        fname = "confusion_matrix" + concept + ".png" if concept != "" else "confusion_matrix.png"
        cm.fig.savefig(os.path.join(self.checkpoint_folder, "validation", fname), **plotting.SAVE_KWS)

    def validate_multichannel(self) -> list[dict]:
        """Validation for multichannel models: compute accuracy for all combinations of masked concepts"""

        def apply_masks(embedding_list: list[torch.tensor], idx_to_mask: list[int]) -> torch.tensor:
            """For a given list of tensors, mask those contained in `idx_to_mask` with zeros"""
            for embedding_idx in range(len(embedding_list)):
                embed = embedding_list[embedding_idx]
                if embedding_idx in idx_to_mask:
                    embed *= torch.zeros_like(embed)
                yield embed

        combo_accuracies = {}
        track_names, track_targets = [], []
        with torch.no_grad():
            for roll, targets, tracks in tqdm(self.validation_dataloader, desc=f"Multi-channel Prediction: "):
                # Set devices correctly for all tensors
                roll = roll.to(DEVICE)
                targets = targets.to(DEVICE)
                # Append the track names and targets to the list
                track_names.extend(tracks)
                track_targets.append(targets)
                # Compute the individual embeddings
                embeddings = self.model.forward_features(roll)
                # Iterate through all indexes to mask: i.e., [0, 1, 2, 3], [0, 1, 2], [1, 2, 3]...
                for mask_idxs in MASK_COMBINATIONS:
                    # Mask the required embeddings for this combination
                    new_embeddings = list(apply_masks(deepcopy(embeddings), mask_idxs))
                    # Stack into a tensor, calculate logits, and softmax
                    stacked = torch.cat(new_embeddings, dim=1)
                    logits = self.model.forward_pooled(stacked)
                    softmaxed = torch.nn.functional.softmax(logits, dim=1)
                    # Get clip level accuracy
                    combo_preds = torch.argmax(softmaxed, dim=1)
                    combo_accuracy = self.acc_fn(combo_preds, targets).item()
                    # Get the names of the concepts that ARE NOT masked this iteration
                    cmasks = '+'.join(mask for i, mask in enumerate(MASKS) if i not in mask_idxs)
                    if cmasks not in combo_accuracies.keys():
                        combo_accuracies[cmasks] = {"clip_accuracy": [], "track_predictions": []}
                    # Append clip accuracy and predictions to the dictionary for this combination of masks
                    combo_accuracies[cmasks]["clip_accuracy"].append(combo_accuracy)
                    combo_accuracies[cmasks]["track_predictions"].append(softmaxed)

        all_accuracies = []
        # Iterate through all individual combinations of masks
        for mask, vals in combo_accuracies.items():
            # Average clip-level accuracy
            clip_acc = np.mean(vals["clip_accuracy"])
            # Group by track names and calculate average track-level accuracy
            p, t = groupby_tracks(track_names, vals["track_predictions"], track_targets)
            track_acc = self.acc_fn(p, t).item()
            # Create the confusion matrix for this combination of parameters
            self.create_confusion_matrix(p, t, mask)
            # Log and store a dictionary for this combination of masks
            logger.info(f"Results for concepts {mask}: clip accuracy {clip_acc:.2f}, track accuracy {track_acc:.2f}")
            all_accuracies.append(dict(model=self.run, concepts=mask, clip_acc=clip_acc, track_acc=track_acc))
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
        # Create the confusion matrix
        self.create_confusion_matrix(predictions, targets)
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
