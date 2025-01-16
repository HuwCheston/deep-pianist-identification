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
from deep_pianist_identification.utils import seed_everything, SEED, DEVICE, get_project_root

MASKS = ["melody", "harmony", "rhythm", "dynamics"]
MASK_COMBINATIONS = []
for L in range(len(MASKS) + 1):
    for subset in combinations(range(len(MASKS)), L):
        if len(subset) == len(MASKS):
            continue
        MASK_COMBINATIONS.append(subset)


class ValidateModule:
    def __init__(self, **kwargs):
        # Set all keyword arguments to class parameters
        self.params = kwargs
        for key, val in self.params.items():
            setattr(self, key, val)
        # CLASSIFICATION PROBLEM
        self.classify_dataset = kwargs.get("classify_dataset", False)
        # CLASS MAPPING AND NUMBER OF CLASSES
        self.class_mapping = self.get_class_mapping()
        self.num_classes = len(self.class_mapping.keys())
        logger.debug(f"Model will be evaluated on {self.num_classes} classes!")
        # MODEL
        logger.debug('Initialising model...')
        logger.debug(f"Using encoder {self.encoder_module} with parameters {self.model_cfg}")
        self.model = get_model(self.encoder_module)(num_classes=self.num_classes, **self.model_cfg).to(DEVICE)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(f'Model parameters: {n_params}')
        # Create the validation dataset loader
        logger.debug(f'Initialising validation dataloader with batch size {self.batch_size} '
                     f'and parameters {self.test_dataset_cfg}')
        try:
            self.validation_dataloader = DataLoader(
                MIDILoader(
                    'validation',
                    classify_dataset=self.classify_dataset,
                    data_split_dir=self.data_split_dir,
                    # We can just use the test dataset configuration here
                    **self.test_dataset_cfg
                ),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=remove_bad_clips_from_batch
            )
        except FileNotFoundError:
            raise FileNotFoundError('Validation split not created for this dataset!')
        # LOSS AND VALIDATION METRICS
        logger.debug('Initialising validation metrics...')
        if self.test_dataset_cfg.get("multichannel", False):
            self.validation_fn = self.validate_multichannel
        else:
            self.validation_fn = self.validate_singlechannel
        if self.classify_dataset:
            self.acc_fn = Accuracy(task="binary", num_classes=self.num_classes).to(DEVICE)
            self.confusion_matrix_fn = ConfusionMatrix(
                task="binary", num_classes=self.num_classes, normalize="true"
            ).to(DEVICE)
        else:
            self.acc_fn = Accuracy(task="multiclass", num_classes=self.num_classes).to(DEVICE)
            self.confusion_matrix_fn = ConfusionMatrix(
                task="multiclass",
                num_classes=self.num_classes,
                normalize='true'
            ).to(DEVICE)
        # CHECKPOINTS
        self.checkpoint_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run)
        self.load_checkpoint()  # Will raise errors when checkpoints cannot be found!
        self.baseline_accuracy = None  # Will be filled in later
        self.full_accuracy = None

    def get_class_mapping(self) -> dict:
        if self.classify_dataset:
            return {0: "jtd", 1: "pijama"}
        else:
            csv_loc = os.path.join(
                get_project_root(), 'references/data_splits', self.data_split_dir, 'pianist_mapping.csv'
            )
            pianist_mapping = pd.read_csv(csv_loc, index_col=1)
            return {i: row.iloc[0] for i, row in pianist_mapping.iterrows()}

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
        # Model should always be in evaluation mode during validation
        self.model.eval()
        start = time()
        # Calculate the validation accuracy using the required function
        accuracies = self.validation_fn()
        logger.debug(f'Took {time() - start:.2f} seconds to calculate validation accuracy')
        # Convert validation accuracy (list of dictionaries) to a dataframe
        accuracy_df = pd.DataFrame(accuracies)
        self.save_csv(accuracy_df, "validation.csv")
        # Creating multichannel plots
        if self.validation_fn == self.validate_multichannel:
            lp = plotting.LollipopPlotMaskedConceptsAccuracy(
                accuracy_df, "track_acc", "Accuracy", baseline_accuracy=self.baseline_accuracy
            )
            bp1 = plotting.BarPlotSingleConceptAccuracy(
                accuracy_df, baseline_accuracy=self.baseline_accuracy, full_accuracy=self.full_accuracy
            )
            bp2 = plotting.BarPlotSingleMaskAccuracy(accuracy_df)
            for plot_cls in [bp1, bp2, lp]:
                plot_cls.create_plot()
                plot_cls.save_fig()

    def create_confusion_matrix(self, predictions, targets, concept: str = ""):
        """Creates a confusion matrix for (track-wise) predictions and targets"""
        # Create the confusion matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = self.confusion_matrix_fn(predictions, targets) * 100
        # Convert to a numpy array
        mat = mat.cpu().numpy()
        # Create the plot and save
        title = ', '.join(sorted([i.title() for i in concept.split('+')]))
        cm = plotting.HeatmapConfusionMatrix(mat, pianist_mapping=self.class_mapping, title=title)
        cm.create_plot()
        cm.save_fig()
        # Save the CSV file to inside the confusion matrices folder
        df = pd.DataFrame(mat)
        df.columns = self.class_mapping.values()  # adding column headings
        df.index = self.class_mapping.values()  # adding row headings
        self.save_csv(df, f'confusion_matrices/confusion_matrix_{title}.csv')

    def get_zero_rate_accuracy(self, track_targets: list[torch.tensor], track_names: list[str]) -> float:
        """Get accuracy for a model that just predicts the most frequent class"""
        ground_truth = torch.cat(track_targets)
        # Get the index of the pianist that appears most frequently (should be 2, i.e. Brad Mehldau)
        most_frequent_pianist = torch.mode(ground_truth)[0].item()
        # Create an array filled with 2s with shape (total_tracks)
        total_tracks = len(set(track_names))
        most_common_targets = torch.full((total_tracks,), most_frequent_pianist).to(DEVICE)
        # This code simply returns the target class for each track
        actual_targets = torch.tensor(
            pd.DataFrame([track_names, ground_truth.tolist()])
            .transpose()
            .groupby(0)
            .max()[1]
            .to_numpy()
            .astype(int)
        ).to(DEVICE)
        # Compute the accuracy function
        return self.acc_fn(most_common_targets, actual_targets).item()

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
                # TODO: the DisentangleNet api has changed since this was written. Check to make sure this still works!
                embeddings = self.model.extract_concepts(roll)
                # Iterate through all indexes to mask: i.e., [0, 1, 2, 3], [0, 1, 2], [1, 2, 3]...
                for mask_idxs in MASK_COMBINATIONS:
                    # Mask the required embeddings for this combination
                    new_embeddings = list(apply_masks(deepcopy(embeddings), mask_idxs))
                    # Stack into a tensor, calculate logits, and softmax
                    x = torch.cat(new_embeddings, dim=1)
                    # Self-attention across channels
                    if self.model.use_attention:
                        x, _ = self.model.self_attention(x, x, x)
                    # Permute to (batch, features, channels)
                    x = x.permute(0, 2, 1)
                    # Pool across channels
                    x = self.model.pooling(x)
                    # Remove singleton dimension
                    x = x.squeeze(2)
                    # Pass through linear layers
                    x = self.model.fc1(x)
                    logits = self.model.fc2(x)
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

        self.baseline_accuracy = self.get_zero_rate_accuracy(track_targets, track_names)
        global_accuracies, perclass_accuracies = [], []
        # Iterate through all individual combinations of masks
        sc_names, sc_cms = [], []
        for mask, vals in combo_accuracies.items():
            # Average clip-level accuracy
            clip_acc = np.mean(vals["clip_accuracy"])
            # Group by track names and calculate average track-level accuracy
            p, t = groupby_tracks(track_names, vals["track_predictions"], track_targets)
            track_acc = self.acc_fn(p, t).item()
            if mask == "melody+harmony+rhythm+dynamics":
                self.full_accuracy = track_acc
            # Create confusion matrix
            cm = self.confusion_matrix_fn(p, t)
            # Append the confusion matrix for our single-concept confusion matrix plot
            if mask.count('+') == 0:
                sc_names.append(mask)
                sc_cms.append(cm.cpu().numpy())
            # Calculate perclass accuracy as diagonal of normalized confusion matrix
            perclass_acc = cm.diag().cpu().numpy()
            for class_idx, class_acc in enumerate(perclass_acc):
                class_name = self.class_mapping[class_idx]
                perclass_accuracies.append(dict(model=self.run, concepts=mask, pianist=class_name, class_acc=class_acc))
            # Create the confusion matrix for this combination of parameters
            self.create_confusion_matrix(p, t, mask)
            # Log and store a dictionary for this combination of masks
            logger.info(f"Results for concepts {mask}: clip accuracy {clip_acc:.5f}, track accuracy {track_acc:.5f}")
            logger.info(f"Accuracy loss vs. full model: {1 - (track_acc / self.full_accuracy):.5f}")
            global_accuracies.append(dict(model=self.run, concepts=mask, clip_acc=clip_acc, track_acc=track_acc))
        # Create plot of all single-concept confusion matrices
        hm = plotting.HeatmapConfusionMatrices(sc_cms, sc_names, pianist_mapping=self.class_mapping)
        hm.create_plot()
        hm.save_fig()
        # Create the per-class, per-concept accuracy dataframe and save to a csv
        perclass_acc_df = pd.DataFrame(perclass_accuracies)
        self.save_csv(perclass_acc_df, "class_accuracy.csv")
        # Plot per-class accuracy for each concept
        bp = plotting.BarPlotConceptClassAccuracy(perclass_acc_df)
        bp.create_plot()
        bp.save_fig()
        return global_accuracies

    @staticmethod
    def save_csv(df: pd.DataFrame, name: str):
        # Create the folder if it doesn't exist
        fold = os.path.join(get_project_root(), "reports/figures/ablated_representations")
        if not os.path.isdir(fold):
            os.makedirs(fold)
        # Convert to a DataFrame if not currently one
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Dump to a csv with the required name, adding the file extension if needed
        if not name.endswith('.csv'):
            name = name + '.csv'
        df.to_csv(os.path.join(fold, name), index=False)

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
        self.baseline_accuracy = self.get_zero_rate_accuracy(track_targets, track_names)
        # Group tracks by the name and get the average predictions and target classes
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        # Calculate track-level accuracy and mean clip-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        clip_acc = np.mean(clip_accs)
        # Create the confusion matrix
        self.create_confusion_matrix(predictions, targets)
        # Append all the results as a dictionary and log
        logger.info(f"Results for single channel: clip accuracy {clip_acc:.5f}, track accuracy {track_acc:.5f}")
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
