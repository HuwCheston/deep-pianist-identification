#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

import os
from time import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, ConfusionMatrix
from tqdm import tqdm

from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.encoders import CRNNet, CNNet, DisentangleNet
from deep_pianist_identification.utils import DEVICE, N_CLASSES, seed_everything, get_project_root

# Any key-value pairs we don't define in our custom config will be overwritten using these
DEFAULT_CONFIG = yaml.safe_load(open(os.path.join(get_project_root(), 'config', 'debug_local.yaml')))


class TrainModule:
    def __init__(self, **kwargs):
        # Set all keyword arguments to class parameters
        self.params = kwargs
        for key, val in self.params.items():
            setattr(self, key, val)
        self.current_epoch = 0
        # MODEL
        logger.debug('Initialising model...')
        self.model = self.get_model().to(DEVICE)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(f'Model parameters: {n_params}')
        # DATALOADERS
        logger.debug(f'Initialising training dataloader with batch size {self.batch_size} '
                     f'and parameters {self.train_dataset_cfg}')
        self.train_loader = DataLoader(
            MIDILoader(
                'train',
                **self.train_dataset_cfg
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        logger.debug(f'Initialising test dataloader with batch size {self.batch_size} '
                     f'and parameters {self.test_dataset_cfg}')
        self.test_loader = DataLoader(
            MIDILoader(
                'test',
                **self.test_dataset_cfg
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        # LOSS AND METRICS
        logger.debug('Initialising loss and metrics...')
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
        self.acc_fn = Accuracy(task="multiclass", num_classes=N_CLASSES).to(DEVICE)
        self.confusion_matrix_fn = ConfusionMatrix(
            task="multiclass",
            num_classes=N_CLASSES,
            normalize='true'
        ).to(DEVICE)
        # OPTIMISER
        logger.debug(f'Initialising optimiser with learning rate {self.learning_rate}')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # CHECKPOINTS
        if self.checkpoint_cfg["load_checkpoints"]:
            self.load_checkpoint()

    def get_model(self):
        logger.debug(f"Using encoder {self.encoder_module} with parameters {self.model_cfg}")
        if self.encoder_module == "cnn":
            return CNNet(**self.model_cfg)
        elif self.encoder_module == "crnn":
            return CRNNet(**self.model_cfg)
        elif self.encoder_module == "disentangle":
            assert all((
                self.train_dataset_cfg.get("multichannel") is True,
                self.test_dataset_cfg.get("multichannel") is True,
            )), "Must set `multichannel == True` in dataloader config when using disentangle encoder!"
            return DisentangleNet(**self.model_cfg)

    def step(self, features: torch.tensor, targets: torch.tensor) -> tuple:
        # Set device correctly for both features and targets
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = self.model(features)
        # No need to softmax beforehand, `nn.CrossEntropyLoss` does this automatically
        loss = self.loss_fn(logits, targets)
        softmaxed = nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(softmaxed, dim=1)
        # Compute clip level metrics
        acc = self.acc_fn(predictions, targets)
        return loss, acc, softmaxed

    @staticmethod
    def groupby_tracks(
            track_names: list[str],
            batched_track_predictions: list[torch.tensor],
            batched_track_targets: list[torch.tensor]
    ) -> tuple[torch.tensor, torch.tensor]:
        """Groups clip predictions by source track, averages class probabilities, returns predicted and actual class"""
        # Create a DataFrame: one row per clip, one column per class predicted
        pred_df = pd.DataFrame(torch.cat(batched_track_predictions).tolist())
        # Set columns containing the names of each track and the target class
        pred_df['track_name'] = track_names
        pred_df['ground_truth'] = torch.cat(batched_track_targets).tolist()
        # Group by underlying track name and average the class probabilities across all clips from one track
        grouped = pred_df.groupby(['track_name', 'ground_truth']).mean()
        # Extract the predicted and target class indexes
        pred_class = np.argmax(grouped.to_numpy(), axis=1)
        target_class = grouped.index.get_level_values(1).to_numpy()
        # Convert predicted and target class indexes to tensors and return on the correct device
        return torch.tensor(pred_class, device=DEVICE, dtype=int), torch.tensor(target_class, device=DEVICE, dtype=int)

    def training(self, epoch_num: int) -> tuple[float, float, float]:
        # Lists to track metrics across all batches
        clip_losses, clip_accs = [], []
        track_names, track_preds, track_targets = [], [], []
        # Set model mode correctly
        self.model.train()
        # Iterate through all batches, with a nice TQDM counter
        for features, targets, batch_names in tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f'Training, epoch {epoch_num} / {self.epochs}...'
        ):
            # Forwards pass
            loss, acc, preds = self.step(features, targets)
            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            # Optimizer step
            self.optimizer.step()
            # Append all clip-level metrics from this batch
            clip_losses.append(loss.item())
            clip_accs.append(acc.item())
            # Append the names of the tracks and the corresponding predictions for track-level metrics
            track_names.extend(batch_names)  # Single list of strings
            track_preds.append(preds)  # List of tensors, one tensor per clip, containing class logits
            track_targets.append(targets)  # List of tensors, one tensor per batch, containing target classes
        # Group by tracks, average probabilities, and get predicted and target classes
        predictions, targets = self.groupby_tracks(track_names, track_preds, track_targets)
        # Compute track-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        # Return clip loss + accuracy, track accuracy
        return np.mean(clip_losses), np.mean(clip_accs), track_acc

    def testing(self, epoch_num: int) -> tuple[float, float, float]:
        # Lists to track metrics across all batches
        clip_losses, clip_accs = [], []
        track_names, track_preds, track_targets = [], [], []
        # Set model mode correctly
        self.model.eval()
        # Iterate through all batches, with a nice TQDM counter
        with torch.no_grad():
            for features, targets, batch_names in tqdm(
                    self.test_loader,
                    total=len(self.test_loader),
                    desc=f'Testing, epoch {epoch_num} / {self.epochs}...'
            ):
                # Forwards pass
                loss, acc, preds = self.step(features, targets)
                # No backwards pass, just append all clip-level metrics from this batch
                clip_losses.append(loss.item())
                clip_accs.append(acc.item())
                # Append the names of the tracks and the corresponding predictions for track-level metrics
                track_names.extend(batch_names)  # Single list of strings
                track_preds.append(preds)  # List of tensors, one tensor per clip, containing class logits
                track_targets.append(targets)  # List of tensors, one tensor per batch, containing target classes
        # Group by tracks, average probabilities, and get predicted and target classes
        predictions, targets = self.groupby_tracks(track_names, track_preds, track_targets)
        # Compute track-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        # Return clip loss + accuracy, track accuracy
        return np.mean(clip_losses), np.mean(clip_accs), track_acc

    def load_checkpoint(self) -> None:
        """Load the latest checkpoint for the current experiment and run"""
        checkpoint_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run)
        # If we haven't created a checkpoint for this run, skip loading and train from scratch
        if not os.path.exists(checkpoint_folder):
            logger.warning('Checkpoint folder does not exist for this experiment/run, skipping load!')
            return

        # Get all the checkpoints for the current experiment/run combination
        checkpoints = os.listdir(checkpoint_folder)
        if len(checkpoints) == 0:
            logger.warning('No checkpoints have been created yet for this experiment/run, skipping load!')
            return

        # Sort the checkpoints and load the latest one
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = f'{checkpoint_folder}/{latest_checkpoint}'
        # This will raise a warning about possible ACE exploits, but we don't care
        loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        # Set state dictionary for all torch objects
        self.model.load_state_dict(loaded["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        # Increment epoch by 1
        self.current_epoch = loaded["epoch"] + 1
        logger.debug(f'Loaded the checkpoint at {checkpoint_path}!')

    def save_checkpoint(self, metrics) -> None:
        """Save the model checkpoint at the current epoch"""
        checkpoint_after = self.checkpoint_cfg.get("checkpoint_after_n_epochs", 1)
        # We always want to checkpoint on the final epoch!
        if (self.current_epoch % checkpoint_after == 0) or (self.current_epoch + 1 == self.epochs):
            # Get the folder of checkpoints for the current experiment/run, and create if it doesn't exist
            checkpoint_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder, exist_ok=True)
            # Save everything, including the metrics, state dictionaries, and current epoch
            torch.save(
                dict(
                    **metrics,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    epoch=self.current_epoch
                ),
                os.path.join(checkpoint_folder, f'checkpoint_{str(self.current_epoch).zfill(3)}.pth'),
            )
            logger.debug(f'Saved a checkpoint to {checkpoint_folder}')

    def run_training(self) -> None:
        # Log parameters in mlflow
        for key, val in self.params.items():
            mlflow.log_param(key, val)
        # Start training
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time()
            # Training
            train_loss, train_acc_clip, train_acc_track = self.training(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, training finished: '
                         f'loss {train_loss:.3f}, '
                         f'accuracy (clip) {train_acc_clip:.3f}, '
                         f'accuracy (track) {train_acc_track:.3f}')
            # Testing
            test_loss, test_acc_clip, test_acc_track = self.testing(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, testing finished: '
                         f'loss {test_loss:.3f}, '
                         f'accuracy (clip) {test_acc_clip:.3f}, '
                         f'accuracy (track) {test_acc_track:.3f}')
            # Log parameters from this epoch in MLFlow
            metrics = dict(
                epoch_time=time() - epoch_start,
                train_loss=train_loss,
                train_acc=train_acc_clip,
                train_acc_track=train_acc_track,
                test_loss=test_loss,
                test_acc=test_acc_clip,
                test_acc_track=test_acc_track,
            )
            # Checkpoint the run, if we need to
            if self.checkpoint_cfg["save_checkpoints"]:
                self.save_checkpoint(metrics)
            # Report results to MLFlow, if we're using this
            if self.mlflow_cfg["use"]:
                mlflow.log_metrics(metrics, step=epoch)


def parse_config_yaml(arg_parser) -> None:
    try:
        with open(os.path.join(get_project_root(), 'config', arg_parser['config'])) as stream:
            cfg = yaml.safe_load(stream)
    except FileNotFoundError:
        logger.warning(f'Config file {arg_parser["config"]} could not be found, falling back to defaults...')
    else:
        logger.debug(f'Starting training using config file {arg_parser["config"]}')
        for key, val in cfg.items():
            arg_parser[key] = val


if __name__ == "__main__":
    import yaml
    import argparse

    seed_everything()

    parser = argparse.ArgumentParser(description='Run model training')
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config file")
    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    args = vars(parser.parse_args())
    if args['config'] is not None:
        parse_config_yaml(args)

    tm = TrainModule(**args)
    if args["mlflow_cfg"]["use"]:
        mlflow.set_tracking_uri(uri=args["mlflow_cfg"]["tracking_uri"])
        mlflow.set_experiment(args["experiment"])
        with mlflow.start_run(run_name=args["run"]):
            tm.run_training()
    else:
        tm.run_training()
