#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Replicate Kong et al. (2020) classical composer classification from GiantMIDI dataset using our implementation"""

import os
from time import time

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.dataloader import remove_bad_clips_from_batch
from deep_pianist_identification.encoders import CRNNet
from deep_pianist_identification.extractors import RollExtractor, normalize_array, ExtractorError
from deep_pianist_identification.training import groupby_tracks, get_tracking_uri

# Dataset constants
CSV_LOC = os.path.join(utils.get_project_root(), 'references/giant-midi/giant-midi_split.csv')
DATA_LOC = os.path.join(utils.get_project_root(), 'references/giant-midi')
GM_DF = pd.read_csv(CSV_LOC, index_col=0)
COMPOSER_MAPPING = {k: n for n, k in enumerate(sorted(GM_DF['composer'].unique().tolist()))}
INDEX_MAPPING = {n: k for k, n in COMPOSER_MAPPING.items()}

# Set a few variables based on the initial paper
BATCH_SIZE = 16
LR = 0.001  # Possibly 0.0001 instead?
EPOCHS = 100  # A reasonable value given this isn't mentioned in the paper
NUM_CLASSES = 10  # We're not replicating the 100-class results here

# Used for mlflow
EXPERIMENT_NAME = "kong2020-replication"
CHECKPOINT_AFTER = 10  # Checkpoint after this many epochs


def get_checkpoint_dir() -> str:
    """Attempts to get the location to store checkpoints based on system hostname"""
    import socket

    hostname = socket.gethostname().lower()
    # Job is running locally: don't save checkpoints
    if "desktop" in hostname:
        return None
    # Job is running on the department server, save to root directory
    elif "musix" in hostname:
        return os.path.join(utils.get_project_root(), "checkpoints")
    # Job is running on HPC, save to external drive
    else:
        return "/rds/user/hwc31/hpc-work/deep-pianist-identification/checkpoints"


class KongLoader(Dataset):
    """Adapted dataloader to load classical performances from Giant-MIDI dataset"""

    def __init__(
            self,
            split: str,
            normalize_velocity: bool = True,
            n_clips: int = None
    ):
        self.split = split
        self.split_df = GM_DF[GM_DF['split'] == split]
        self.clips = self.get_split_clips()
        self.normalize_velocity = normalize_velocity
        if n_clips is not None:
            self.clips = self.clips[:n_clips]

    @staticmethod
    def get_clips(duration: int):
        nclips = []
        for clip_start in range(0, int(duration), utils.HOP_SIZE // 2):
            clip_end = clip_start + utils.CLIP_LENGTH
            if clip_end >= duration:
                break
            nclips.append((clip_start, clip_end))
        return nclips

    def get_split_clips(self) -> list[tuple]:
        res = []
        # Iterate through every source track
        for _, track in self.split_df.iterrows():
            # Get the start and end point for each clip
            # We just use a 50% overlap between clips by default
            n_clips = self.get_clips(track['duration'])
            # Append a tuple for every clip for this source track
            for cs, ce in n_clips:
                res.append((track['track'], track['composer'], cs, ce))
        return res

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple:
        # Unpack the tuple
        clip_name, composer, clip_start, _ = self.clips[idx]
        # Get the index of the composer from the mapping
        composer_mapped = COMPOSER_MAPPING[composer]
        # Create the pretty MIDI object and convert to a piano roll
        pm_obj = PrettyMIDI(os.path.join(DATA_LOC, clip_name))
        # Extract the piano roll
        try:
            roll = RollExtractor(pm_obj, clip_start=clip_start).roll
        # Will be caught and removed in dataloader
        except ExtractorError:
            logger.error(f'Failed to create piano roll for {clip_name} at start {clip_start}, skipping!')
            return None
        # Normalize the array if required
        if self.normalize_velocity:
            roll = normalize_array(roll)
        # Add an extra dimension to the piano roll
        added = np.expand_dims(roll, axis=0)
        # As with the main dataloader, we return the piano roll, the target composer, and the overall track (for agg)
        return added, composer_mapped, clip_name


class KongTrainer:
    """Adapted training module for 10-way classical composer identification to validate our implementation"""

    def __init__(
            self,
            run_name: str,
            use_checkpoints: bool = True,
            use_mlflow: bool = True,
    ):
        self.run_name = run_name
        self.use_checkpoints = use_checkpoints
        self.use_mlflow = use_mlflow
        # Initialise the model and set device correctly
        logger.debug('Initializing model')
        self.model = CRNNet(num_classes=NUM_CLASSES, pool_type='max', norm_type='bn').to(utils.DEVICE)
        # Initialise all the dataloaders for each split
        logger.debug('Initializing dataloaders')
        self.train_loader = DataLoader(
            KongLoader('train', normalize_velocity=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        self.test_loader = DataLoader(
            KongLoader('test', normalize_velocity=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        self.valid_loader = DataLoader(
            KongLoader('valid', normalize_velocity=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        # Initialise loss and accuracy metrics
        logger.debug('Initializing metrics')
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(utils.DEVICE)
        self.acc_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(utils.DEVICE)
        # Initialise optimizer
        logger.debug('Initializing optimizer')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        # Set current epoch: will be updated if we load a checkpoint
        self.current_epoch = 0
        # Get the directory to save a checkpoint at
        self.checkpoint_dir = get_checkpoint_dir()
        # If we're running locally, never save a checkpoint
        if self.checkpoint_dir is None:
            self.use_checkpoints = False
        # Load the most recent checkpoint for this run if we want to do this
        if self.use_checkpoints:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load the latest checkpoint for the current experiment and run"""
        run_dir = os.path.join(self.checkpoint_dir, EXPERIMENT_NAME, self.run_name)
        # If we haven't created a checkpoint for this run, skip loading and train from scratch
        if not os.path.exists(run_dir):
            logger.warning('Checkpoint folder does not exist for this experiment/run, skipping load!')
            return

        # Get all the checkpoints for the current experiment/run combination
        checkpoints = [i for i in os.listdir(run_dir) if i.endswith(".pth")]
        if len(checkpoints) == 0:
            logger.warning('No checkpoints have been created yet for this experiment/run, skipping load!')
            return

        # Sort the checkpoints and load the latest one
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(run_dir, latest_checkpoint)
        # This will raise a warning about possible ACE exploits, but we don't care
        loaded = torch.load(checkpoint_path, map_location=utils.DEVICE, weights_only=False)
        # Set state dictionary for all torch objects
        self.model.load_state_dict(loaded["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        # Increment epoch by 1 so we start in the correct place
        self.current_epoch = loaded["epoch"] + 1
        logger.debug(f'Loaded the checkpoint at {checkpoint_path}!')
        # Re-shuffle the random seed, so we don't just use the seed from epoch 1 again
        #  This probably won't make a difference as we're not using augmentation for this experiment, however
        utils.seed_everything(utils.SEED * self.current_epoch)

    def save_checkpoint(self, metrics: dict) -> None:
        """Save the model checkpoint at the current epoch to disk"""
        # The name of the checkpoint and where it'll be saved
        new_check_name = f'checkpoint_{str(self.current_epoch).zfill(3)}.pth'
        run_folder = os.path.join(self.checkpoint_dir, EXPERIMENT_NAME, self.run_name)
        # We always want to checkpoint on the final epoch!
        if (self.current_epoch % CHECKPOINT_AFTER == 0) or (self.current_epoch + 1 == EPOCHS):
            # Get the folder of checkpoints for the current experiment/run, and create if it doesn't exist
            if not os.path.exists(run_folder):
                os.makedirs(run_folder, exist_ok=True)
            # Save everything, including the metrics, state dictionaries, and current epoch
            torch.save(
                dict(
                    **metrics,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),  # No need for scheduler here
                    epoch=self.current_epoch
                ),
                os.path.join(run_folder, new_check_name),
            )
            logger.debug(f'Saved a checkpoint to {run_folder}')

    def step(self, features: torch.tensor, targets: torch.tensor) -> tuple:
        """Step forward in model and compute loss for given features and targets"""
        # Set device correctly
        features = features.to(utils.DEVICE)
        targets = targets.to(utils.DEVICE)
        # Compute logit through forward pass in model
        logits = self.model(features)
        # No need to softmax beforehand, `nn.CrossEntropyLoss` does this automatically
        loss = self.loss_fn(logits, targets)
        # Apply softmax and argmax to get predicted class
        softmaxed = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(softmaxed, dim=1)
        # Compute clip level metrics
        acc = self.acc_fn(predictions, targets)
        # Return loss, clip-level accuracy, and clip predictions
        return loss, acc, softmaxed

    def training(self) -> tuple[float, float, float]:
        # Lists to track metrics across all batches
        clip_losses, clip_accs = [], []
        track_names, track_preds, track_targets = [], [], []
        # Set model mode correctly
        self.model.train()
        # Iterate through all batches, with a nice TQDM counter
        for rolls, composers, tracks in tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f'Training, epoch {self.current_epoch} / {EPOCHS}...'
        ):
            # Forwards pass
            loss, acc, preds = self.step(rolls, composers)
            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            # Optimizer step
            self.optimizer.step()
            # Append all clip-level metrics from this batch
            clip_losses.append(loss.item())
            clip_accs.append(acc.item())
            # Append the names of the tracks and the corresponding predictions for track-level metrics
            track_names.extend(tracks)  # Single list of strings
            track_preds.append(preds)  # List of tensors, one tensor per clip, containing class logit
            track_targets.append(composers)  # List of tensors, one tensor per batch, containing target classes
        # Group by tracks, average probabilities, and get predicted and target classes
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        track_acc = self.acc_fn(predictions, targets).item()
        # Return clip loss + accuracy, track accuracy
        return np.mean(clip_losses), np.mean(clip_accs), track_acc

    def evaluation(self, loader: DataLoader):
        """Runs evaluation with provided dataloader"""
        # Lists to track metrics across all batches
        clip_losses, clip_accs = [], []
        track_names, track_preds, track_targets = [], [], []
        # Set model mode correctly
        self.model.eval()
        # Iterate through all batches, with a nice TQDM counter
        with torch.no_grad():
            # Iterate through all batches, with a nice TQDM counter
            for rolls, composers, tracks in tqdm(
                    loader,
                    total=len(loader),
                    desc=f'Testing, epoch {self.current_epoch} / {EPOCHS}...'
            ):
                # Forwards pass
                loss, acc, preds = self.step(rolls, composers)
                # No backwards pass, just append all clip-level metrics from this batch
                clip_losses.append(loss.item())
                clip_accs.append(acc.item())
                # Append the names of the tracks and the corresponding predictions for track-level metrics
                track_names.extend(tracks)  # Single list of strings
                track_preds.append(preds)  # List of tensors, one tensor per clip, containing class logit
                track_targets.append(composers)  # List of tensors, one tensor per batch, containing target classes
        # Group by tracks, average probabilities, and get predicted and target classes
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        # Compute track-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        # Return clip loss + accuracy, track accuracy
        return np.mean(clip_losses), np.mean(clip_accs), track_acc

    def run_training(self):
        training_start = time()
        # Start training
        for epoch in range(self.current_epoch, EPOCHS):
            # Start the timer
            epoch_start = time()
            # Increment the current epoch
            self.current_epoch = epoch
            # Training
            train_loss, train_acc_clip, train_acc_track = self.training()
            logger.debug(f'Epoch {epoch} / {EPOCHS}, training finished: '
                         f'loss {train_loss:.3f}, '
                         f'accuracy (clip) {train_acc_clip:.3f}, '
                         f'accuracy (track) {train_acc_track:.3f}')
            # Testing
            test_loss, test_acc_clip, test_acc_track = self.evaluation(self.test_loader)
            logger.debug(f'Epoch {epoch} / {EPOCHS}, testing finished: '
                         f'loss {test_loss:.3f}, '
                         f'accuracy (clip) {test_acc_clip:.3f}, '
                         f'accuracy (track) {test_acc_track:.3f}')
            # Log parameters from this epoch in MLFlow if required
            metrics = dict(
                epoch_time=time() - epoch_start,
                train_loss=train_loss,
                train_acc=train_acc_clip,
                train_acc_track=train_acc_track,
                test_loss=test_loss,
                test_acc=test_acc_clip,
                test_acc_track=test_acc_track,
            )
            if self.use_mlflow:
                mlflow.log_metrics(metrics, step=epoch)
            # Save the checkpoint for the current epoch, if required
            if self.use_checkpoints:
                self.save_checkpoint(metrics)
        logger.info('Training complete! Beginning validation...')
        valid_loss, valid_acc_clip, valid_acc_track = self.evaluation(self.valid_loader)
        logger.debug(f'Validation finished: '
                     f'loss {valid_loss:.3f}, '
                     f'accuracy (clip) {valid_acc_clip:.3f}, '
                     f'accuracy (track) {valid_acc_track:.3f}')
        valid_metrics = dict(
            validation_loss=valid_loss,
            validation_acc=valid_acc_clip,
            validation_acc_track=valid_acc_track
        )
        if self.use_mlflow:
            mlflow.log_metrics(valid_metrics)  # No need to log the step
        logger.info(f'Finished in {time() - training_start:.2f} seconds!')


def main(run_name: str):
    # Set seed for reproducible results
    utils.seed_everything(utils.SEED)
    # Get tracking URI for Mlflow according to hostname
    tracking_uri = get_tracking_uri()
    # Try and connect to mlflow
    try:
        logger.debug(f'Attempting to connect to MLFlow server at {tracking_uri}...')
        mlflow.set_tracking_uri(uri=tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
    # If we're unable to reach the MLFlow server somehow
    except mlflow.exceptions.MlflowException as err:
        logger.warning(f'Could not connect to MLFlow, falling back to running locally! {err}')
        use_mlflow = False
    # Otherwise, default to using mlflow
    else:
        use_mlflow = True
    # If we can't get the tracking uri, always skip over mlflow
    if tracking_uri is None:
        use_mlflow = False
    # Create the training module
    train_module = KongTrainer(run_name=run_name, use_mlflow=use_mlflow)
    # Start training with Mlflow
    if use_mlflow:
        with mlflow.start_run(run_name=run_name):
            train_module.run_training()
    # Start training without mlflow
    else:
        train_module.run_training()


if __name__ == "__main__":
    from argparse import ArgumentParser

    logger.info('Replicating results from Kong et al. (2020) using our implementation of their model...')
    # Parsing arguments from the command line interface
    parser = ArgumentParser(description='Run model training')
    parser.add_argument(
        "-r", "--run-name",
        default="giantmidi-10class-pianoroll-noaugment",
        type=str,
        help="Name of the run, used in MLflow and checkpointing"
    )
    # Parse all the arguments
    args = vars(parser.parse_args())
    # Start training
    logger.debug(f'Starting training using run name {args["run_name"]}')
    main(run_name=args["run_name"])
