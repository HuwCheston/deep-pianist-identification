#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

import os
import warnings
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

import deep_pianist_identification.plotting as plotting
from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
from deep_pianist_identification.encoders import CRNNet, CNNet, DisentangleNet, ResNet50
from deep_pianist_identification.utils import DEVICE, seed_everything, get_project_root, SEED

__all__ = ["groupby_tracks", "get_model", "TrainModule", "parse_config_yaml", "DEFAULT_CONFIG", "NoOpScheduler"]

# Any key-value pairs we don't define in our custom config will be overwritten using these
DEFAULT_CONFIG = yaml.safe_load(open(os.path.join(get_project_root(), 'config', 'debug_local.yaml')))


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
    # Quick check, lengths should be the same
    assert len(pred_class) == len(target_class)
    # Convert predicted and target class indexes to tensors and return on the correct device
    return torch.tensor(pred_class, device=DEVICE, dtype=int), torch.tensor(target_class, device=DEVICE, dtype=int)


def get_model(encoder_module: str):
    """Given a string, returns the correct encoder module"""
    encoders = ["cnn", "crnn", "disentangle", "resnet50"]
    assert encoder_module in encoders, "`encoder_module` must be one of" + ", ".join(encoders)
    if encoder_module == "cnn":
        return CNNet
    elif encoder_module == "crnn":
        return CRNNet
    elif encoder_module == "disentangle":
        return DisentangleNet
    elif encoder_module == "resnet50":
        return ResNet50


def parse_config_yaml(arg_parser, raise_on_not_found: bool = False) -> None:
    try:
        with open(os.path.join(get_project_root(), 'config', arg_parser['config'])) as stream:
            cfg = yaml.safe_load(stream)
    except FileNotFoundError:
        if raise_on_not_found:
            raise FileNotFoundError(f'Config file {arg_parser["config"]} could not be found!')
        else:
            logger.error(f'Config file {arg_parser["config"]} could not be found, falling back to defaults...')
    else:
        for key, val in cfg.items():
            arg_parser[key] = val


class NoOpScheduler(torch.optim.lr_scheduler.LRScheduler):
    """A LR scheduler that does not modify anything but has the same API as all other schedulers."""

    def __init__(self, optimizer, last_epoch=-1):
        super(NoOpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Just returns the current learning rates without modification"""
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainModule:
    def __init__(self, **kwargs):
        # Set all keyword arguments to class parameters
        self.params = kwargs
        for key, val in self.params.items():
            setattr(self, key, val)
        self.current_epoch = 0
        # CLASSIFICATION PROBLEM
        self.classify_dataset = kwargs.get("classify_dataset", False)
        # CLASS MAPPING AND NUMBER OF CLASSES
        self.class_mapping = self.get_class_mapping()
        self.num_classes = len(self.class_mapping.keys())
        logger.debug(f"Model will be trained with {self.num_classes} classes!")
        # MODEL
        logger.debug('Initialising model...')
        logger.debug(f"Using encoder {self.encoder_module} with parameters {self.model_cfg}")
        model_cls = get_model(self.encoder_module)
        self.model = model_cls(
            num_classes=self.num_classes,
            **self.model_cfg
        ).to(DEVICE)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(f'Model parameters: {n_params}')
        # DATALOADERS
        logger.debug(f'Loading data splits from `references/data_splits/{self.data_split_dir}`')
        logger.debug(f'Initialising training dataloader with batch size {self.batch_size} '
                     f'and parameters {self.train_dataset_cfg}')
        self.train_loader = DataLoader(
            MIDILoader(
                'train',
                classify_dataset=self.classify_dataset,
                data_split_dir=self.data_split_dir,
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
                classify_dataset=self.classify_dataset,
                data_split_dir=self.data_split_dir,
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
        # OPTIMISER
        optim_type = kwargs.get("optim_type", "adam")
        optim_cfg = kwargs.get("optim_cfg", dict(learning_rate=0.0001))
        logger.debug(f'Initialising optimiser {optim_type} with parameters {optim_cfg}...')
        self.optimizer = self.get_optimizer(optim_type)(self.model.parameters(), **optim_cfg)
        # SCHEDULER
        sched_type = kwargs.get("sched_type", None)
        sched_cfg = kwargs.get("sched_cfg", dict())
        logger.debug(f'Initialising LR scheduler {sched_type} with parameters {sched_cfg}...')
        self.scheduler = self.get_scheduler(sched_type)(self.optimizer, **sched_cfg)
        # CHECKPOINTS
        if self.checkpoint_cfg["load_checkpoints"]:
            self.load_checkpoint()

    def get_class_mapping(self) -> dict:
        if self.classify_dataset:
            return {0: "jtd", 1: "pijama"}
        else:
            csv_loc = os.path.join(
                get_project_root(), 'references/data_splits', self.data_split_dir, 'pianist_mapping.csv'
            )
            pianist_mapping = pd.read_csv(csv_loc, index_col=1)
            return {i: row.iloc[0] for i, row in pianist_mapping.iterrows()}

    @staticmethod
    def get_optimizer(optim_type: str):
        """Given a string, returns the correct optimizer"""
        optims = ["adam", "sgd"]
        assert optim_type in optims, "`optim_type` must be one of " + ", ".join(optims)
        if optim_type == "adam":
            return torch.optim.Adam
        elif optim_type == "sgd":
            return torch.optim.SGD

    @staticmethod
    def get_scheduler(sched_type: str | None):
        """Given a string, returns the correct optimizer"""
        schs = ["plateau", "cosine", "step", "linear", None]
        assert sched_type in schs, "`sched_type` must be one of " + ", ".join(schs)
        # This scheduler won't modify anything, but provides the same API for simplicity
        if sched_type is None:
            return NoOpScheduler
        elif sched_type == "reduce":
            return torch.optim.lr_scheduler.ReduceLROnPlateau
        elif sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR
        elif sched_type == "linear":
            return torch.optim.lr_scheduler.LinearLR

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
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        # Compute track-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        # Return clip loss + accuracy, track accuracy
        return np.mean(clip_losses), np.mean(clip_accs), track_acc

    def save_figure(self, figure, filename: str) -> None:
        """Saves a figure either locally or on MLflow as an artifact, depending on configuration"""
        # Will be passed to `fig.savefig`
        save_kwargs = dict(format='png', facecolor=plotting.WHITE)
        # If we're using mlflow, log directly using their API
        if self.mlflow_cfg["use"]:
            mlflow.log_figure(figure, filename, save_kwargs=save_kwargs)
            logger.debug(f'Saved figure {filename} on MLFlow as an artifact for {self.experiment}/{self.run}')
        # Otherwise, create the figure locally in the checkpoints directory for this run
        else:
            figures_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run, 'figures')
            # Make the directory if it does not exist
            if not os.path.exists(figures_folder):
                os.makedirs(figures_folder, exist_ok=True)
            figure.savefig(os.path.join(figures_folder, filename), **save_kwargs)
            logger.debug(f'Saved figure {filename} in {figures_folder}')

    def plot_confusion_matrix(self, predictions: torch.tensor, targets: torch.tensor, ext: str = '') -> None:
        """Plots a confusion matrix between `predictions` and `targets`"""
        # Create the confusion matrix as a tensor, expressed in percentages
        # This will raise weird errors about NaNs being present, but the output is fine, so ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = self.confusion_matrix_fn(predictions, targets) * 100
        # Detach from computational graph if required
        if mat.requires_grad:
            mat = mat.detach()
        # Convert to a numpy array
        mat = mat.cpu().numpy()
        # Create the plot
        cm = plotting.HeatmapConfusionMatrix(mat, pianist_mapping=self.class_mapping)
        cm.create_plot()
        # Save the figure, either on MLflow or locally (depending on our configuration)
        self.save_figure(cm.fig, f"epoch_{str(self.current_epoch).zfill(3)}_confusionmatrix{ext}.png")
        # Close the figure in matplotlib
        cm.close()

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
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        # Compute track-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        # Save a confusion matrix for test set predictions if we're using checkpoints
        # TODO: implement confusion matrixes for binary (accompanied/unaccopanied) classification
        if self.current_epoch % plotting.PLOT_AFTER_N_EPOCHS == 0 and not self.classify_dataset:
            self.plot_confusion_matrix(predictions, targets)
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
        checkpoints = [i for i in os.listdir(checkpoint_folder) if i.endswith(".pth")]
        if len(checkpoints) == 0:
            logger.warning('No checkpoints have been created yet for this experiment/run, skipping load!')
            return

        # Sort the checkpoints and load the latest one
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(checkpoint_folder, latest_checkpoint)
        # This will raise a warning about possible ACE exploits, but we don't care
        loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        # Set state dictionary for all torch objects
        self.model.load_state_dict(loaded["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        # For backwards compatibility with no LR scheduler runs: don't worry if we can't load the LR scheduler dict
        try:
            self.scheduler.load_state_dict(loaded['scheduler_state_dict'])
        except KeyError:
            logger.warning("Could not find scheduler state dictionary in checkpoint, scheduler will be restarted!")
            pass
        # Increment epoch by 1
        self.current_epoch = loaded["epoch"] + 1
        self.scheduler.last_epoch = self.current_epoch
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
                    scheduler_state_dict=self.scheduler.state_dict(),
                    epoch=self.current_epoch
                ),
                os.path.join(checkpoint_folder, f'checkpoint_{str(self.current_epoch).zfill(3)}.pth'),
            )
            logger.debug(f'Saved a checkpoint to {checkpoint_folder}')

    def log_run_params_to_mlflow(self):
        """If we're using MLFlow, log all run parameters to the dashboard"""
        if self.mlflow_cfg["use"]:
            # TODO: we should expand individual dictionaries here and account for duplicate keys somehow
            for key, val in self.params.items():
                mlflow.log_param(key, val)
            logger.debug("Logged all run parameters to MLFlow dashboard!")

    def get_scheduler_lr(self) -> float:
        """Tries to return current LR from scheduler. Returns 0. on error, for safety with MLFlow logging"""
        try:
            return self.scheduler.get_last_lr()[0]
        except (IndexError, AttributeError, RuntimeError) as sched_e:
            logger.warning(f"Failed to get LR from scheduler! Returning 0.0... {sched_e}")
            return 0.

    def run_training(self) -> None:
        training_start = time()
        self.log_run_params_to_mlflow()
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
                lr=self.get_scheduler_lr()
            )
            # Checkpoint the run, if we need to
            if self.checkpoint_cfg["save_checkpoints"]:
                self.save_checkpoint(metrics)
            # Report results to MLFlow, if we're using this
            if self.mlflow_cfg["use"]:
                mlflow.log_metrics(metrics, step=epoch)
            # Step forward in the LR scheduler
            self.scheduler.step()
            logger.debug(f'LR for epoch {epoch + 1} will be {self.get_scheduler_lr()}')
        logger.info('Training complete! Beginning validation...')
        self.validation()
        logger.info(f'Finished in {round(time() - training_start)} seconds!')

    def validation(self) -> None:
        # Try and create the validation dataloader
        try:
            validation_loader = DataLoader(
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
        # Skip if we don't have a validation split created for this dataset
        except FileNotFoundError as e:
            logger.error(f'Could not find validation split for dataset {self.data_split_dir}, skipping validation! {e}')
            return
        # Lists to track metrics across all batches
        clip_loss, clip_accs = [], []
        track_names, track_preds, track_targets = [], [], []
        # Set model mode correctly
        self.model.eval()
        # Iterate through all batches, with a nice TQDM counter
        with torch.no_grad():
            for features, targets, batch_names in tqdm(
                    validation_loader,
                    total=len(validation_loader),
                    desc=f'Validating...'
            ):
                # Forwards pass
                loss, acc, preds = self.step(features, targets)
                # No backwards pass, just append all clip-level metrics from this batch
                clip_loss.append(loss.item())
                clip_accs.append(acc.item())
                # Append the names of the tracks and the corresponding predictions for track-level metrics
                track_names.extend(batch_names)  # Single list of strings
                track_preds.append(preds)  # List of tensors, one tensor per clip, containing class logits
                track_targets.append(targets)  # List of tensors, one tensor per batch, containing target classes
        # Compute clip-level accuracy and loss
        clip_acc, clip_loss = np.mean(clip_accs), np.mean(clip_loss)
        # Group by tracks, average probabilities, and get predicted and target classes
        predictions, targets = groupby_tracks(track_names, track_preds, track_targets)
        # Compute track-level accuracy
        track_acc = self.acc_fn(predictions, targets).item()
        # Report all validation metrics to the console
        logger.info(f'Validation finished: '
                    f'loss {clip_loss:.3f}, '
                    f'accuracy (clip) {clip_acc:.3f}, '
                    f'accuracy (track) {track_acc:.3f}')
        # Create the confusion matrix and log to mlflow if we're using it
        self.plot_confusion_matrix(predictions, targets, ext='_validation')
        # Report results to MLFlow, if we're using this
        if self.mlflow_cfg["use"]:
            metrics = dict(
                validation_loss=clip_loss,
                validation_acc=clip_acc,
                validation_acc_track=track_acc
            )
            mlflow.log_metrics(metrics)  # Don't need to log the step


def get_tracking_uri(port: str = "5000") -> str:
    """Attempts to get the MLflow tracking URI on given port based on system hostname"""
    import socket

    hostname = socket.gethostname().lower()
    # Job is running locally or on the department server
    if "desktop" in hostname or "musix" in hostname:
        return f"http://127.0.0.1:{port}"
    # Job is running on HPC
    else:
        return f"http://musix.mus.cam.ac.uk:{port}"


if __name__ == "__main__":
    import yaml
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
        logger.debug(f'Starting training using config file {args["config"]}')
        parse_config_yaml(args)
    else:
        logger.warning("No config file found, using default settings!")

    # Running training with logging on MLFlow
    if args["mlflow_cfg"]["use"]:
        # Get the tracking URI based on the hostname of the device running the job
        uri = get_tracking_uri()
        logger.debug(f'Attempting to connect to MLFlow server at {uri}...')
        mlflow.set_tracking_uri(uri=uri)
        try:
            mlflow.set_experiment(args["experiment"])
        # If we're unable to reach the MLFlow server somehow
        except mlflow.exceptions.MlflowException as err:
            logger.warning(f'Could not connect to MLFlow, falling back to running locally! {err}')
            # This will mean we break out of the current IF statement, and activate the next IF NOT statement
            # in order to train locally, without using MLFlow
            args["mlflow_cfg"]["use"] = False
        else:
            # Otherwise, start training with the arguments we've passed in
            tm = TrainModule(**args)
            with mlflow.start_run(run_name=args["run"]):
                tm.run_training()

    # Running training locally
    if not args["mlflow_cfg"]["use"]:
        tm = TrainModule(**args)
        tm.run_training()
