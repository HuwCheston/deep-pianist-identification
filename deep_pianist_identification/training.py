#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

import os
from time import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from tqdm import tqdm

from deep_pianist_identification.dataloader import MIDILoader
from deep_pianist_identification.encoders import CRNNet, CNNet
from deep_pianist_identification.utils import DEVICE, N_CLASSES, seed_everything, get_project_root

# Any key-value pairs we don't define in our custom config will be overwritten using these
DEFAULT_CONFIG = yaml.safe_load(open(os.path.join(get_project_root(), 'config', 'debug_local.yaml')))


class TrainModule:
    def __init__(self, **kwargs):
        # Set all keyword arguments to class parameters
        for key, val in kwargs.items():
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
        )
        # LOSS AND METRICS
        logger.debug('Initialising loss and metrics...')
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
        self.f1_fn = F1Score(task="multiclass", num_classes=N_CLASSES, average="macro").to(DEVICE)
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

    def step(self, features: torch.tensor, targets: torch.tensor) -> tuple:
        # Set device correctly for both features and targets
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = self.model(features)
        # No need to softmax beforehand, `nn.CrossEntropyLoss` does this automatically
        loss = self.loss_fn(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        # Compute metrics
        f1 = self.f1_fn(predictions, targets)
        acc = self.acc_fn(predictions, targets)
        return loss, f1, acc, predictions

    def training(self, epoch_num: int) -> tuple[float, float, float]:
        train_loss, train_fs, train_acc = [], [], []
        self.model.train()
        for features, targets in tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f'Training, epoch {epoch_num} / {self.epochs}...'
        ):
            loss, f1, acc, _ = self.step(features, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Append all metrics from this batch
            train_loss.append(loss.item())
            train_fs.append(f1.item())
            train_acc.append(acc.item())
        return np.mean(train_loss), np.mean(train_fs), np.mean(train_acc)

    def testing(self, epoch_num: int) -> tuple[float, float, float]:
        test_loss, test_fs, test_acc = [], [], []
        self.model.eval()
        with torch.no_grad():
            for features, targets in tqdm(
                    self.test_loader,
                    total=len(self.test_loader),
                    desc=f'Testing, epoch {epoch_num} / {self.epochs}...'
            ):
                # Forwards pass
                loss, f1, acc, _ = self.step(features, targets)
                # No backwards pass
                test_loss.append(loss.item())
                test_fs.append(f1.item())
                test_acc.append(acc.item())
        return np.mean(test_loss), np.mean(test_fs), np.mean(test_acc)

    def load_checkpoint(self) -> None:
        """Load the latest checkpoint for the current experiment and run"""
        checkpoint_folder = os.path.join(get_project_root(), 'checkpoints', self.experiment, self.run)
        # If we haven't created a checkpoint for this run, skip loading and train from scratch
        if not os.path.exists(checkpoint_folder):
            logger.warning('Checkpoint folder does not exist, skipping load!')
            return

        # Sort the checkpoints and load the latest one
        latest_checkpoint = sorted(os.listdir(checkpoint_folder))[-1]
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
        if self.current_epoch % checkpoint_after == 0:
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
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time()
            # Training
            train_loss, train_f, train_acc = self.training(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, training finished: loss {train_loss}, accuracy {train_acc}')
            # Testing
            test_loss, test_f, test_acc = self.testing(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, testing finished: loss {test_loss}, accuracy {test_acc}')
            # Log parameters from this epoch in MLFlow
            metrics = dict(
                epoch_time=time() - epoch_start,
                train_loss=train_loss,
                train_f=train_f,
                train_acc=train_acc,
                test_loss=test_loss,
                test_f=test_f,
                test_acc=test_acc
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
        mlflow.set_tracking_uri(uri=args["mlflow"]["tracking_uri"])
        mlflow.set_experiment(args["mlflow"]["experiment"])
        with mlflow.start_run(run_name=args["mlflow"]["run"]):
            tm.run_training()
    else:
        tm.run_training()
