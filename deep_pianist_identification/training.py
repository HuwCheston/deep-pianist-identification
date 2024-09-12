#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

from time import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchmetrics.classification as classification
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_pianist_identification.dataloader import TrackLoader
from deep_pianist_identification.encoders import CRNNet
from deep_pianist_identification.utils import SEED

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
N_CLASSES = 25


class TrainModule:
    def __init__(self):
        self.model = CRNNet(has_gru=True, n_classes=N_CLASSES).to(DEVICE)
        self.train_loader = DataLoader(
            TrackLoader('train'),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            # num_workers=os.cpu_count()
        )
        self.test_loader = DataLoader(
            TrackLoader('test'),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            # num_workers=os.cpu_count()
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
        self.f1_fn = classification.F1Score(task="multiclass", num_classes=N_CLASSES, average="macro").to(DEVICE)
        self.acc_fn = classification.Accuracy(task="multiclass", num_classes=N_CLASSES).to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def step(self, features, targets):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = self.model(features)
        # No need to softmax beforehand, `nn.CrossEntropyLoss` does this automatically
        loss = self.loss_fn(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        f1 = self.f1_fn(predictions, targets)
        acc = self.acc_fn(predictions, targets)
        return loss, f1, acc

    def training(self):
        train_loss, train_fs, train_acc = [], [], []
        self.model.train()
        for features, targets in tqdm(self.train_loader, total=len(self.train_loader), desc='Training...'):
            loss, f1, acc = self.step(features, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # Append all metrics from this batch
            train_loss.append(loss.item())
            train_fs.append(f1.item())
            train_acc.append(acc.item())
        return np.mean(train_loss), np.mean(train_fs), np.mean(train_acc)

    def testing(self):
        test_loss, test_fs, test_acc = [], [], []
        self.model.eval()
        with torch.no_grad():
            for features, targets in tqdm(self.test_loader, total=len(self.test_loader), desc='Testing...'):
                # Forwards pass
                loss, f1, acc = self.step(features, targets)
                # No backwards pass
                test_loss.append(loss.item())
                test_fs.append(f1.item())
                test_acc.append(acc.item())
        return np.mean(test_loss), np.mean(test_fs), np.mean(test_acc)

    def run_training(self):
        for epoch in range(EPOCHS):
            epoch_start = time()
            train_loss, train_f, train_acc = self.training()
            test_loss, test_f, test_acc = self.testing()
            # Log parameters from this epoch in MLFlow
            mlflow.log_params(dict(
                epoch=epoch,
                epoch_time=time() - epoch_start,
                train_loss=train_loss,
                train_f=train_f,
                train_acc=train_acc,
                test_loss=test_loss,
                test_f=test_f,
                test_acc=test_acc
            ))


if __name__ == "__main__":
    torch.manual_seed(SEED)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("crnn-jtd+pijama")

    with mlflow.start_run():
        tm = TrainModule()
        tm.run_training()
