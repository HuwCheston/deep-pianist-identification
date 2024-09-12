#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_pianist_identification.dataloader import TrackLoader
from deep_pianist_identification.encoders import CRNNet
from deep_pianist_identification.utils import timer, SEED

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100


class TrainModule:
    def __init__(self):
        self.model = CRNNet(has_gru=True).to(DEVICE)
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
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def step(self, features, targets):
        features = features.to(DEVICE)
        embeds = self.model(features)
        logits = F.softmax(embeds, dim=1)
        loss = self.loss_fn(logits, targets.to(DEVICE))
        return loss

    def training(self):
        train_loss = []
        self.model.train()
        start = time.time()
        for features, targets in tqdm(self.train_loader, total=len(self.train_loader), desc='Training...'):
            end = time.time()
            print(f'Took {end - start} seconds to load batch')
            # Forwards pass
            with timer('forwards pass'):
                loss = self.step(features, targets)
            # Backwards pass
            with timer('backwards pass'):
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                train_loss.append(loss.item())
            start = time.time()
        return np.mean(train_loss)

    def testing(self):
        test_loss = []
        self.model.eval()
        with torch.no_grad():
            for features, targets in tqdm(self.test_loader, total=len(self.test_loader), desc='Testing...'):
                # Forwards pass
                loss = self.step(features, targets)
                # No backwards pass
                test_loss.append(loss.item())
        return np.mean(test_loss)

    def run_training(self):
        for epoch in range(EPOCHS):
            train_loss = self.training()
            test_loss = self.testing()
            print(f'Epoch {epoch}, train loss: {train_loss}, test loss: {test_loss}')


if __name__ == "__main__":
    torch.manual_seed(SEED)
    tm = TrainModule()
    tm.run_training()
