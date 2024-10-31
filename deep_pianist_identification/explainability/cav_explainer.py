#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Explainer modules for generating CAVs"""

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from deep_pianist_identification import utils
from deep_pianist_identification.explainability.cav_dataloader import VoicingLoaderReal, VoicingLoaderFake

__all__ = ["SCALER", "Explainer"]

SCALER = StandardScaler()


class Explainer:
    def __init__(
            self,
            cav_idx: int,
            model,
            real_loader=VoicingLoaderReal,
            fake_loader=VoicingLoaderFake,
            n_clips: int = None,
            transpose_range: int = 6
    ):
        # Create the dataloaders for the desired CAV
        self.cav_idx = cav_idx
        self.real = DataLoader(
            real_loader(
                self.cav_idx,
                n_clips=n_clips,
                transpose_range=transpose_range
            ),
            shuffle=True,
            batch_size=10,
            drop_last=False
        )
        self.fake = DataLoader(
            fake_loader(
                self.cav_idx,
                n_clips=len(self.real.dataset),
                transpose_range=transpose_range
            ),
            shuffle=True,
            batch_size=10,
            drop_last=False
        )
        assert len(self.real.dataset) == len(self.fake.dataset)
        self.model = model
        self.predictor = LogisticRegression(random_state=utils.SEED, max_iter=10000)
        self.class_centroid = None
        self.cavs = None

    def get_embeds(self):
        real_embeds, fake_embeds = [], []
        for (real_batch, _), (fake_batch, _) in zip(self.real, self.fake):
            # Create both sets of embeddings
            with torch.no_grad():
                # Through the model
                real_embed = self.model(real_batch.to(utils.DEVICE)).squeeze(1)
                fake_embed = self.model(fake_batch.to(utils.DEVICE)).squeeze(1)
                # Append to the list
                real_embeds.append(real_embed.cpu().numpy())
                fake_embeds.append(fake_embed.cpu().numpy())
        return np.concatenate(real_embeds), np.concatenate(fake_embeds)

    def fit(self):
        # Get embeddings from the model
        real_embeds, fake_embeds = self.get_embeds()
        self.class_centroid = real_embeds.mean(axis=0)
        # Create arrays of targets with same shape as embeddings
        real_targets, fake_targets = np.ones(real_embeds.shape[0]), np.zeros(fake_embeds.shape[0])
        # Combine both real/fake features and targets into single matrix
        X = np.vstack([real_embeds, fake_embeds])
        y = np.hstack([real_targets, fake_targets])
        # Scale the features
        X = SCALER.fit_transform(X)
        # Create the model
        self.predictor.fit(X, y)
        acc = self.predictor.score(X, y)
        logger.info(f'Accuracy for CAV {self.cav_idx}: {acc:.2f}')
        # Extract coefficients - these are our CAVs
        self.cavs = self.predictor.coef_
