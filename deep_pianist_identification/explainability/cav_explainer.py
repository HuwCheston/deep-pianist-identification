#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Explainer modules for generating CAVs"""

import os

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.explainability.cav_dataloader import VoicingLoaderReal, VoicingLoaderFake

__all__ = ["SCALER", "ConceptExplainer", "Explainer"]

SCALER = StandardScaler()
EPSILON = 1e-4


class ConceptExplainer:
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


class Explainer:
    def __init__(
            self,
            concept_encoder: torch.nn.Module,
            clip_dataloaders: list[DataLoader],
            n_cavs: int,
            roll_idx: int = 1,
            real_cavloader=VoicingLoaderReal,
            fake_cavloader=VoicingLoaderFake,
            classifier_head: torch.nn.Linear = None
    ):
        self.concept_encoder = concept_encoder
        self.clip_dataloaders = clip_dataloaders
        self.n_cavs = n_cavs
        self.real_cavloader = real_cavloader
        self.fake_cavloader = fake_cavloader
        self.roll_idx = roll_idx
        self.classifier_head = classifier_head

        # Create embeddings that will be filled in later
        self.cavs = None
        self.clip_embeddings, self.clip_targets, self.clip_tracknames = None, None, None
        self.sensitivities = None
        self.tcavs = None

    def create_cavs(self) -> None:
        """Get CAV coefficient vectors for all concepts"""
        explainers = []
        for idx in tqdm(range(1, self.n_cavs + 1), desc='Creating all CAVs: '):
            # Create the Explainer instance, passing in the correct dataloaders, and fit
            explain = ConceptExplainer(
                cav_idx=idx,
                model=self.concept_encoder,
                real_loader=self.real_cavloader,
                fake_loader=self.fake_cavloader,
            )
            explain.fit()
            explainers.append(explain)
        # Concatenate all the CAV coefficients to an array of (n_cavs, d)
        self.cavs = np.concatenate([i.cavs for i in explainers])

    def create_embeddings(self):
        actual_embeds, actual_targets, tns = [], [], []
        for loader in self.clip_dataloaders:
            # Create list of all indexes
            indices = [list(range(li, li + loader.batch_size)) for li in
                       range(0, len(loader.dataset), loader.batch_size)]
            # Iterate over all indexes
            for idxs in tqdm(indices, desc="Getting embeddings"):
                # Subset the indexes so we don't end up exceeding the number of clips
                idxs = [i for i in idxs if i < len(loader.dataset)]
                rolls, targets, paths = [], [], []
                # Now we actually use the __getitem__ call
                for n, i in enumerate(idxs):
                    temp = loader.dataset.__getitem__(i)
                    # replaces collate_fn to remove bad items from batch
                    if temp is not None:
                        r, t, p = temp
                        rolls.append(r)
                        targets.append(t)
                        tn, _, _, clip_idx = loader.dataset.clips[i]
                        assert tn.split(os.path.sep)[-1] == p
                        pather = os.path.sep.join([tn.split(os.path.sep)[-2], p, f'clip_{str(clip_idx).zfill(3)}.mid'])
                        paths.append(pather)
                assert len(rolls) == len(targets) == len(paths)
                # Convert into tensors
                roll = torch.tensor(np.array(rolls))
                target = torch.tensor(np.array(targets))
                # Get the required concept from the piano roll (harmony by default)
                required_roll = roll[:, self.roll_idx, :, :].unsqueeze(1)
                # Call the forward function without gradient computation
                with torch.no_grad():
                    harm_embed = self.concept_encoder(required_roll.to(utils.DEVICE)).squeeze(1)
                # Store original data
                tns.extend(paths)
                # Store layer activations
                actual_embeds.append(harm_embed.cpu().numpy())
                # Store target indexes
                actual_targets.append(target.cpu().numpy())
        # Convert everything to arrays
        self.clip_embeddings = np.concatenate(actual_embeds)
        self.clip_targets = np.concatenate(actual_targets)
        self.clip_tracknames = np.array(tns)

    def cav_sensitivity_dotproduct(self, normalize: bool = False) -> np.ndarray:
        if self.cavs is None:
            raise AttributeError('Must create cavs by calling `Explainer.create_cavs()` first')
        if self.clip_embeddings is None:
            raise AttributeError('Must create clip embeddings by calling `Explainer.create_embeddings()` first')

        if normalize:
            clip_features = torch.nn.functional.normalize(torch.tensor(self.clip_embeddings), p=2, dim=-1).cpu().numpy()
            cav_features = torch.nn.functional.normalize(torch.tensor(self.cavs), p=2, dim=-1).cpu().numpy()
        else:
            clip_features, cav_features = self.clip_embeddings, self.cavs
        return clip_features @ cav_features.T

    def _perturber(self, feature, cav, target_idx):
        # Perturb the embeddings towards and away from the CAV
        away = feature - (cav * EPSILON)
        towards = feature + (cav * EPSILON)
        # Compute the logits for both cases
        with torch.no_grad():
            away_logits = self.classifier_head(away)
            towards_logits = self.classifier_head(towards)
        # Apply the softmax
        away_sm = torch.nn.functional.softmax(away_logits, dim=-1)
        toward_sm = torch.nn.functional.softmax(towards_logits, dim=-1)
        # Get the logits for the target class
        away_desired = away_sm[target_idx]
        toward_desired = toward_sm[target_idx]
        # Compute the sensitivity by subtracting the results
        return toward_desired - away_desired

    def cav_sensitivity_perturb(self):
        if self.cavs is None:
            raise AttributeError('Must create cavs by calling `Explainer.create_cavs()` first')
        if self.clip_embeddings is None:
            raise AttributeError('Must create clip embeddings by calling `Explainer.create_embeddings()` first')
        if self.classifier_head is None:
            raise AttributeError('Must pass `classifier_head` when creating `Explainer` to compute perturb sensitivity')

        all_s = []
        # Iterate through each individual clip embeddings and target
        for feat, targ in zip(self.clip_embeddings, self.clip_targets):
            # Set devices correctly
            feat = torch.tensor(feat).to(utils.DEVICE)
            targ = torch.tensor(targ).to(utils.DEVICE)
            # Iterate through each individual CAV
            clip_res = np.array([self._perturber(feat, cav, targ, ) for cav in self.cavs])
            # Shape = (n_cavs)
            all_s.append(clip_res)
        # Shape = (n_clips, n_cavs)
        return np.array(all_s)

    def compute_tcavs(
            self,
            method: str = "dotproduct",
    ):
        accept = ["dotproduct", "perturb"]
        assert method in accept, "Expected `method` should be one of " + ', '.join(accept) + f'but got {method}'
        # Create CAVs and/or embeddings if they haven't already been created
        if self.cavs is None:
            self.create_cavs()
        if self.clip_embeddings is None:
            self.create_embeddings()
        # Compute sensitivity using desired method
        if method == "dotproduct":
            self.sensitivities = self.cav_sensitivity_dotproduct()
        else:
            self.sensitivities = self.cav_sensitivity_perturb()
