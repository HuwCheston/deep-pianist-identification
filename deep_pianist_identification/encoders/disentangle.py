#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Working towards an interpretable performer identification model from a multi-dimensional piano roll representation"""

import numpy as np
import torch
import torch.nn as nn

import deep_pianist_identification.utils as utils
from deep_pianist_identification.encoders.cnn import ConvLayer, LinearLayer
from deep_pianist_identification.encoders.crnn import GRU

__all__ = ["DisentangleNet", "GeM"]


class GeM(nn.Module):
    """Compresses a multidimensional input to a fixed-length vector with generalized mean (GeM) pooling."""

    def __init__(self, p: int = 3, eps: float = 1e-6):
        super(GeM, self).__init__()
        # This sets p to a trainable parameter with an initial parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        # Margin value to prevent division by zero
        self.eps = eps

    def forward(self, x) -> torch.tensor:
        """Pools input of (batch, features, channels) to (batch, features) with GeM"""
        return nn.functional.avg_pool1d(x.clamp(min=self.eps).pow(self.p), kernel_size=x.size(-1)).pow(1.0 / self.p)


class Concept(nn.Module):
    """Individual embedding per piano roll channel, i.e. harmony, melody, rhythm, dynamics."""

    def __init__(self, num_layers: int = 4, use_gru: bool = True):
        super(Concept, self).__init__()
        # Create the required number of convolutional layers
        self.layers = self.create_convolutional_layers(num_layers)
        # TODO: add option to not use GRU here
        self.gru = GRU(512 * int(utils.PIANO_KEYS / len(self.layers)))  # final_output * (input_height / num_layers)
        self.maxpool = nn.AdaptiveMaxPool2d((512, 1))

    @staticmethod
    def create_convolutional_layers(num_layers: int) -> list[nn.Module]:
        assert num_layers in [2, 4, 8], "Module `num_layers` must be either 2, 4, or 8"
        # TODO: consider layer pooling strategies
        if num_layers == 2:
            return nn.ModuleList([
                ConvLayer(1, 256, has_pool=False),
                ConvLayer(256, 512, has_pool=True),
            ])
        elif num_layers == 4:
            # TODO: previously we used pooling on layer 1, 2, and 3
            return nn.ModuleList([
                ConvLayer(1, 64, has_pool=False),
                ConvLayer(64, 128, has_pool=True),
                ConvLayer(128, 256, has_pool=False),
                ConvLayer(256, 512, has_pool=True)
            ])
        else:
            # Convolutional layers output size: [64, 64, 128, 128, 256, 256, 512, 512]
            # Each layer goes: conv -> dropout -> relu -> batchnorm -> avgpool
            # No pooling or IBN for final layer
            return nn.ModuleList([
                ConvLayer(1, 64),
                ConvLayer(64, 64, has_pool=True),
                ConvLayer(64, 128),
                ConvLayer(128, 128, has_pool=True),
                ConvLayer(128, 256),
                ConvLayer(256, 256, has_pool=True),
                ConvLayer(256, 512),
                ConvLayer(512, 512, has_pool=False)
            ])

    def forward(self, x):
        # Iterate through all convolutional layers and process input
        for layer_num in range(len(self.layers)):
            x = self.layers[layer_num](x)
        # Final GRU and maxpooling modules
        x = self.gru(x)
        x = self.maxpool(x)
        # (batch, channels, features)
        x = x.permute(0, 2, 1)
        return x


class DisentangleNet(nn.Module):
    def __init__(
            self,
            num_layers: int = 4,
            use_gru: bool = True,
            num_attention_heads: int = 2,
            use_masking: bool = False,
            mask_probability: float = 0.3,
            max_masked_concepts: int = 3,
            pool_type: str = "avg"
    ):
        super(DisentangleNet, self).__init__()
        # Whether to randomly mask individual channels after processing
        self.use_masking = use_masking
        self.mask_probability = mask_probability
        self.max_masked_concepts = max_masked_concepts  # Max up to this number of concepts
        self.dropper = nn.Dropout(p=1.0)  # Always deactivated during testing!
        # Layers for each individual musical concept
        self.melody_concept = Concept(num_layers, use_gru)
        self.harmony_concept = Concept(num_layers, use_gru)
        self.rhythm_concept = Concept(num_layers, use_gru)
        self.dynamics_concept = Concept(num_layers, use_gru)
        # Learn connections and pooling between concepts (i.e., channels)
        self.self_attention = nn.MultiheadAttention(512, num_heads=num_attention_heads, batch_first=True)
        self.pooling = self.get_pooling_module(pool_type)
        # Linear layers project on to final output: these have dropout added
        self.fc1 = LinearLayer(512, 128, p=0.5)
        self.fc2 = LinearLayer(128, utils.N_CLASSES, p=0.5)

    @staticmethod
    def get_pooling_module(pool_type: str) -> nn.Module:
        """Returns the correct final pooling module to use after the self-attention layer"""
        accept = ["gem", "max", "avg"]
        assert pool_type in accept, "Module `pool_type` must be one of" + ", ".join(accept)
        if pool_type == "max":
            return nn.AdaptiveMaxPool1d(1)
        elif pool_type == "avg":
            return nn.AdaptiveAvgPool1d(1)
        else:
            return GeM()

    def mask(self, embeddings: list[torch.tensor]) -> list[torch.tensor]:
        """Randomly sets 0:self.max_masked_concepts embeddings to 0 and detaches from computation graph"""
        # When training, we want to use a random combination of masks N% of the time
        if np.random.uniform(0, 1) <= self.mask_probability:
            # Choose between 1 and N concepts to mask
            n_to_zero = np.random.randint(1, self.max_masked_concepts + 1)
            # Get the required indexes from the list
            idx_to_zero = np.random.choice(len(embeddings), n_to_zero, replace=False)
        else:
            idx_to_zero = []
        # Apply the masking by setting the corresponding tensor to 0
        for idx in idx_to_zero:
            embeddings[idx] = torch.zeros(
                embeddings[idx].size(),
                dtype=embeddings[idx].dtype,
                device=embeddings[idx].device,
                requires_grad=False  # ensure that gradient flow is prevented for the masked features
            )
            assert embeddings[idx].requires_grad is False
        return embeddings

    def forward_features(self, x: torch.tensor) -> tuple[torch.tensor]:
        """Processes inputs in shape (batch, channels, height, width) to lists of (batch, 1, features)"""
        # Split multi-dimensional piano roll into each concept
        melody, harmony, rhythm, dynamics = torch.split(x, 1, 1)
        # Process concepts individually
        melody = self.melody_concept(melody)
        harmony = self.harmony_concept(harmony)
        rhythm = self.rhythm_concept(rhythm)
        dynamics = self.dynamics_concept(dynamics)
        # Combine all embeddings into a single list
        return [melody, harmony, rhythm, dynamics]

    def forward_pooled(self, x: torch.tensor) -> torch.tensor:
        """Pools a (possibly masked) input in shape (batch, channels, features), to (batch, classes)"""
        # Self-attention across channels
        x, _ = self.self_attention(x, x, x)
        # Pool across channels
        batch, channels, features = x.size()
        x = x.reshape((batch, features, channels))
        x = self.pooling(x)  # Can be either Avg, Max, or GeM pooling
        x = x.squeeze(2)  # Remove unnecessary final layer
        # Project onto final output
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Model forward pass. Processes single embedding per concept, masks, pools, and projects to class logits."""
        # (batch, channels, height, width) -> list of (batch, 1, features) for each concept
        x = self.forward_features(x)
        # If required, apply random masking to concept embeddings during training
        if self.use_masking and self.training:
            x = self.mask(x)
        # Combine list to (batch, channels, features), where channels = 4
        x = torch.cat(x, dim=1)
        # Pool output to (batch, classes)
        x = self.forward_pooled(x)
        return x


if __name__ == '__main__':
    from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
    from torch.utils.data import DataLoader

    size = 1
    loader = DataLoader(
        MIDILoader(
            'train',
            n_clips=10,
            multichannel=True,
        ),
        batch_size=size,
        shuffle=True,
        collate_fn=remove_bad_clips_from_batch
    )
    model = DisentangleNet(
        use_masking=True,
        mask_probability=1.0,
        num_layers=8,
        pool_type="avg",
        use_gru=False,
        num_attention_heads=2
    ).to(utils.DEVICE)
    model.eval()
    print(utils.total_parameters(model))
    for feat, _ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
