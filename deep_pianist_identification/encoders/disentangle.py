#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Working towards an interpretable performer identification model from a multi-dimensional piano roll representation"""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import deep_pianist_identification.utils as utils
from deep_pianist_identification.encoders.cnn import ConvLayer, LinearLayer
from deep_pianist_identification.encoders.crnn import GRU

__all__ = ["DisentangleNet", "GeM", "MaskedAvgPool", "LinearFlatten"]


class MaskedAvgPool(nn.Module):
    """Average pooling that omits dimensions which have been masked with zeros. When masks=3, equivalent to max pool."""

    def __init__(self):
        super(MaskedAvgPool, self).__init__()
        self.dim = 2

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Computes average for all non-masked columns of a 3D tensor. Expected shape is (batch, features, channels)"""
        mask = x != 0
        pooled = (x * mask).sum(dim=self.dim) / mask.sum(dim=self.dim)
        # We add an extra dimension here for parity with AdaptiveAvgPool/MaxPool in torch
        return pooled.unsqueeze(2)


class LinearFlatten(nn.Module):
    """For a tensor of shape (batch, features, channels), returns a tensor of shape (batch, features * channels)"""

    def __init__(self):
        super(LinearFlatten, self).__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        # We add an extra dimension here for parity with AdaptiveAvgPool/MaxPool in torch
        return torch.flatten(x.permute(0, 2, 1), start_dim=1).unsqueeze(2)


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

    def __init__(self, num_layers: int = 8, use_gru: bool = True):
        super(Concept, self).__init__()
        # Create the required number of convolutional layers
        self.layers = self.create_convolutional_layers(num_layers)
        # We pool after every two layers, apart from in the final layer
        output_pools = (num_layers // 2) - 1
        # The number of output features after the final convolutional layer
        self.output_features = 2 ** (num_layers // 2 + 5)
        # The number of output channels in final layer (equivalent to the initial height, divided by 2 for each pool)
        self.output_channels = utils.PIANO_KEYS // 2 ** output_pools
        # Whether we're processing the input with a final BiGRU layer
        self.use_gru = use_gru
        # HACK FOR GETTING OLD RESULTS BACK
        if num_layers == 4:
            logger.debug("Using old results hack...")
            self.output_features = 512
            self.output_channels = 11
        if self.use_gru:
            self.gru = GRU(
                input_size=self.output_features * self.output_channels,  # channels * features (== 5632 always)
                hidden_size=self.output_features // 2
            )
        self.maxpool = nn.AdaptiveMaxPool2d((self.output_features, 1))

    @staticmethod
    def create_convolutional_layers(num_layers: int) -> nn.ModuleList:
        assert num_layers % 2 == 0, f"Number of convolutional layers must be even, but received {num_layers} layers."
        # HACK FOR GETTING OLD RESULTS BACK
        if num_layers == 4:
            return nn.ModuleList([
                ConvLayer(1, 64, has_pool=True),
                ConvLayer(64, 128, has_pool=True),
                ConvLayer(128, 256, has_pool=True),
                ConvLayer(256, 512, has_pool=False)  # No pooling for final layer
            ])

        # Arbitrarily long list of embedding values, i.e. 64, 128, 256, 512, 1024, 2048, ...
        dims = [1] + [64 * 2 ** i for i in range(512)]
        layers = nn.ModuleList([])
        # We create two layers for each iteration for simplicity
        for i in range(0, num_layers // 2):
            input_channels, output_channels = dims[i], dims[i + 1]
            # Odd-numbered layers increase channel dimension, have no pooling
            ol = ConvLayer(
                input_channels,
                output_channels,
                has_pool=False
            )
            # Even-numbered layers maintain channel dimension, but have pooling
            el = ConvLayer(
                output_channels,
                output_channels,
                has_pool=False if (i + 1 == num_layers // 2) else True  # No pooling for final layer
            )
            # Add the layers to our list
            layers.extend([ol, el])
        # Check that the final convolutional layer does not have pooling
        assert len(layers[-1]) == 4
        return layers

    def forward(self, x):
        # Iterate through all convolutional layers and process input
        for layer_num in range(len(self.layers)):
            x = self.layers[layer_num](x)
        # Final GRU and maxpooling modules
        if self.use_gru:
            x = self.gru(x)
            x = self.maxpool(x)
        # With no GRU, just pool and remove final redundant dimension
        else:
            x = nn.functional.adaptive_max_pool2d(x, 1).squeeze(-1)
        # (batch, channels, features)
        x = x.permute(0, 2, 1)
        return x


class DisentangleNet(nn.Module):
    def __init__(
            self,
            use_gru: bool = True,
            use_masking: bool = False,
            use_attention: bool = True,
            num_layers: int = 4,
            num_attention_heads: int = 2,
            mask_probability: float = 0.3,
            max_masked_concepts: int = 3,
            pool_type: str = "avg",
            classify_dataset: bool = False
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
        self.concept_embed_dim = self.melody_concept.output_features
        # Attention and pooling between concepts (i.e., channels)
        self.use_attention = use_attention
        if self.use_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.concept_embed_dim,
                num_heads=num_attention_heads,
                batch_first=True,  # Means we don't have to permute tensor in forward
                dropout=0.1
            )
        self.pooling = self.get_pooling_module(pool_type)
        # Linear layers project on to final output: these have dropout added with p=0.5
        self.fc1 = LinearLayer(
            # If we're flattening -> linear at the end (rather than pooling), input size will be 2048, not 512
            in_channels=self.concept_embed_dim * 4 if pool_type == "linear" else self.concept_embed_dim,
            out_channels=self.concept_embed_dim // 4,
            p=0.5
        )
        self.fc2 = LinearLayer(
            in_channels=self.concept_embed_dim // 4,  # i.e., fc1 out_channels
            out_channels=2 if classify_dataset else utils.N_CLASSES,  # either binary or multiclass
            p=0.5
        )

    def get_pooling_module(self, pool_type: str) -> nn.Module:
        """Returns the correct final pooling module to use after the self-attention layer"""
        accept = ["gem", "max", "avg", "masked-avg", "linear"]
        assert pool_type in accept, "Module `pool_type` must be one of" + ", ".join(accept)
        if pool_type == "max":
            return nn.AdaptiveMaxPool1d(1)
        elif pool_type == "avg":
            return nn.AdaptiveAvgPool1d(1)
        elif pool_type == "masked-avg":
            return MaskedAvgPool()
        elif pool_type == "linear":
            # We can't both compute self attention and use our linear pooling hack
            if self.use_attention:
                raise AttributeError("Cannot set both `use_attention=True` and `pool_type='linear'`")
            return LinearFlatten()
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
            embeddings[idx] = self.dropper(embeddings[idx])
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
        # TODO: make sure the tensor is in the correct format!
        if self.use_attention:
            x, _ = self.self_attention(x, x, x)
        # Permute to (batch, features, channels)
        x = x.permute(0, 2, 1)
        # Pool across channels
        x = self.pooling(x)
        # Remove singleton dimension
        x = x.squeeze(2)
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

    size = 2
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
        use_attention=False,
        use_masking=True,
        use_gru=True,
        max_masked_concepts=1,
        mask_probability=1.0,
        num_layers=4,
        pool_type="linear",
        num_attention_heads=2
    ).to(utils.DEVICE)
    model.train()
    print(utils.total_parameters(model))
    for feat, _, __ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
