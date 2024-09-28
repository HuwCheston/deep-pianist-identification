#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Working towards an interpretable performer identification model from a multi-dimensional piano roll representation"""

import torch
import torch.nn as nn

import deep_pianist_identification.utils as utils
from deep_pianist_identification.encoders.cnn import ConvLayer, LinearLayer
from deep_pianist_identification.encoders.crnn import GRU

__all__ = ["DisentangleNet"]


class Concept(nn.Module):
    def __init__(self, num_layers: int = 4, use_gru: bool = True):
        super(Concept, self).__init__()
        # Convolutional layers output size: [64, 64, 128, 128, 256, 256, 512, 512]
        # Each layer goes: conv -> dropout -> relu -> batchnorm -> avgpool
        self.layer1 = ConvLayer(1, 64, has_pool=True)
        self.layer2 = ConvLayer(64, 128, has_pool=True)
        self.layer3 = ConvLayer(128, 256, has_pool=True)
        self.layer4 = ConvLayer(256, 512, has_pool=False)  # No pooling for final layer
        self.gru = GRU(512 * 11)
        self.maxpool = nn.AdaptiveMaxPool2d((512, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
            num_heads: int = 2,
            use_masking: bool = False,
            masking_prob: float = 0.5
    ):
        super(DisentangleNet, self).__init__()
        # Whether to randomly mask individual channels after processing
        self.use_masking = use_masking
        self.masking_prob = masking_prob
        # Layers for each individual musical concept
        self.melody_concept = Concept(num_layers, use_gru)
        self.harmony_concept = Concept(num_layers, use_gru)
        self.rhythm_concept = Concept(num_layers, use_gru)
        self.dynamics_concept = Concept(num_layers, use_gru)
        # Learn connections and pooling between concepts (i.e., channels)
        self.self_attention = nn.MultiheadAttention(512, num_heads=num_heads, batch_first=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Linear layers project on to final output
        self.fc1 = LinearLayer(512, 128)
        self.fc2 = LinearLayer(128, utils.N_CLASSES)

    def forward(self, x):
        # Split multi-dimensional piano roll into each concept
        melody, harmony, rhythm, dynamics = torch.split(x, 1, 1)
        # Process concepts individually
        melody = self.melody_concept(melody)
        harmony = self.harmony_concept(harmony)
        rhythm = self.rhythm_concept(rhythm)
        dynamics = self.dynamics_concept(dynamics)
        # TODO: masking goes here
        # Stack into a single tensor: (batch, channels, features)
        x = torch.cat([melody, harmony, rhythm, dynamics], dim=1)
        # Self-attention across channels
        x, _ = self.self_attention(x, x, x)
        # Pool across channels
        batch, channels, features = x.size()
        x = x.reshape((batch, features, channels))
        x = self.pooling(x)
        x = x.squeeze(2)  # Remove unnecessary final layer
        # Project onto final output
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
    from torch.utils.data import DataLoader

    size = 1
    model = DisentangleNet().to(utils.DEVICE)
    loader = DataLoader(
        MIDILoader(
            'train',
            n_clips=2,
            multichannel=True,
        ),
        batch_size=size,
        shuffle=True,
        collate_fn=remove_bad_clips_from_batch
    )
    model = DisentangleNet().to(utils.DEVICE)
    print(utils.total_parameters(model))
    for feat, _ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
