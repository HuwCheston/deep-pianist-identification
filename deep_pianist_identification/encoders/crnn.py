#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CRNN architecture adapted from Kong et al. (2021), 'Large-scale MIDI-based Composer Classification'"""

import torch
import torch.nn as nn

from deep_pianist_identification.encoders.cnn import ConvLayer, LinearLayer
from deep_pianist_identification.utils import N_CLASSES

__all__ = ["CRNNet", "GRU"]


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, a) -> torch.tensor:
        # (batch_size, 512, height, width)
        batch, features, height, width = a.size()
        # (batch_size, width, 512, height)
        a = a.permute(0, 3, 1, 2).contiguous().view(batch, width, features * height)
        # (batch_size, sequence_length, features)
        a, _ = self.gru(a)
        # (batch_size, features, sequence_length)
        a = a.permute(0, 2, 1).contiguous()
        return a


class CRNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers output size: [64, 64, 128, 128, 256, 256, 512, 512]
        # Each layer goes: conv -> dropout -> relu -> batchnorm (-> avgpool, every 2 layers)
        self.layer1 = ConvLayer(1, 64)
        self.layer2 = ConvLayer(64, 64, has_pool=True)
        self.layer3 = ConvLayer(64, 128)
        self.layer4 = ConvLayer(128, 128, has_pool=True)
        self.layer5 = ConvLayer(128, 256)
        self.layer6 = ConvLayer(256, 256, has_pool=True)
        self.layer7 = ConvLayer(256, 512)
        self.layer8 = ConvLayer(512, 512, has_pool=False)  # No pooling for final layer
        # Bidirectional GRU, operates on features * height dimension
        self.gru = GRU(512 * 11)  # final_output * (input_height / num_layers)
        self.maxpool = nn.AdaptiveMaxPool2d((512, 1))
        # Linear layers output size: [128, n_classes]
        self.fc1 = LinearLayer(512, 128)
        self.fc2 = LinearLayer(128, N_CLASSES)

    def forward(self, x) -> torch.tensor:
        # (batch_size, channels, height, width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # (batch_size, features, width)
        x = self.gru(x)
        # (batch_size, features)
        x = self.maxpool(x)
        batch_size, features, _ = x.size()
        x = x.reshape((batch_size, features))
        # (batch_size, n_classes)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
