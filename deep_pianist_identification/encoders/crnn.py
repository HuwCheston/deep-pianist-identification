#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CRNN architecture adapted from Kong et al. (2021), 'Large-scale MIDI-based Composer Classification'"""

import torch
import torch.nn as nn

__all__ = ["CRNNet"]


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.drop = nn.Dropout2d(p=0.2)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.has_pool = has_pool
        if has_pool:
            self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), padding=0)

    def forward(self, a) -> torch.tensor:
        a = self.conv(a)
        a = self.drop(a)
        a = self.relu(a)
        a = self.batchnorm(a)
        if self.has_pool:
            a = self.avgpool(a)
        return a


class LinearLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.drop = nn.Dropout(0.5)

    def forward(self, a) -> torch.tensor:
        a = self.fc(a)
        a = self.drop(a)
        return a


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
    def __init__(self, n_classes: int = 25, has_gru: bool = True):
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
        self.layer8 = ConvLayer(512, 512, has_pool=False)  # No pooling for final convolutional layer
        self.has_gru = has_gru
        if self.has_gru:
            # Bidirectional GRU, operates on features * height dimension
            self.gru = GRU(512 * 11)  # final_output * (input_height / num_layers)
            self.maxpool = nn.AdaptiveMaxPool2d((512, 1))
        else:
            # No GRU (i.e., CNN only)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
        # Linear layers output size: [128, n_classes]
        self.fc1 = LinearLayer(512, 128)
        self.fc2 = LinearLayer(128, n_classes)

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
        # batch_size, features, width)
        if self.has_gru:
            x = self.gru(x)
        # (batch_size, features)
        x = self.maxpool(x)
        batch_size, features, _ = x.size()
        x = x.reshape((batch_size, features))
        # (batch_size, n_classes)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
