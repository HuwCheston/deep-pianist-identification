#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN architecture adapted from Kong et al. (2021), 'Large-scale MIDI-based Composer Classification'"""

import torch
import torch.nn as nn

from deep_pianist_identification.utils import N_CLASSES

__all__ = ["CNNet", "ConvLayer", "LinearLayer", "IBN"]


class IBN(nn.Module):
    """Single IBN layer. Batch normalize half the input channels, instance normalize the other."""

    def __init__(self, out_channels: int, ratio: float = 0.5):
        super(IBN, self).__init__()
        self.half = int(out_channels * ratio)
        self.instance_norm = nn.InstanceNorm2d(self.half, affine=True)
        self.batch_norm = nn.BatchNorm2d(out_channels - self.half)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Split tensor along the channel dimension
        split = torch.split(x, self.half, 1)
        #  Batch normalization is applied to half the channels, instance normalization to the other half
        out1 = self.instance_norm(split[0].contiguous())
        out2 = self.batch_norm(split[1].contiguous())
        # Concatenate the results of both normalization procedures
        out = torch.cat((out1, out2), 1)
        return out


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            has_pool: bool = False,
            use_ibn: bool = False
    ):
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
        if use_ibn:
            self.norm = IBN(out_channels)
        else:
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.has_pool = has_pool
        if has_pool:
            self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), padding=0)

    def __len__(self) -> int:
        """Returns the number of modules in the convolutional layer."""
        return 5 if self.has_pool else 4

    def forward(self, a) -> torch.tensor:
        a = self.conv(a)
        a = self.drop(a)
        a = self.relu(a)
        a = self.norm(a)
        if self.has_pool:
            a = self.avgpool(a)
        return a


class LinearLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.drop = nn.Dropout(p)

    def forward(self, a) -> torch.tensor:
        a = self.fc(a)
        a = self.drop(a)
        return a


class CNNet(nn.Module):
    def __init__(self, use_ibn: bool = False):
        super().__init__()
        # Convolutional layers output size: [64, 64, 128, 128, 256, 256, 512, 512]
        # Each layer goes: conv -> dropout -> relu -> batchnorm (-> avgpool, every 2 layers)
        self.layer1 = ConvLayer(1, 64, use_ibn=use_ibn)
        self.layer2 = ConvLayer(64, 64, use_ibn=use_ibn, has_pool=True)
        self.layer3 = ConvLayer(64, 128, use_ibn=use_ibn)
        self.layer4 = ConvLayer(128, 128, use_ibn=use_ibn, has_pool=True, )
        self.layer5 = ConvLayer(128, 256, use_ibn=use_ibn)
        self.layer6 = ConvLayer(256, 256, use_ibn=use_ibn, has_pool=True)
        self.layer7 = ConvLayer(256, 512, use_ibn=use_ibn)
        # No pooling or IBN for final layer
        self.layer8 = ConvLayer(512, 512, use_ibn=False, has_pool=False)
        # No GRU (i.e., CNN only)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
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
        # (batch_size, features)
        x = self.maxpool(x)
        batch_size, features, _, __ = x.size()
        x = x.reshape((batch_size, features))
        # (batch_size, n_classes)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    from deep_pianist_identification.dataloader import MIDILoader
    from deep_pianist_identification.utils import DEVICE
    from torch.utils.data import DataLoader

    size = 1
    n_batches = 10
    times = []
    loader = DataLoader(
        MIDILoader('train', n_clips=2),
        batch_size=size,
        shuffle=True,
    )
    model = CNNet(use_ibn=False).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    for feat, _ in loader:
        embeds = model(feat.to(DEVICE))
        print(embeds.size())
