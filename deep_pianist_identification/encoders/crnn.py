#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CRNN architecture adapted from Kong et al. (2021), 'Large-scale MIDI-based Composer Classification'"""

import torch
import torch.nn as nn

from deep_pianist_identification.encoders import CNNet

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


class CRNNet(CNNet):
    def __init__(self, use_ibn: bool = False, classify_dataset: bool = False):
        super().__init__(use_ibn=use_ibn, classify_dataset=classify_dataset)
        # Bidirectional GRU, operates on features * height dimension
        self.gru = GRU(512 * 11)  # final_output * (input_height / num_layers)
        self.maxpool = nn.AdaptiveMaxPool2d((512, 1))

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


if __name__ == "__main__":
    import deep_pianist_identification.utils as utils
    from deep_pianist_identification.dataloader import MIDILoader
    from torch.utils.data import DataLoader

    size = 1
    n_batches = 10
    times = []
    loader = DataLoader(
        MIDILoader('train', n_clips=2),
        batch_size=size,
        shuffle=True,
    )
    model = CRNNet(use_ibn=True).to(utils.DEVICE)
    print(utils.total_parameters(model))
    for feat, _, __ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
