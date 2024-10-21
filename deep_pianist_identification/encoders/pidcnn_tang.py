#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Adapted from original repo (https://github.com/tangjjbetsy/PID-CNN/blob/master/network.py) for our input shape"""

import torch
import torch.nn as nn

__all__ = ["TangCNN"]


class TangCNN(nn.Module):
    """Mostly unchanged from original implementation, some adjustments for forwards compatibility"""

    def __init__(
            self,
            num_classes: int,  # Now this argument is required
            num_of_feature: int = 1,
            kernel_size: list = None,
            dropout: float = 0.5,
            dense_size: int = 512
    ):
        super(TangCNN, self).__init__()
        if kernel_size is None:
            kernel_size = [5, 3, 3, 3]

        # These layers have been updated to nn.XXXX2d
        self.convnet = nn.Sequential(
            # (N -> 64)
            nn.Conv2d(num_of_feature, 64, kernel_size[0], padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            # (64 -> 64)
            nn.Conv2d(64, 64, kernel_size[1], padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            # Pooling layer 1
            nn.MaxPool2d(kernel_size[1], stride=kernel_size[1]), nn.Dropout(dropout),
            # (64 -> 128)
            nn.Conv2d(64, 128, kernel_size[1], padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            # (128 -> 128)
            nn.Conv2d(128, 128, kernel_size[2], padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            # Pooling layer 2
            nn.MaxPool2d(kernel_size[2], stride=kernel_size[2]), nn.Dropout(dropout),
            # (128 -> 128)
            nn.Conv2d(128, 128, kernel_size[3], padding=1), nn.ReLU(),
            # Final aggregation
            nn.BatchNorm2d(128), nn.AdaptiveAvgPool2d(1)
        )

        # We remove the softmax from this block (applied in training.py) and split it into two attributes
        self.fc1 = nn.Sequential(
            nn.Linear(128, dense_size),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(dense_size, num_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from deep_pianist_identification.dataloader import MIDILoader
    from deep_pianist_identification import utils

    size = 2
    n_batches = 10
    times = []
    loader = DataLoader(
        MIDILoader(
            split='train',
            n_clips=size * n_batches,
            multichannel=False,
            data_split_dir="25class_0min"
        ),
        batch_size=size,
        shuffle=True,
    )
    model = TangCNN(num_classes=25).to(utils.DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    for feat, _, __ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
