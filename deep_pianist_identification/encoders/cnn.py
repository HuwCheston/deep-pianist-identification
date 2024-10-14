#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN architecture adapted from Kong et al. (2021), 'Large-scale MIDI-based Composer Classification'"""

import torch
import torch.nn as nn

from deep_pianist_identification.encoders.shared import GeM, ConvLayer, LinearLayer, IBN

__all__ = ["CNNet"]


class CNNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            pool_type: str = "max",
            norm_type: str = "bn"
    ):
        super().__init__()
        # Convolutional layers output size: [64, 64, 128, 128, 256, 256, 512, 512]
        # Each layer goes: conv -> dropout -> relu -> batchnorm (-> avgpool, every 2 layers)
        norm_layer = self.get_norm_module(norm_type)
        self.layer1 = ConvLayer(1, 64, norm_layer=norm_layer)
        self.layer2 = ConvLayer(64, 64, norm_layer=norm_layer, has_pool=True)
        self.layer3 = ConvLayer(64, 128, norm_layer=norm_layer)
        self.layer4 = ConvLayer(128, 128, norm_layer=norm_layer, has_pool=True, )
        self.layer5 = ConvLayer(128, 256, norm_layer=norm_layer)
        self.layer6 = ConvLayer(256, 256, norm_layer=norm_layer, has_pool=True)
        self.layer7 = ConvLayer(256, 512, norm_layer=norm_layer)
        # No pooling or IBN for final layer
        self.layer8 = ConvLayer(512, 512, norm_layer=norm_layer, has_pool=False)
        # No GRU (i.e., CNN only)
        self.pooling = self.get_pooling_module(pool_type)
        # Linear layers output size: [128, n_classes]
        self.fc1 = LinearLayer(512, 128)
        self.fc2 = LinearLayer(128, num_classes)

    @staticmethod
    def get_norm_module(norm_type: str) -> nn.Module:
        """Returns the correct normalization module"""
        accept = ["bn", "ibn"]
        assert norm_type in accept, "Module `norm_type` must be one of" + ", ".join(accept)
        if norm_type == "bn":
            return nn.BatchNorm2d
        else:
            return IBN

    @staticmethod
    def get_pooling_module(pool_type: str) -> nn.Module:
        """Returns the correct final pooling module"""
        accept = ["gem", "avg", "max"]
        assert pool_type in accept, "Module `pool_type` must be one of" + ", ".join(accept)
        if pool_type == "avg":
            return nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            return nn.AdaptiveMaxPool2d(1)
        else:
            return GeM()

    def forward_features(self, x: torch.tensor) -> torch.tensor:
        """Returns feature embeddings prior to final linear projection layer(s)"""
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
        x = self.pooling(x)
        x = x.squeeze(2).squeeze(2)  # squeeze to remove singleton dimension
        return x

    def forward(self, x) -> torch.tensor:
        # (batch_size, 512)
        x = self.forward_features(x)
        # (batch_size, 128)
        x = self.fc1(x)
        # (batch_size, n_classes)
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
        MIDILoader(
            'train',
            n_clips=2,
            data_split_dir="25class_0min"
        ),
        batch_size=size,
        shuffle=True,
    )
    model = CNNet(
        num_classes=25,
        norm_type="bn",
        pool_type="max"
    ).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    for feat, _, __ in loader:
        embeds = model(feat.to(DEVICE))
        print(embeds.size())
