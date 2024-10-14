#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CRNN architecture adapted from Kong et al. (2021), 'Large-scale MIDI-based Composer Classification'"""

import torch
import torch.nn as nn

from deep_pianist_identification.encoders import CNNet
from deep_pianist_identification.encoders.shared import GRU, GeM

__all__ = ["CRNNet"]


class CRNNet(CNNet):
    def __init__(
            self,
            num_classes: int,
            pool_type: str = "max",
            norm_type: str = "bn"
    ):
        super().__init__(num_classes=num_classes, pool_type=pool_type, norm_type=norm_type)
        # Bidirectional GRU, operates on features * height dimension
        self.gru = GRU(512 * 11)  # final_output * (input_height / num_layers)

    @staticmethod
    def get_pooling_module(pool_type: str) -> nn.Module:
        """Returns the correct final pooling module"""
        accept = ["gem", "avg", "max"]
        assert pool_type in accept, "Module `pool_type` must be one of" + ", ".join(accept)
        # NB this is different to the parent class
        if pool_type == "avg":
            return nn.AdaptiveAvgPool2d((512, 1))
        elif pool_type == "max":
            return nn.AdaptiveMaxPool2d((512, 1))
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
        # (batch_size, features, width)
        x = self.gru(x)
        # (batch_size, features)
        x = self.pooling(x)
        x = x.squeeze(2)  # squeeze to remove singleton dimension
        return x


if __name__ == "__main__":
    import deep_pianist_identification.utils as utils
    from deep_pianist_identification.dataloader import MIDILoader
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
    model = CRNNet(
        num_classes=25,
        norm_type="bn",
        pool_type="max"
    ).to(utils.DEVICE)
    print(utils.total_parameters(model))
    for feat, _, __ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
