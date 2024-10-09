#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Adapted from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/models/resnet.py)"""

import torch
import torch.nn as nn

from deep_pianist_identification import utils
from deep_pianist_identification.encoders import GeM

__all__ = ["ResNet50", "Bottleneck"]


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(
            self,
            layers=None,
            classify_dataset: bool = False,
            pool_type: str = "avg"
    ):
        super(ResNet50, self).__init__()
        # Use ResNet50 by default
        if layers is None:
            layers = [3, 4, 6, 3]

        num_classes = 2 if classify_dataset else utils.N_CLASSES
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2, dilate=False)
        self.pooling = self.get_pooling_module(pool_type)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_pooling_module(pool_type: str) -> nn.Module:
        """Returns the correct final pooling module"""
        accept = ["gem", "avg"]
        assert pool_type in accept, "Module `pool_type` must be one of" + ", ".join(accept)
        if pool_type == "avg":
            return nn.AdaptiveAvgPool2d((1, 1))
        else:
            return GeM()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            nn.BatchNorm2d,
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=nn.BatchNorm2d,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from deep_pianist_identification.dataloader import MIDILoader
    from torch.utils.data import DataLoader

    size = 2
    n_batches = 10
    times = []
    loader = DataLoader(
        MIDILoader(
            split='train',
            n_clips=size * n_batches,
            multichannel=False
        ),
        batch_size=size,
        shuffle=True,
    )
    model = ResNet50(pool_type='gem').to(utils.DEVICE)
    print(sum(p.numel() for p in model.parameters()))
    for feat, _, __ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
