#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Working towards an interpretable performer identification model from a multi-dimensional piano roll representation"""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import resnet18, resnet34

import deep_pianist_identification.utils as utils
from deep_pianist_identification.encoders.shared import (
    ConvLayer, LinearLayer, GRU, MaskedAvgPool, LinearFlatten, Conv1x1, GeM, Identity
)

__all__ = ["DisentangleNet", "Concept"]

DEFAULT_CONV_LAYERS = [
    dict(in_channels=1, out_channels=64, conv_kernel_size=3, has_pool=True, pool_kernel_size=2),
    dict(in_channels=64, out_channels=128, conv_kernel_size=3, has_pool=True, pool_kernel_size=2),
    dict(in_channels=128, out_channels=256, conv_kernel_size=3, has_pool=True, pool_kernel_size=2),
    dict(in_channels=256, out_channels=512, conv_kernel_size=3, has_pool=False),
]


class Concept(nn.Module):
    """Individual embedding per piano roll channel, i.e. harmony, melody, rhythm, dynamics."""

    def __init__(self, layers: list[dict], use_gru: bool = True):
        super(Concept, self).__init__()
        # Create the required number of convolutional layers
        self.layers: nn.ModuleList[nn.Module] = self.create_convolutional_layers(layers)
        # Get parameters from the layers
        self.output_channels = self.layers[-1].conv.out_channels  # Output from last convolutional layer
        pool_kernels = [i.avgpool.kernel_size[0] for i in self.layers if len(i) == 5]  # Kernel size of all pools
        self.output_height = utils.PIANO_KEYS // np.prod(pool_kernels)  # Input height divided by product of kernels
        self.output_width = (utils.CLIP_LENGTH * utils.FPS) // np.prod(pool_kernels)  # Width divided by kernels
        # Whether we're processing the input with a final BiGRU layer
        self.use_gru = use_gru
        if self.use_gru:
            self.gru = GRU(
                input_size=self.output_channels * self.output_height,  # channels * features (== 5632 always)
                hidden_size=self.output_channels // 2
            )
        self.maxpool = nn.AdaptiveMaxPool2d((self.output_channels, 1))

    @staticmethod
    def create_convolutional_layers(layers: list[dict]) -> nn.ModuleList:
        # Check that we have an even number of convolutional layers
        num_layers = len(layers)
        assert num_layers % 2 == 0, f"Number of convolutional layers must be even, but received {num_layers} layers."
        # Return the layers with required parameters as a torch ModuleList that can be iterated through
        return nn.ModuleList([ConvLayer(**layer) for layer in layers])

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
            num_classes: int,
            use_gru: bool = True,
            use_masking: bool = False,
            use_attention: bool = True,
            layers: list[dict] = None,
            num_attention_heads: int = 2,
            mask_probability: float = 0.3,
            max_masked_concepts: int = 3,
            pool_type: str = "avg",
            _use_resnet: bool = False,
            _resnet_cls: str = None
    ):
        super(DisentangleNet, self).__init__()
        # Whether to randomly mask individual channels after processing
        self.use_masking = use_masking
        self.mask_probability = mask_probability
        self.max_masked_concepts = max_masked_concepts  # Max up to this number of concepts
        self.dropper = nn.Dropout(p=1.0)  # Always deactivated during testing!
        # Layers for each individual musical concept
        if _use_resnet:
            logger.debug(f"Using ResNet class {_resnet_cls} as concept encoder!")
            self.melody_concept = self.get_resnet(num_classes, _resnet_cls)
            self.harmony_concept = self.get_resnet(num_classes, _resnet_cls)
            self.rhythm_concept = self.get_resnet(num_classes, _resnet_cls)
            self.dynamics_concept = self.get_resnet(num_classes, _resnet_cls)
            # Get the number of output channels in the final convolutional layer
            self.concept_channels = self.melody_concept.layer4[0].conv1.out_channels
        else:
            # Use default convolutional layer strategy, if this is not passed
            if layers is None:
                logger.warning("No convolution layers were passed in `model_cfg`, so falling back on defaults!")
                layers = DEFAULT_CONV_LAYERS
            self.melody_concept = Concept(layers, use_gru)
            self.harmony_concept = Concept(layers, use_gru)
            self.rhythm_concept = Concept(layers, use_gru)
            self.dynamics_concept = Concept(layers, use_gru)
            # Log the shape of each concept encoder
            self.concept_channels = self.melody_concept.output_channels
            self.log_encoder_architecture(self.melody_concept)
        # Attention and pooling between concepts (i.e., channels)
        self.use_attention = use_attention
        if self.use_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.concept_channels,
                num_heads=num_attention_heads,
                batch_first=True,  # Means we don't have to permute tensor in forward
                dropout=0.1
            )
        self.pooling = self.get_pooling_module(pool_type)
        # Linear layers project on to final output: these have dropout added with p=0.5
        self.fc1 = LinearLayer(
            # If we're flattening -> linear at the end (rather than pooling), input size will be 2048, not 512
            in_channels=self.concept_channels * 4 if pool_type == "linear" else self.concept_channels,
            out_channels=self.concept_channels // 4,
            p=0.5
        )
        self.fc2 = LinearLayer(
            in_channels=self.concept_channels // 4,  # i.e., fc1 out_channels
            out_channels=num_classes,  # either binary or multiclass
            p=0.5
        )

    @staticmethod
    def get_resnet(num_classes: int, resnet_cls: str):
        """Returns a resnet withoug pretraining, with the classification head removed and the input channels set to 1"""
        # Passed in resnet class must be one of these
        all_cls = ["resnet18", "resnet34"]
        assert resnet_cls in all_cls, "`resnet_cls` must be one of " + ", ".join(all_cls)
        # Get the correct resnet class from the string
        if resnet_cls == "resnet18":
            maker = resnet18
        else:
            maker = resnet34
        # Create the resnet with the correct number of classes (possibly not needed)
        rn = maker(weights=None, num_classes=num_classes)
        # Remove the classification head by setting it to a custom module that just returns the input tensor with +1 dim
        rn.fc = Identity(expand_dim=True).to(utils.DEVICE)
        # Modify the first convolutional layer as it expects a 3-channel image, and we're only using one channel
        rn.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=rn.conv1.out_channels,
            kernel_size=rn.conv1.kernel_size,
            stride=rn.conv1.stride,
            padding=rn.conv1.padding,
            bias=rn.conv1.bias
        ).to(utils.DEVICE)
        return rn

    @staticmethod
    def log_encoder_architecture(encoder: Concept):
        """Nicely logs the architecture of a particular `Concept` to the console"""
        # Extract output from convolution
        concept_chan = encoder.output_channels
        concept_height = encoder.output_height
        concept_width = encoder.output_width
        lay = encoder.layers
        # Extract input and output channels from convolutional layers
        convs = [f'{i.conv.in_channels, i.conv.out_channels}' for i in lay if hasattr(i, "conv")]
        # Extract which layers have pools and their kernel size
        pidx, pker = zip(*[(num, str(i.avgpool.kernel_size)) for num, i in enumerate(lay) if hasattr(i, "avgpool")])
        # Log everything with loguru
        logger.info(f'Encoders have {len(convs)} convolutional layers with I/O channels {">>".join(convs)}')
        logger.info(f'Encoders have {len(pidx)} pools at layers {pidx} with kernels {", ".join(pker)}')
        # Log GRU specific details
        if encoder.use_gru:
            gru = encoder.gru.gru
            logger.info(f'Encoders are using GRU with input size {gru.input_size}, hidden size {gru.hidden_size}')
        logger.info(f"After convolution, shape will be (B, {concept_chan}, {concept_height}, {concept_width})")
        logger.info(f'After pooling, shape will be (B, {concept_chan})')

    def get_pooling_module(self, pool_type: str) -> nn.Module:
        """Returns the correct final pooling module to reduce across the channels dimension"""
        accept = ["gem", "max", "avg", "masked-avg", "linear", "conv1x1"]
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
                raise NotImplementedError("Cannot set both `use_attention=True` and `pool_type='linear'`")
            return LinearFlatten()
        elif pool_type == "conv1x1":
            return Conv1x1(in_channels=4)
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

    def extract_concepts(self, x: torch.tensor) -> tuple[torch.tensor]:
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

    def forward_features(self, x: torch.tensor) -> torch.tensor:
        """Returns feature embeddings prior to final linear projection layer(s)"""
        # (batch, channels, height, width) -> list of (batch, 1, features) for each concept
        x = self.extract_concepts(x)
        # If required, apply random masking to concept embeddings during training
        if self.use_masking and self.training:
            x = self.mask(x)
        # Combine list to (batch, channels, features), where channels = 4
        x = torch.cat(x, dim=1)
        # Self-attention across channels
        if self.use_attention:
            x, _ = self.self_attention(x, x, x)
        # Permute to (batch, features, channels)
        x = x.permute(0, 2, 1)
        # Pool across channels
        x = self.pooling(x)
        # Remove singleton dimension
        x = x.squeeze(2)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Model forward pass. Processes single embedding per concept, masks, pools, and projects to class logits."""
        # (batch, 512)
        x = self.forward_features(x)
        # (batch, 128)
        x = self.fc1(x)
        # (batch, num_classes)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    from deep_pianist_identification.dataloader import MIDILoader, remove_bad_clips_from_batch
    from torch.utils.data import DataLoader

    size = 2
    loader = DataLoader(
        MIDILoader(
            'train',
            data_split_dir="25class_0min",
            n_clips=10,
            multichannel=True
        ),
        batch_size=size,
        shuffle=True,
        collate_fn=remove_bad_clips_from_batch
    )
    model = DisentangleNet(
        num_classes=25,
        use_attention=False,
        use_masking=True,
        use_gru=True,
        max_masked_concepts=1,
        layers=DEFAULT_CONV_LAYERS,
        mask_probability=1.0,
        pool_type="avg",
        num_attention_heads=2
    ).to(utils.DEVICE)
    model.train()
    print(utils.total_parameters(model))
    for feat, _, __ in loader:
        embeds = model(feat.to(utils.DEVICE))
        print(embeds.size())
