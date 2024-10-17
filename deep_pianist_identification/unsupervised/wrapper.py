#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PyTorch model wrapper, adapted from original repo (https://github.com/zhangrh93/InvertibleCE)"""

import os

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader

from deep_pianist_identification.utils import DEVICE, PIANO_KEYS, FPS, CLIP_LENGTH, get_project_root

__all__ = ["ModelWrapper"]


class ModelWrapper:
    def __init__(
            self,
            model: torch.nn.Module,
            layer_dict: dict = None,
            predict_target=None,
            input_channel_first: bool = True,  # True if input image is channel first
            model_channel_first: bool = True,  # True if model use channel first
            # switch_channel = None, #"f_to_l" or "l_to_f" if switch channel is required from loader to model
            numpy_out: bool = True,
            input_size: list = None,  # model's input size
            batch_size: int = 20,
    ):  # target: (layer_name,unit_nums)

        self.model = model
        self.batch_size = batch_size
        self.layer_dict = layer_dict if layer_dict is not None else dict()
        self.layer_dict.update(dict(model.named_children()))
        self.predict_target = predict_target
        self.input_channel = "f" if input_channel_first else "l"
        self.model_channel = "f" if model_channel_first else "l"
        self.numpy_out = numpy_out
        self.input_size = list(input_size) if input_size is not None else [1, PIANO_KEYS, FPS * CLIP_LENGTH]
        # Should this be set to True?
        self.non_negative = False

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = torch.clone(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _switch_channel_f_to_l(x):  # transform from channel first to channel last
        if x.ndim == 3:
            x = x.permute(1, 2, 0)
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def _switch_channel_l_to_f(x):  # transform from channel last to channel first
        if x.ndim == 3:
            x = x.permute(2, 0, 1)
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        return x

    def _switch_channel(self, x, layer_in="input", layer_out="output", to_model=True):
        if to_model:
            c_from = self.input_channel if layer_in == "input" else "l"
            c_to = self.model_channel
        else:
            c_from = self.model_channel
            c_to = "l"

        if c_from == "f" and c_to == "l":
            x = self._switch_channel_f_to_l(x)
        if c_from == "l" and c_to == "f":
            x = self._switch_channel_l_to_f(x)
        return x

    def _fun(self, x, layer_in: str = "input", layer_out: str = "output"):
        """Wrapper function that returns activations from a model layer"""
        x = x.type(torch.FloatTensor)
        data_in = x.clone().to(DEVICE)
        data_out = []
        handles = []

        def hook_in(*_):
            return data_in

        def hook_out(_, __, o):
            data_out.append(o)

        # This is the case by default
        if layer_in == "input":
            nx = x
        else:
            handles.append(self.layer_dict[layer_in].register_forward_hook(hook_in))
            nx = torch.zeros([x.size()[0]] + self.input_size)
        # Register forward hooks for the models correctly
        if not layer_out == "output":
            handles.append(self.layer_dict[layer_out].register_forward_hook(hook_out))
        # Pass through the model with our forward hook set correctly
        with torch.no_grad():
            ny = self.model(nx.to(DEVICE))
        # If we want the output from the final layer, just take it directly
        if layer_out == "output":
            data_out = ny
        else:
            data_out = data_out[0]
        # Set device correctly
        data_out = data_out.cpu()
        # Remove all handles
        for handle in handles:
            handle.remove()
        # Pass through a ReLU which clips all negative values to 0
        if self.non_negative:
            data_out = torch.relu(data_out)
        # This should just return the activations for the specified layer
        return data_out

    def _batch_fn(self, x, layer_in="input", layer_out="output"):
        # numpy in numpy out
        if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            x = self._to_tensor(x)
            dataset = TensorDataset(x)
            x = DataLoader(dataset, batch_size=self.batch_size)

        out = [self._fun(nx[0], layer_in, layer_out) for nx in x]
        res = torch.cat(out, 0)

        # This seems to be switching channels to (B, H, W, C)
        res = self._switch_channel(
            res, layer_in=layer_in, layer_out=layer_out, to_model=False
        )
        if self.numpy_out:
            res = res.detach().numpy()

        return res

    def set_predict_target(self, predict_target):
        self.predict_target = predict_target

    def get_feature(self, x, layer_name: str):
        if layer_name not in self.layer_dict.keys():
            print("Target layer does not exist")
            return None
        return self._batch_fn(x, layer_out=layer_name)

    def feature_predict(self, feature, layer_name: str = None):
        if layer_name not in self.layer_dict.keys():
            print("Target layer does not exist")
            return None

        out = self._batch_fn(feature, layer_in=layer_name)
        if self.predict_target is not None:
            out = out[:, self.predict_target]
        return out

    def predict(self, x):
        out = self._batch_fn(x)
        if self.predict_target is not None:
            out = out[:, self.predict_target]
        return out


if __name__ == "__main__":
    from deep_pianist_identification.training import DEFAULT_CONFIG, TrainModule
    from deep_pianist_identification.encoders import DisentangleNet

    # Define the name of the model we're going to wrap
    MODEL_NAME = "disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool"
    # Load in the config path for the model
    cfg_path = os.path.join(get_project_root(), 'config', 'disentangle-resnet', MODEL_NAME + '.yaml')
    cfg = yaml.safe_load(open(cfg_path))
    # Replace any non-existing keys with their default value
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg.keys():
            cfg[k] = v
    # Initialise the training module here as this will create the model and automatically load the checkpoint
    tm = TrainModule(**cfg)
    # Simply grab the checkpointed model from the training module
    disentangle_model: DisentangleNet = tm.model.to(DEVICE)
    disentangle_model.eval()
    # Wrap the melody ResNet from the model with the wrapper
    melody_concept = disentangle_model.melody_concept
    wrappedup = ModelWrapper(melody_concept, layer_dict=dict(melody_concept.named_children()))
    # Create a random batch of inputs in the expected format (B, C, H, W)
    rand = torch.rand(10, 1, PIANO_KEYS, FPS * CLIP_LENGTH)
    # Get the activations for the final layer of the ResNet, before the classification head
    # Output is (B, C, H, W)
    layer4_act = wrappedup.get_feature(rand, "layer4")
