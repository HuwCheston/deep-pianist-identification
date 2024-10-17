#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Helper for using sklearn decomposition on high-dim tensors. From https://github.com/zhangrh93/InvertibleCE """

import numpy as np
import tensorly as tl
from torch.utils.data import DataLoader

from deep_pianist_identification.dataloader import MIDILoaderTargeted, remove_bad_clips_from_batch


class TuckerDecomposer:
    """Initialize the ChannelDecomposition for tensors.

    Parameters
    ----------
    dimension : int
        the number of dimensions (e.g., 3 for a 3d matrix). Only 3 and 4 are supported.
    rank : list[int]
        list of ranks. Its length must match `dimension`.
    iter_max : int
        number of maximum iteration for the tucker decomposition.
    """

    def __init__(
            self,
            model,
            target_idx: int,
            layer: str,
            num_classes: int = 20,
            data_split_dir: str = "20class_80min",
            dimension: int = 3,
            rank: list[int] = None,
            iter_max: int = 1000,
            batch_size: int = 20,
            non_negative: bool = False
    ):
        # Construct the layer dictionary
        self.model = model
        self.model.eval()
        self.layer = layer
        self.layer_dict: dict[torch.nn.Module] = dict(self.model.named_children())
        assert layer in self.layer_dict.keys(), f'`layer` must be an attribute of `model`, but got `{layer}`'
        # Initialise dataloaders
        dl_cfg = dict(
            data_split_dir=data_split_dir,
            split="validation",
            normalize_velocity=True,
            data_augmentation=False,
            jitter_start=False,
            multichannel=True,
            classify_dataset=False,
            # TODO: update this, set to use melody concept by default
            use_concepts=0
        )
        self.target_loader = DataLoader(
            MIDILoaderTargeted(
                target=target_idx,
                # TODO: just set to a temporary value for now
                n_clips=40,
                **dl_cfg
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        # self.target_acts = self.get_layer_activations(self.target_loader)
        self.other_loader = DataLoader(
            MIDILoaderTargeted(
                target=[i for i in range(num_classes) if i != target_idx],
                n_clips=len(self.target_loader.dataset),
                **dl_cfg
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=remove_bad_clips_from_batch
        )
        # Computed when calling self.get_layer_activations()
        self.target_acts = None
        self.other_acts = None

        # Initialise arguments for tucker decomposition
        self.non_negative = non_negative
        if dimension not in [3, 4]:
            raise NotImplementedError("Only dimension 3 and 4 are supported.")
        self.dimension = dimension
        if len(rank) != dimension:
            raise ValueError("The number of ranks must be the same as dimension.")
        self.rank = rank
        # Default arguments taken straight from repo
        self.precomputed_tensors = None
        self._is_fit = False
        self.trained_shape = None
        self.orig_shape = None
        self._iter_max = iter_max
        self.orig_dataset = None
        self.reducer_conv = None

    def _get_layer_activations(self, dataloader: DataLoader) -> np.ndarray:
        def getter(batch: torch.tensor) -> list[torch.tensor]:
            data_out, handles = [], []

            def hook_out(_, __, o) -> None:
                data_out.append(o)

            # Register forward hooks for the models correctly
            handles.append(self.layer_dict[self.layer].register_forward_hook(hook_out))
            # Pass through the model with our forward hook set correctly
            # We don't need the output from forward directly,
            # we just need to use it to hit the `hook_out` function
            with torch.no_grad():
                batch = batch.to(DEVICE).unsqueeze(1)  # adding an extra channel dimension in
                _ = self.model(batch.to(DEVICE))
            data_out = data_out[0].cpu()
            # Remove all hooks
            for handle in handles:
                handle.remove()
            # Pass through a ReLU which clips all negative values to 0
            if self.non_negative:
                data_out = torch.relu(data_out)
            # This should just return the activations for the specified layer
            return data_out

        # This gives us activations in the form (B, C, H, W)
        acts = [getter(nx) for nx, _, __ in dataloader]
        # Permute to (B, H, W, C) and return
        return torch.cat(acts, 0).permute(0, 2, 3, 1).detach().numpy()

    def get_layer_activations(self):
        """Compute target activation for both target and other dataloaders"""
        # Shape is (B, H, W, C)
        self.target_acts = self._get_layer_activations(self.target_loader)
        self.other_acts = self._get_layer_activations(self.other_loader)

    def _flat_transpose_pad(self, acts, pad=False):
        """Flatten the input matrix A to (c x (h x w) x n)"""
        # acts is the input matrix to factorize.
        # first transpose it to c x h x w x n (this is necessary because we can't fix last mode in tensorly)
        acts = np.swapaxes(acts, 0, -1)
        if self.dimension == 3:
            # flat the h and w dimension to a single one
            acts = acts.reshape(
                [acts.shape[0], acts.shape[1] * acts.shape[2], acts.shape[3]]
            )
        # zero pad it to have size of the original tucker matrix if not already
        if not pad:
            return acts
        else:
            pad_value = self.trained_shape[-1] - acts.shape[-1]
            if self.dimension == 3:
                pad_width = ((0, 0), (0, 0), (0, pad_value))
            else:  # dimension == 4
                pad_width = ((0, 0), (0, 0), (0, 0), (0, pad_value))
            padded_acts = np.pad(acts, pad_width, mode="constant", constant_values=0)
            return padded_acts

    def _inverse_flat_transpose(self, acts, reduced_channel=True):
        """Reshape W according to the initial shape of A"""
        # transpose to put channel at the end and pieces at first
        acts = np.swapaxes(acts, 0, -1)
        # reshape to reconstruct h x w
        if reduced_channel:
            acts = acts.reshape(list(self.orig_shape[:-1]) + [-1])
        else:
            acts = acts.reshape(self.orig_shape)
        return acts

    def fit_transform(self, acts):
        """Fit a tucker model and return the error."""
        # matrix now has the shape n x h x w x c
        self.orig_shape = acts.shape
        self.orig_dataset = acts
        # transpose and flat to c x h x w x c
        acts_flat = self._flat_transpose_pad(acts)
        # now run tucker decomposition
        print("Running tucker on the matrix of shape", acts_flat.shape)
        print(f"Tucker ranks: {self.rank}")
        tensors, error = tl.decomposition.non_negative_tucker_hals(
            acts_flat, rank=self.rank, n_iter_max=self._iter_max, return_errors=True
        )
        print("Minimum Tucker error", error[-1])
        normalized_tensors = normalize_tensors(tensors)
        self.precomputed_tensors = normalized_tensors
        self._is_fit = True
        self.trained_shape = acts_flat.shape
        self.reducer_conv = error
        # this is not returning a transformed version of the data. Behaviour is different from NMF

    def transform(self, acts):
        # indices = []
        equal_axis = (-1, -2, -3) if self.dimension == 4 else (-1, -2)
        indices = [
            np.all(self.orig_dataset == piece, axis=equal_axis).nonzero()[0][0]
            for piece in acts
        ]
        # for piece in acts:
        #     indices.append(np.all(self.orig_dataset==piece, axis = equal_axis).nonzero()[0][0])
        #     # indices.append((self.orig_dataset == piece).all(axis=-1).nonzero()[0][0])
        output = tl.tenalg.multi_mode_dot(
            self.precomputed_tensors.core, self.precomputed_tensors.factors, skip=0
        )
        # reshape and translate the output so we return n x h x w x c'
        output = self._inverse_flat_transpose(output)
        # delete zero-padded pieces
        output = output[indices, :, :, :]
        return output, indices

        # """Return the data transformed by and already fitted Tucker model."""
        # # acts is not the piano roll for only some pieces
        # # first flat transpose and pad
        # padded_acts = self._flat_transpose_pad(acts, pad=True)
        # # now tucker decompose with fixed modes (except piece mode)
        # fixed_modes = [0, 1] if self.dimension == 3 else [0, 1, 2]
        # (core, factors), errors = tl.decomposition.non_negative_tucker_hals(
        #     padded_acts,
        #     rank=self.rank,
        #     return_errors=True,
        #     n_iter_max=self._iter_max,
        #     fixed_modes=fixed_modes,
        #     init=self.precomputed_tensors.tucker_copy(),
        # )
        # print("Minimum fixed Mode Tucker error", errors[-1])
        # # 3 and 2 mode multiplication, skip channel-mode mult
        # output = tensorly.tenalg.multi_mode_dot(core, factors, skip=0)
        # # reshape and translate the output so we return n x h x w x c'
        # output = self._inverse_flat_transpose(output)
        # # delete zero-padded pieces
        # output = output[: acts.shape[0], :, :, :]
        # return output

    def inverse_transform(self, _, indices):
        if self._is_fit:
            reconstructed = tl.tucker_to_tensor(self.precomputed_tensors)
            # transpose and reshape it in original shape n x h x w x c
            reconstructed = self._inverse_flat_transpose(
                reconstructed, reduced_channel=False
            )
            return reconstructed[indices, :, :, :]
        else:
            raise Exception("The Reducer must be fit first")

    # def inverse_transform(self, acts):
    #     # flat transpose
    #     padded_acts = self._flat_transpose_pad(acts, pad=True)
    #     if self._is_fit:
    #         # only step missing to the reconstructed matrix is the 1-mode multiplication
    #         # 3 and 2 mode mult has been performed in transform()
    #         mode1_matrix = self.precomputed_tensors.factors[0]
    #         reconstructed = tensorly.tenalg.mode_dot(padded_acts, mode1_matrix, 0)
    #         # transpose and reshape it in original shape n x h x w x c
    #         reconstructed = self._inverse_flat_transpose(
    #             reconstructed, reduced_channel=False
    #         )
    #         return reconstructed[: acts.shape[0], :, :, :]
    #     else:
    #         raise Exception("The Reducer must be fit first")


def normalize_tensors(tucker_tensor):
    """Returns tucker_tensor with factors normalised to unit length with the normalizing constants absorbed into
    `core`.
    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    Returns
    -------
    TuckerTensor((core, factors))
    """
    core, factors = tucker_tensor
    normalized_factors = []
    for i, factor in enumerate(factors):
        scales = tl.norm(factor, axis=0)
        scales_non_zero = tl.where(
            scales == 0, tl.ones(tl.shape(scales), **tl.context(factor)), scales
        )
        core = core * tl.reshape(
            scales, (1,) * i + (-1,) + (1,) * (tl.ndim(core) - i - 1)
        )
        normalized_factors.append(factor / tl.reshape(scales_non_zero, (1, -1)))
    return tl.tucker_tensor.TuckerTensor((core, normalized_factors))


if __name__ == "__main__":
    import os

    import torch
    import yaml

    from deep_pianist_identification.training import DEFAULT_CONFIG, TrainModule
    from deep_pianist_identification.encoders import DisentangleNet
    from deep_pianist_identification.utils import get_project_root, DEVICE

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
    melody_concept = disentangle_model.melody_concept
    td = TuckerDecomposer(
        model=melody_concept,
        target_idx=1,
        layer="layer4",
        dimension=4,
        # Replace 13 and 3 with height/width (check which?)
        rank=[4, 13, 3, 375],
    )
    td.get_layer_activations()
