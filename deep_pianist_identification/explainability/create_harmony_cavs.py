#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create and plot harmony CAVs using MIDI from Dan Haerle's 'Jazz Piano Voicing Skills'"""

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.explainability import cav_utils
from deep_pianist_identification.training import DEFAULT_CONFIG, TrainModule


def get_training_module(cfg_path: str) -> TrainModule:
    """Create the training module with given config"""
    # Load in the config path for the model
    cfg = yaml.safe_load(open(cfg_path))
    # Replace any non-existing keys with their default value
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg.keys():
            cfg[k] = v
    # Create the training module with the required config and return
    return TrainModule(**cfg)


def create_all_cavs(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        features: torch.tensor,
        targets: torch.tensor,
        class_mapping: dict,
        n_experiments: int,
        n_cavs: int,
        attribution_fn: str = "gradient_x_activation",
        multiply_by_inputs: bool = True,
        batch_size: int = cav_utils.BATCH_SIZE
) -> list[cav_utils.CAV]:
    all_cavs = []
    for idx in range(1, n_cavs + 1):
        logger.info(f'Creating CAV {idx}...')
        cav = cav_utils.CAV(
            cav_idx=idx,
            model=model,
            layer=layer,
            layer_attribution=attribution_fn,
            multiply_by_inputs=multiply_by_inputs,
            batch_size=batch_size
        )
        cav.fit(n_experiments=n_experiments)
        # Interpret using the features and targets
        cav.interpret(features, targets, class_mapping)
        all_cavs.append(cav)
    return all_cavs


def create_random_cav(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        features: torch.tensor,
        targets: torch.tensor,
        class_mapping: dict,
        n_experiments: int,
        n_clips: int,
        attribution_fn: str = "gradient_x_activation",
        multiply_by_inputs: bool = True,
        batch_size: int = cav_utils.BATCH_SIZE
) -> cav_utils.RandomCAV:
    rc = cav_utils.RandomCAV(
        n_clips=n_clips,
        model=model,
        layer=layer,
        layer_attribution=attribution_fn,
        multiply_by_inputs=multiply_by_inputs,
        batch_size=batch_size
    )
    rc.fit(n_experiments=n_experiments)
    rc.interpret(features, targets, class_mapping)
    return rc


def get_all_data(loaders: list) -> tuple[torch.tensor, torch.tensor, np.array]:
    """Get all features and targets from all dataloaders"""
    features, targets, track_names = [], [], []
    # Foscarin et al. only use (unseen/held-out) test data
    for loader in loaders:
        # This is the number of tracks in the dataloader
        idxs = range(len(loader.dataset))
        # Iterate over all indexes
        for idx in tqdm(idxs, desc='Getting inputs...'):
            # Unpack the idx to get clip metadata
            track_path, tmp_class, _, clip_idx = loader.dataset.clips[idx]
            # Get the corresponding batched output
            batch = loader.dataset.__getitem__(idx)
            # Skip over cases where piano roll can't be generated correctly
            if batch is None:
                continue
            # Unpack the batch
            roll, targ, tmp_path = batch
            # Get the full path to this clip
            path = os.path.join(track_path, f'clip_{str(clip_idx).zfill(3)}.mid')
            # Check to make sure everything is correct
            assert tmp_class == targ
            assert track_path.split(os.path.sep)[-1] == tmp_path
            # Append everything to our lists
            features.append(roll)
            targets.append(targ)
            track_names.append(path)
    return torch.tensor(np.stack(features)), torch.tensor(targets), np.hstack(track_names)


def create_or_load_cavs(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        features: torch.tensor,
        targets: torch.tensor,
        class_mapping: dict,
        attribution_fn: str,
        multiply_by_inputs: bool,
        n_experiments: int,
        batch_size: int,
        n_cavs: int,
        n_random_clips: int,
        save_loc: str = os.path.join(utils.get_project_root(), 'references/cavs')
) -> tuple[list[cav_utils.CAV], cav_utils.CAV]:
    """Load CAVs from disk at `save_loc` or create them from scratch with given arguments"""
    logger.info('Creating CAVs...')
    try:
        # Try unpickling all objects at provided location
        cav_list = pickle.load(open(os.path.join(save_loc, 'cavs.p'), 'rb'))
        random_cav = pickle.load(open(os.path.join(save_loc, 'random_cav.p'), 'rb'))
    except FileNotFoundError:
        logger.info(f"... couldn't find CAVs at {save_loc}, creating them from scratch!")
        # Create concept AVs
        cav_args = dict(
            model=model,
            layer=layer,
            features=features,
            targets=targets,
            class_mapping=class_mapping,
            attribution_fn=attribution_fn,
            multiply_by_inputs=multiply_by_inputs,
            n_experiments=n_experiments,
            batch_size=batch_size
        )
        cav_list = create_all_cavs(**cav_args, n_cavs=n_cavs)
        logger.info(f'... created {len(cav_list)} concept CAVs!')
        # Create random AVs
        random_cav = create_random_cav(**cav_args, n_clips=n_random_clips)
        logger.info(f'... created random CAV!')
        # Dump CAVs and random CAV to disk
        with open(os.path.join(save_loc, 'cavs.p', 'wb')) as f:
            pickle.dump(cav_list, f)
        with open(os.path.join(save_loc, 'random_cav.p', 'wb')) as f:
            pickle.dump(random_cav, f)
        logger.info(f'... dumped CAVs to {save_loc}!')
    else:
        logger.info(f'... loaded {len(cav_list)} concept CAVs from {save_loc}!')
        logger.info(f'... loaded random CAV from {save_loc}!')
    return cav_list, random_cav


def create_kernel_heatmaps(track_names: list[str], cavs: list[cav_utils.CAV], class_mapping: dict) -> None:
    """Creates kernel heatmaps for all combinations of tracks and CAVs"""
    for track_name in tqdm(track_names, desc='Creating heatmaps...'):
        for cav_idx in range(len(cavs)):
            # Create the slider instance: slides a kernel over transcription and comptues change in sensitivity to CAV
            slider = cav_utils.CAVKernelSlider(
                clip_path=track_name,
                cav=cavs[cav_idx],
                kernel_size=(24, 2.5),
                stride=(2.0, 2.0),
                class_mapping=class_mapping
            )
            # Create the plot, which takes in the attributes computed from the CAVKernelSlider class
            hm = plotting.HeatmapCAVKernelSensitivity(
                clip_path=track_name,
                sensitivity_array=slider.compute_kernel_sensitivities(),
                clip_roll=slider.clip_roll,
                cav_name=cav_utils.CAV_MAPPING[cav_idx],
            )
            hm.create_plot()
            hm.save_fig()


def main(
        model: str,
        attribution_fn: str,
        multiply_by_inputs: bool,
        n_experiments: int,
        n_random_clips: int,
        n_cavs: int,
        batch_size: int
):
    """Creates all CAVs, generates similarity matrix, and saves plots"""
    logger.info(f"Initialising model with config file {model}")
    # Get the config file for training the model
    cfg_loc = os.path.join(utils.get_project_root(), 'config', model + '.yaml')
    if not os.path.isfile(cfg_loc):
        raise FileNotFoundError(f"Model {cfg_loc} does not have a config file associated with it!")
    # Get the training module
    tm = get_training_module(cfg_loc)
    model = tm.model.to(utils.DEVICE)
    model.eval()
    # Use the default layer
    # TODO: this should be changeable based on an argument
    layer = model.harmony_concept.layer4
    # Get all data from held-out sets
    alldata = [tm.test_loader, tm.validation_loader]
    logger.info('Getting data...')
    features, targets, track_names = get_all_data(alldata)
    logger.info(f"... got data with shape {features.shape}")
    # Load CAVs from disk or create them from scratch
    cav_list, random_cav = create_or_load_cavs(
        model, layer, features, targets, tm.class_mapping, attribution_fn,
        multiply_by_inputs, n_experiments, batch_size, n_cavs, n_random_clips,
    )
    # Log accuracies for all CAVs
    all_accs = np.concatenate([i.acc for i in cav_list])
    for func, name in zip([np.mean, np.std, np.max, np.min], ['mean', 'std', 'max', 'min']):
        logger.info(f'CAV accuracy {name}: {func(all_accs):.5f}')
    # Get matrices of average sign count and p-values
    all_sign_counts = np.column_stack([i.sign_counts.mean(axis=1) for i in cav_list])
    # p_vals = np.column_stack([cav_utils.get_pvals(sc.sign_counts, random_cav.sign_counts) for sc in cav_list])
    # Convert p-values to asterisks (with Bonferroni correction for multiple tests)
    # ast = cav_utils.get_pval_significance(p_vals)
    # All should be in the shape: (n_cavs, n_performers)
    # assert p_vals.shape == all_sign_counts.shape == ast.shape
    # Create heatmap with significance asterisks
    hm = plotting.HeatmapCAVSensitivity(
        sensitivity_matrix=all_sign_counts,
        class_mapping=tm.class_mapping,
        cav_names=cav_utils.CAV_MAPPING,
        performer_birth_years=cav_utils.BIRTH_YEARS,
        significance_asterisks=np.empty_like(all_sign_counts, dtype=str)
    )
    hm.create_plot()
    hm.save_fig()
    # Create correlation heatmap between sign-count values for different concepts
    sign_counts = np.stack([c.sign_counts.flatten() for c in cav_list]).T
    hmpc = plotting.HeatmapCAVPairwiseCorrelation(sign_counts, cav_utils.CAV_MAPPING)
    hmpc.create_plot()
    hmpc.save_fig()
    # Create plots for individual tracks
    create_kernel_heatmaps(track_names, cav_list, tm.class_mapping)


if __name__ == '__main__':
    # For reproducible results
    utils.seed_everything(utils.SEED)
    # Declare argument parser and add arguments
    parser = ArgumentParser(description='Create TCAVs for DisentangleNet models')
    args = cav_utils.parse_arguments(parser)
    logger.info(f'Creating CAVs using args {args}')
    # Run the CAV creation and plotting script with given arguments from the console
    main(
        model=args["model"],
        attribution_fn=args["attribution_fn"],
        multiply_by_inputs=args["multiply_by_inputs"],
        n_experiments=args["n_experiments"],
        n_random_clips=args["n_random_clips"],
        n_cavs=args['n_cavs'],
        batch_size=args['batch_size']
    )
