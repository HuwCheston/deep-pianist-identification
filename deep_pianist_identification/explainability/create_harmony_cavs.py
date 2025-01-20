#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create and plot harmony CAVs using MIDI from Dan Haerle's 'Jazz Piano Voicing Skills'"""

import json
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

# We'll dump everything we create here
OUTPUT_DIR = os.path.join(utils.get_project_root(), "reports/figures/ablated_representations/cav_plots")
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)  # create folder if it doesn't already exist


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
        track_names: np.ndarray,
        class_mapping: dict,
        n_experiments: int,
        n_cavs: int,
        attribution_fn: str = "gradient_x_activation",
        multiply_by_inputs: bool = True,
        batch_size: int = cav_utils.BATCH_SIZE
) -> list[cav_utils.CAV]:
    all_cavs = []
    for idx in range(1, n_cavs + 1):
        logger.info(f'Creating CAV {idx}: {cav_utils.CAV_MAPPING[idx - 1]}...')  # log name and index of CAV
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
        cav.interpret(features, targets, class_mapping, track_names)
        all_cavs.append(cav)
    return all_cavs


def create_random_cav(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        features: torch.tensor,
        targets: torch.tensor,
        track_names: np.ndarray,
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
    rc.interpret(features, targets, class_mapping, track_names)
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
        track_names: np.ndarray,
        class_mapping: dict,
        attribution_fn: str,
        multiply_by_inputs: bool,
        n_experiments: int,
        batch_size: int,
        n_cavs: int,
        n_random_clips: int,
) -> tuple[list[cav_utils.CAV], cav_utils.CAV]:
    """Load CAVs from disk at `save_loc` or create them from scratch with given arguments"""
    logger.info('Creating CAVs...')
    try:
        # Try unpickling all objects at provided location
        cav_list = pickle.load(open(os.path.join(OUTPUT_DIR, 'cavs.p'), 'rb'))
        random_cav = pickle.load(open(os.path.join(OUTPUT_DIR, 'random_cav.p'), 'rb'))
    except FileNotFoundError:
        logger.info(f"... couldn't find CAVs at {OUTPUT_DIR}, creating them from scratch!")
        # Create concept AVs
        cav_args = dict(
            model=model,
            layer=layer,
            features=features,
            targets=targets,
            track_names=track_names,
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
        # Dump CAVs to disk
        with open(os.path.join(OUTPUT_DIR, 'cavs.p'), 'wb') as f:
            pickle.dump(cav_list, f)
            logger.info(f'... dumped fitted CAV classes to {OUTPUT_DIR}/cavs.p!')
        # Dump random CAVs to disk
        with open(os.path.join(OUTPUT_DIR, 'random_cav.p'), 'wb') as f:
            pickle.dump(random_cav, f)
            logger.info(f'... dumped random CAV class to {OUTPUT_DIR}/random_cav.p!')
        # Dump clip sensitivities for all concepts to a single JSON within our output dictionary
        with open(os.path.join(OUTPUT_DIR, 'clip_sensitivities.json'), 'w') as f:
            json.dump({cav.cav_name: cav.clip_sensitivity_dict for cav in cav_list}, f, ensure_ascii=False, indent=4)
            logger.info(f'... dumped clip sensitivity JSON to {OUTPUT_DIR}/clip_sensitivities.json!')
    else:
        logger.info(f'... loaded {len(cav_list)} concept CAVs from {OUTPUT_DIR}!')
        logger.info(f'... loaded random CAV from {OUTPUT_DIR}!')
    return cav_list, random_cav


def create_static_sensitivity_heatmap(
        clip_paths: list[str],
        cavs: list[cav_utils.CAV],
        class_mapping: dict
) -> None:
    """Creates kernel heatmaps for all combinations of tracks and CAVs"""
    # If we've somehow only ended up with a single string
    if isinstance(clip_paths, str):
        # Convert this to a list of strings
        clip_paths = [clip_paths]
    # Iterate over all the clips we've passed in
    for clip_path in clip_paths:
        # Formatting the clip path correctly
        if not clip_path.endswith('.mid'):
            clip_path += '.mid'
        full_clip_path = os.path.join(utils.get_project_root(), 'data/clips', clip_path)
        assert os.path.isfile(full_clip_path), f"Could not find MIDI at {clip_path}!"
        # Iterate over all concepts for this clip
        for cav_idx in tqdm(range(len(cavs)), f'Creating heatmaps for clip {clip_path}...'):
            # Create the slider instance: slides a kernel over transcription and computes change in sensitivity to CAV
            slider = cav_utils.CAVKernelSlider(
                clip_path=full_clip_path,
                cav=cavs[cav_idx],
                kernel_size=(24, 2.5),
                stride=(2.0, 2.0),
                class_mapping=class_mapping
            )
            # Create the plot, which takes in the attributes computed from the CAVKernelSlider class
            hm = plotting.HeatmapCAVKernelSensitivity(
                clip_path=full_clip_path,
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
        batch_size: int,
        sensitivity_heatmap_clips: list[str]
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
        model, layer, features, targets, track_names, tm.class_mapping, attribution_fn,
        multiply_by_inputs, n_experiments, batch_size, n_cavs, n_random_clips,
    )
    # Shape (n_cavs, n_performers, n_experiments)
    all_sign_counts = np.stack([cav.sign_counts for cav in cav_list])
    # Shape (n_performers, n_cavs)
    mean_sign_counts = all_sign_counts.mean(axis=2).T
    # Shape (n_performers, n_cavs)
    ast_sign_counts = cav_utils.get_sign_count_significance(all_sign_counts, random_cav.sign_counts).T
    # Create heatmap with significance asterisks
    hm = plotting.HeatmapCAVSensitivity(
        sensitivity_matrix=mean_sign_counts,
        class_mapping=tm.class_mapping,
        cav_names=cav_utils.CAV_MAPPING,
        performer_birth_years=cav_utils.BIRTH_YEARS,
        significance_asterisks=ast_sign_counts
    )
    hm.create_plot()
    hm.save_fig()
    # Create correlation heatmap between sign-count values for different concepts
    # Flatten to (n_performers * n_experiments, n_cavs)
    flat_sign_counts = all_sign_counts.reshape(all_sign_counts.shape[0], -1).T
    hmpc = plotting.HeatmapCAVPairwiseCorrelation(
        cav_matrix=flat_sign_counts,
        cav_mapping=cav_utils.CAV_MAPPING
    )
    hmpc.create_plot()
    hmpc.save_fig()
    # Create plots for individual tracks
    create_static_sensitivity_heatmap(sensitivity_heatmap_clips, cav_list, tm.class_mapping)


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
        batch_size=args['batch_size'],
        sensitivity_heatmap_clips=args['sensitivity_heatmap_clips']
    )
