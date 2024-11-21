#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create and plot harmony CAVs using MIDI from Dan Haerle's 'Jazz Piano Voicing Skills'"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.explainability.cav_utils import CAV, TCAV
from deep_pianist_identification.training import DEFAULT_CONFIG, TrainModule

DEFAULT_MODEL = ("disentangle-resnet-channel/"
                 "disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc")
# Array of names for all CAVs
_mapping_loc = os.path.join(utils.get_project_root(), 'references/_unsupervised_resources/voicings/cav_mapping.csv')
CAV_MAPPING = np.array([row.iloc[0] for i, row in pd.read_csv(_mapping_loc, index_col=1).iterrows()])
# Array of pianist birth years (not sorted)
BIRTH_YEARS = np.genfromtxt(
    os.path.join(utils.get_project_root(), 'references/data_splits/20class_80min/birth_years.txt'), dtype=int
)


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
        attribution_fn: str = "gradient_x_activation",
        multiply_by_inputs: bool = True,
        n_cavs: int = 20,
) -> list[CAV]:
    explainers = []
    for idx in tqdm(range(1, n_cavs + 1), desc='Creating all CAVs: '):
        # Create the Explainer instance, passing in the correct dataloaders, and fit
        explain = CAV(
            cav_idx=idx,
            model=model,
            layer=layer,
            layer_attribution=attribution_fn,
            multiply_by_inputs=multiply_by_inputs
        )
        explain.fit()
        # Interpret using the features and targets
        explain.interpret(features, targets, class_mapping)
        explainers.append(explain)
    return explainers


def create_all_tcavs(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        features: torch.tensor,
        targets: torch.tensor,
        class_mapping: dict,
        attribution_fn: str = "gradient_x_activation",
        multiply_by_inputs: bool = True,
        n_cavs: int = 20,
        n_experiments: int = 10
) -> list[TCAV]:
    explainers = []
    for idx in tqdm(range(1, n_cavs + 1), desc='Creating all TCAVs: '):
        # Create the Explainer instance, passing in the correct dataloaders, and fit
        explain = TCAV(
            n_experiments=n_experiments,
            cav_idx=idx,
            model=model,
            layer=layer,
            layer_attribution=attribution_fn,
            multiply_by_inputs=multiply_by_inputs
        )
        explain.fit()
        # Interpret using the features and targets
        explain.interpret(features, targets, class_mapping)
        explainers.append(explain)
    return explainers


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


def main(
        model: str,
        attribution_fn: str,
        multiply_by_inputs: bool,
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
    # Get all data from held-out sets
    alldata = [tm.test_loader, tm.validation_loader]
    logger.info('Getting data...')
    features, targets, track_names = get_all_data(alldata)
    logger.info(f"... got data with shape {features.shape}")
    # Interpret all CAVs
    logger.info('Creating CAVs...')
    interpreted = create_all_cavs(
        model,
        model.harmony_concept.layer4,
        features,
        targets,
        tm.class_mapping,
        attribution_fn=attribution_fn,
        multiply_by_inputs=multiply_by_inputs,
        n_cavs=20
    )
    logger.info(f'... created {len(interpreted)} CAVs')
    # Log CAV classifier accuracy
    all_accs = np.array([i.acc for i in interpreted])
    logger.info(f'... mean classifier acc: {np.mean(all_accs):.5f}, SD: {np.std(all_accs):.5f}')
    logger.info(f'... min: {np.min(all_accs):.5f}, max: {np.max(all_accs):.5f}')
    # Get sign counts and magnitudes for all performers
    # Each `i.sign_count` is a 1D vector of shape (N_performers), so we stack to get (N_performers, N_cavs)
    sign_counts = np.column_stack([i.sign_counts for i in interpreted])
    magnitudes = np.column_stack([i.magnitudes for i in interpreted])
    # Create heatmaps
    for df, name in zip([sign_counts, magnitudes], ['TCAV Sign Count', 'TCAV Magnitude']):
        hm = plotting.HeatmapCAVSensitivity(
            sensitivity_matrix=df,
            class_mapping=tm.class_mapping,
            cav_names=CAV_MAPPING,
            performer_birth_years=BIRTH_YEARS,
            sensitivity_type=name
        )
        hm.create_plot()
        hm.save_fig()
    # Create pairwise correlation heatmaps for sensitivity
    cav_df = pd.DataFrame(np.vstack([i.sens for i in interpreted]).T, columns=CAV_MAPPING)
    hmpc = plotting.HeatmapCAVPairwiseCorrelation(cav_df)
    hmpc.create_plot()
    hmpc.save_fig()

    # TCAVS
    logger.info('Creating TCAVs...')
    tcavs = create_all_tcavs(
        model,
        model.harmony_concept.layer4,
        features,
        targets,
        tm.class_mapping,
        attribution_fn=attribution_fn,
        multiply_by_inputs=multiply_by_inputs,
        n_cavs=20
    )
    tcav_sign_counts = np.column_stack([i.sign_counts for i in tcavs])
    np.savetxt('sign_counts.txt', tcav_sign_counts)
    tcav_pvals = np.column_stack([i.p_vals for i in tcavs])
    np.savetxt('pvals.txt', tcav_pvals)


def parse_me(argparser: ArgumentParser) -> dict:
    """Adds required CLI arguments to an `ArgumentParser` object and parses them to a dictionary"""
    argparser.add_argument(
        '-m', '--model',
        default=DEFAULT_MODEL,
        type=str,
        help="Name of a trained model with saved checkpoints"
    )
    argparser.add_argument(
        '-a', '--attribution-fn',
        default='gradient_x_activation',
        type=str,
        help='Function to use to compute layer attributions (see captum.TCAV)'
    )
    argparser.add_argument(
        '-i', '--multiply-by-inputs',
        default=True,
        type=bool,
        help='Multiply layer activations by input (see captum.TCAV)'
    )
    return vars(parser.parse_args())


if __name__ == '__main__':
    # For reproducible results
    utils.seed_everything(utils.SEED)
    # Declare argument parser and add arguments
    parser = ArgumentParser(description='Create TCAVs for DisentangleNet models')
    args = parse_me(parser)
    # Run the CAV creation and plotting script with given arguments from the console
    main(
        model=args["model"],
        attribution_fn=args["attribution_fn"],
        multiply_by_inputs=args["multiply_by_inputs"]
    )
