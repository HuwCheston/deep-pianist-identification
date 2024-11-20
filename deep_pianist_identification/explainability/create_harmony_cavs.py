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
from deep_pianist_identification.explainability.cav_utils import ConceptExplainer
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


def get_all_cavs(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        attribution_fn: str = "gradient_x_activation",
        multiply_by_inputs: bool = True,
        n_cavs: int = 20
) -> list[ConceptExplainer]:
    """Get CAV coefficient vectors for all concepts"""
    explainers = []
    for idx in tqdm(range(1, n_cavs + 1), desc='Creating all CAVs: '):
        # Create the Explainer instance, passing in the correct dataloaders, and fit
        explain = ConceptExplainer(
            cav_idx=idx,
            model=model,
            layer=layer,
            layer_attribution=attribution_fn,
            multiply_by_inputs=multiply_by_inputs
        )
        explain.fit()
        explainers.append(explain)
    # Concatenate all the CAV coefficients to an array of (n_cavs, d)
    return explainers


def interpret_all_cavs(
        cavs: list[ConceptExplainer],
        features: torch.tensor,
        targets: torch.tensor,
        class_mapping: dict
) -> list[ConceptExplainer]:
    interpreted_cavs = []
    for cav in cavs:
        cav.interpret(features, targets, class_mapping)
        interpreted_cavs.append(cav)
    return interpreted_cavs


def get_all_features_targets(loaders: list) -> tuple[torch.tensor, torch.tensor]:
    """Get all features and targets from all dataloaders"""
    features, targets = [], []
    for loader in loaders:
        for rolls, targs, _ in tqdm(loader, total=len(loader), desc="Getting inputs..."):
            features.append(rolls)
            targets.append(targs)
    return torch.cat(targets), torch.cat(features)


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
    # Extract all CAVs
    logger.info("Extracting CAVs...")
    explainers = get_all_cavs(
        model=model,
        layer=model.harmony_concept.layer4,
        attribution_fn=attribution_fn,
        multiply_by_inputs=multiply_by_inputs
    )
    logger.info(f'... got {len(explainers)} CAVs')
    # Get all data from held-out sets
    alldata = [tm.test_loader, tm.validation_loader]
    logger.info('Getting data...')
    features, targets = get_all_features_targets(alldata)
    logger.info(f"... got data with shape {features.shape}")
    # Interpret all CAVs
    logger.info('Interpreting CAVs...')
    interpreted = interpret_all_cavs(explainers, features, targets, tm.class_mapping)
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

    # # Create heatmaps for top-K clips for all CAVs
    # logger.info(f'... CAV heatmaps will be size {heatmap_size} using kernel {kernel_size}')
    # for cav_idx, (cav_name, topk_tracks, cav) in enumerate(zip(CAV_MAPPING, list(topk_dict.values()), cavs)):
    #     for track in topk_tracks:
    #         # Create the non-interactive (matplotlib) heatmap
    #         hmks = HeatmapCAVKernelSensitivity(
    #             track,
    #             cav,
    #             encoder=harmony_concept,
    #             cav_name=cav_name,
    #             extractor_cls=HarmonyExtractor,
    #             cav_type="Voicings"
    #         )
    #         hmks.create_plot()
    #         hmks.save_fig()
    #         logger.info(f'... kernel sensitivity plot done for track {track}, CAV {cav_name}')
    #         # Create the interactive (plotly) heatmap
    #         hmks = HeatmapCAVKernelSensitivityInteractive(
    #             track,
    #             cav,
    #             encoder=harmony_concept,
    #             cav_name=cav_name,
    #             extractor_cls=HarmonyExtractor,
    #             cav_type="Voicings",
    #             cav_idx=cav_idx
    #         )
    #         hmks.create_plot()
    #         hmks.save_fig()
    #         logger.info(f'... interactive kernel sensitivity plot done for track {track}, CAV {cav_name}')


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

    )
