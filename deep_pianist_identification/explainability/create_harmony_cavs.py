#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create and plot harmony CAVs using MIDI from Dan Haerle's 'Jazz Piano Voicing Skills'"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.explainability.cav_dataloader import VoicingLoaderFake, VoicingLoaderReal
from deep_pianist_identification.explainability.cav_explainer import ConceptExplainer
from deep_pianist_identification.explainability.cav_plotting import (
    HeatmapPianistCAV, HeatmapCAVKernelSensitivity, HeatmapCAVKernelSensitivityInteractive
)
from deep_pianist_identification.extractors import HarmonyExtractor
from deep_pianist_identification.training import DEFAULT_CONFIG, TrainModule

DEFAULT_MODEL = ("disentangle-resnet-channel/"
                 "disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc")
_mapping_loc = os.path.join(utils.get_project_root(), 'references/_unsupervised_resources/voicings/cav_mapping.csv')
CAV_MAPPING = np.array([row.iloc[0] for i, row in pd.read_csv(_mapping_loc, index_col=1).iterrows()])

TOPK = 5


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


def get_all_cavs(concept_encoder, n_cavs: int = 20) -> np.ndarray:
    """Get CAV coefficient vectors for all concepts"""
    explainers = []
    for idx in tqdm(range(1, n_cavs + 1), desc='Creating all CAVs: '):
        # Create the Explainer instance, passing in the correct dataloaders, and fit
        explain = ConceptExplainer(
            cav_idx=idx,
            model=concept_encoder,
            real_loader=VoicingLoaderReal,
            fake_loader=VoicingLoaderFake,
        )
        explain.fit()
        explainers.append(explain)
    # Concatenate all the CAV coefficients to an array of (n_cavs, d)
    return np.concatenate([i.cavs for i in explainers])


def embed_all_clips(loaders: list, model, roll_idx: int = 1) -> tuple[np.array, np.array, np.array]:
    """Create all clip embeddings. Returns arrays of embeddings, target indices, and track names"""
    actual_embeds, actual_targets, tns = [], [], []
    for loader in loaders:
        # Create list of all indexes
        indices = [list(range(li, li + loader.batch_size)) for li in range(0, len(loader.dataset), loader.batch_size)]
        # Iterate over all indexes
        for idxs in tqdm(indices, desc="Getting embeddings"):
            # Subset the indexes so we don't end up exceeding the number of clips
            idxs = [i for i in idxs if i < len(loader.dataset)]
            rolls, targets, paths = [], [], []
            # Now we actually use the __getitem__ call
            for n, i in enumerate(idxs):
                temp = loader.dataset.__getitem__(i)
                # replaces collate_fn to remove bad items from batch
                if temp is not None:
                    r, t, p = temp
                    rolls.append(r)
                    targets.append(t)
                    tn, _, _, clip_idx = loader.dataset.clips[i]
                    assert tn.split(os.path.sep)[-1] == p
                    paths.append(os.path.sep.join([tn.split(os.path.sep)[-2], p, f'clip_{str(clip_idx).zfill(3)}.mid']))
            assert len(rolls) == len(targets) == len(paths)
            # Convert into tensors
            roll = torch.tensor(np.array(rolls))
            target = torch.tensor(np.array(targets))
            # Get the required concept from the piano roll (harmony by default)
            required_roll = roll[:, roll_idx, :, :].unsqueeze(1)
            # Call the forward function without gradient computation
            with torch.no_grad():
                harm_embed = model(required_roll.to(utils.DEVICE)).squeeze(1)
            # Store original data
            tns.extend(paths)
            # Store layer activations
            actual_embeds.append(harm_embed.cpu().numpy())
            # Store target indexes
            actual_targets.append(target.cpu().numpy())
    return np.concatenate(actual_embeds), np.concatenate(actual_targets), np.array(tns)


def cav_similarity_peturb(
        clip_features: np.ndarray,
        clip_targets: np.ndarray,
        cav_features: np.ndarray,
        classifier_head: torch.nn.Module,
        epsilon: float = 1e-4
):
    all_s = []
    # Iterate through each individual clip embeddings and target
    for feat, targ in zip(clip_features, clip_targets):
        # Set devices correctly
        feat = torch.tensor(feat).to(utils.DEVICE)
        targ = torch.tensor(targ).to(utils.DEVICE)
        # Iterate through each individual CAV
        clip_res = []
        for cav in cav_features:
            # Perturb the embeddings towards and away from the CAV
            away = feat - (cav * epsilon)
            towards = feat + (cav * epsilon)
            # Compute the logits for both cases
            with torch.no_grad():
                away_logits = classifier_head(away)
                towards_logits = classifier_head(towards)
            # Apply the softmax
            away_sm = torch.nn.functional.softmax(away_logits, dim=-1)
            toward_sm = torch.nn.functional.softmax(towards_logits, dim=-1)
            # Get the logits for the target class
            away_desired = away_sm[targ]
            toward_desired = toward_sm[targ]
            # Compute the sensitivity by subtracting the results
            s = toward_desired - away_desired
            clip_res.append(s)
        # Shape = (n_cavs)
        all_s.append(np.array(clip_res))
    # Shape = (n_clips, n_cavs)
    return np.array(all_s)


def clip_cav_similarity(clip_features: np.ndarray, cav_features: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Creats a similarity matrix of shape (n_clips, n_CAVs) from arrays of clip and CAV embeddings"""
    if normalize:
        clip_features = torch.nn.functional.normalize(torch.tensor(clip_features), p=2, dim=-1).cpu().numpy()
        cav_features = torch.nn.functional.normalize(torch.tensor(cav_features), p=2, dim=-1).cpu().numpy()
    return clip_features @ cav_features.T


def performer_cav_correlation(
        cav_similarities: np.array,
        target_idxs: np.array,
        performer_mapping: dict
) -> pd.DataFrame:
    """From a CAV similarity matrix, take the correlation between binary class labels and CAV sensitivity"""
    res = []
    # Iterate over all CAVs
    for cav, cav_name in zip(cav_similarities.T, CAV_MAPPING):
        # cav = coeff_scale.fit_transform(cav.reshape(-1, 1))[:, 0]
        # Iterate over all musicians
        for musician_num in np.unique(target_idxs):
            musician_name = performer_mapping[musician_num]
            # Create binary array of targets
            temp = np.where(target_idxs == musician_num, 1., 0.)
            # Get correlation coefficient
            stack = np.vstack([cav, temp]).T
            corr = np.corrcoef(stack, rowvar=False)[0, 1]
            # Append results to dictionary
            res.append({"musician": musician_name, "cav": cav_name, "corr": corr})
    # Create dataframe and pivot
    return pd.DataFrame(res).pivot_table('corr', ['musician'], 'cav', sort=False)


def generate_topk_json(k: int, tracks: np.array, sims: np.array, cav_type: str = "Voicings") -> dict:
    """Generates and saves a JSON dictionary of the top-K most sensitive tracks to each CAV"""
    # Create the similarity dict: filenames are read in from our HTML file here
    similarity_dict = {}
    for num, cav_sim in enumerate(sims.T):
        topk_idxs = np.argsort(cav_sim)[::-1][:k]
        topk_tracks = tracks[topk_idxs]
        similarity_dict[str(num).zfill(3)] = topk_tracks.tolist()
    # Dump the JSON
    fp = os.path.join(
        utils.get_project_root(),
        "reports/figures/cav_plots",
        cav_type.lower(),
        "similarity.json"
    )
    with open(fp, 'w') as f:
        json.dump(similarity_dict, f)
    return similarity_dict


def main(
        model: str,
        use_all_data: bool,
        normalize: bool,
        topk: int,
        heatmap_size: list[int],
        kernel_size: list[int]
):
    """Creates all CAVs, generates similarity matrix, and saves plots"""
    logger.info(f"Initialising model with config file {model}")
    # Get the config file for training the model
    cfg_loc = os.path.join(utils.get_project_root(), 'config', model + '.yaml')
    if not os.path.isfile(cfg_loc):
        raise FileNotFoundError(f"Model {cfg_loc} does not have a config file associated with it!")
    # Get the training module
    tm = get_training_module(cfg_loc)
    # Get the harmony concept encoder and set to the required device/mode
    harmony_concept = tm.model.harmony_concept.to(utils.DEVICE)
    harmony_concept.eval()
    # Extract all CAVs
    logger.info("Extracting harmony CAVs...")
    cavs = get_all_cavs(harmony_concept)
    logger.info(f"... created CAVs with shape {cavs.shape}")
    # Get the required dataloaders
    logger.info("Embedding validation clips...")
    if use_all_data:
        logger.info("... using train, test, and validation data")
        alldata = [tm.validation_loader, tm.test_loader, tm.train_loader]
    else:
        logger.info("... using test and validation data only")
        alldata = [tm.validation_loader, tm.test_loader]
    # Embed the dataloaders
    features, targets, track_names = embed_all_clips(alldata, harmony_concept)
    logger.info(f"... created embeddings with shape {features.shape}")
    # Get similarities between clip and CAV features
    logger.info(f"... creating CAV similarity matrix with normalize = {normalize}")
    similarities = clip_cav_similarity(features, cavs, normalize=normalize)
    logger.info(f"... CAV similarity matrix has shape {similarities.shape}")
    # Outputs
    logger.info(f"Creating outputs...")
    logger.info(f"... using top-K = {topk}")
    # Generate the topk dictionary for all concepts
    topk_dict = generate_topk_json(topk, track_names, similarities, cav_type="Voicings")
    logger.info('... top-K JSON done!')
    # Get binary correlation between all performers and all CAVs
    corr_df = performer_cav_correlation(similarities, targets, tm.class_mapping)
    # Create the performer heatmap
    hm = HeatmapPianistCAV(corr_df, cav_type="Voicings")
    hm.create_plot()
    hm.save_fig()
    logger.info('... correlation heatmap done!')
    # Create heatmaps for top-K clips for all CAVs
    logger.info(f'... CAV heatmaps will be size {heatmap_size} using kernel {kernel_size}')
    for cav_idx, (cav_name, topk_tracks, cav) in enumerate(zip(CAV_MAPPING, list(topk_dict.values()), cavs)):
        for track in topk_tracks:
            # Create the non-interactive (matplotlib) heatmap
            hmks = HeatmapCAVKernelSensitivity(
                track,
                cav,
                encoder=harmony_concept,
                cav_name=cav_name,
                extractor_cls=HarmonyExtractor,
                cav_type="Voicings"
            )
            hmks.create_plot()
            hmks.save_fig()
            logger.info(f'... kernel sensitivity plot done for track {track}, CAV {cav_name}')
            # Create the interactive (plotly) heatmap
            hmks = HeatmapCAVKernelSensitivityInteractive(
                track,
                cav,
                encoder=harmony_concept,
                cav_name=cav_name,
                extractor_cls=HarmonyExtractor,
                cav_type="Voicings",
                cav_idx=cav_idx
            )
            hmks.create_plot()
            hmks.save_fig()
            logger.info(f'... interactive kernel sensitivity plot done for track {track}, CAV {cav_name}')


def parse_me(argparser: ArgumentParser) -> dict:
    """Adds required CLI arguments to an `ArgumentParser` object and parses them to a dictionary"""
    argparser.add_argument(
        '-m', '--model',
        default=DEFAULT_MODEL,
        type=str,
        help="Name of a trained model with saved checkpoints"
    )
    argparser.add_argument(
        "-d", "--use-all-data",
        default=True,
        type=bool,
        help="If true, will create CAVs using model training + test + validation data. "
             "If false, only uses test + validation data"
    )
    argparser.add_argument(
        "-n", "--normalize",
        default=False,
        type=bool,
        help="If true, will compute CAV similarity using cosine similarity. If false, will use dot-product similarity."
    )
    argparser.add_argument(
        '-t', '--topk',
        type=int,
        default=TOPK,
        help="Value of K to use when creating top-K heatmap plots"
    )
    argparser.add_argument(
        '--heatmap-size',
        nargs='+',
        type=int,
        default=[65, 26],
        help="Size of heatmap created for CAV sensitivity plots"
    )
    argparser.add_argument(
        '--kernel-size',
        nargs='+',
        type=int,
        default=[24, 5],
        help="Size of kernel used for CAV sensitivity plots, shape (semitones, seconds)"
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
        use_all_data=args["use_all_data"],
        normalize=args["normalize"],
        topk=args["topk"],
        heatmap_size=args["heatmap_size"],
        kernel_size=args["kernel_size"],
    )
