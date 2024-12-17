#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates masked piano roll plots for held-out data using trained resnet and LIME"""

import json
import os
from copy import deepcopy
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from lime import lime_image
from loguru import logger
from pretty_midi import note_number_to_name
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries
from skimage.util import invert

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.encoders import ResNet50
from deep_pianist_identification.training import TrainModule, DEFAULT_CONFIG

MODEL_NAME = "baselines/resnet50-jtd+pijama-augment"

N_SAMPLES = 1000
N_FEATURES = 5


class HeatmapLIMEPianoRoll(plotting.BasePlot):
    EXPLAINED = lime_image.LimeImageExplainer(
        random_state=utils.SEED,
    )
    EXPLAIN_KWS = dict(
        top_labels=5,
        hide_color=0,
        random_seed=utils.SEED,
        distance_metric='cosine'
    )
    IMAGE_MASK_KWS = dict(
        positive_only=True,
        hide_rest=True,
    )

    def __init__(
            self,
            piano_roll: np.array,
            clip_path: str,
            classifier: torch.nn.Module,
            pianist_name: str,
            n_features: int = N_FEATURES,
            n_samples: int = N_SAMPLES
    ):
        super().__init__()
        self.clip_path = clip_path
        # Matplotlib figure and axis
        self.clip_start, self.clip_end = self.parse_clip_start_stop(clip_path)
        # Don't modify the underlying piano roll
        self.piano_roll = deepcopy(piano_roll)
        self.classifier = classifier
        self.pianist_name = pianist_name
        self.n_features = n_features
        self.n_samples = n_samples
        self.fig, self.ax = plt.subplots(
            1, 1, figsize=(plotting.WIDTH, plotting.WIDTH // 3)
        )
        self.masked_image = None

    def classification_function(self, tmp):
        # Convert the input to binary
        roller = deepcopy(tmp)
        if isinstance(roller, np.ndarray):
            roller = torch.tensor(roller)
        # LIME annoyingly converts to RGB and changes dimensions
        #  so convert back to grayscale and permute
        grayscaled = rgb2gray(roller)
        fmt = torch.tensor(grayscaled).unsqueeze(-1).permute(0, 3, 1, 2)
        # Through the model
        with torch.no_grad():
            img = fmt.to(utils.DEVICE)
            logits = self.classifier(img)
        # Softmax raw logits and return
        softmaxed = torch.nn.functional.softmax(logits, dim=1)
        return softmaxed.cpu().numpy()

    @staticmethod
    def seconds_to_timedelta(seconds: int):
        # Format in %M:%S format
        td = timedelta(seconds=seconds)
        return str(td)[2:]

    @staticmethod
    def parse_clip_start_stop(clip_path: str):
        """Parse clip start and stop time from clip number using filepath"""
        clip_num = int(clip_path.split('_')[-1])
        clip_start = utils.CLIP_LENGTH * clip_num
        clip_end = clip_start + utils.CLIP_LENGTH
        return clip_start, clip_end

    def parse_human_readable_trackname(self):
        clip_start_fmt = self.seconds_to_timedelta(self.clip_start)
        clip_end_fmt = self.seconds_to_timedelta(self.clip_end)
        # TODO: make this nicer
        for dataset in ['jtd', 'pijama']:
            temp_path = f'{dataset}/{self.clip_path.split("_")[0].split("/")[1]}'
            json_path = os.path.join(utils.get_project_root(), 'data/raw', temp_path, 'metadata.json')
            try:
                loaded = json.load(open(json_path, 'r'))
            except FileNotFoundError:
                continue
            else:
                return (f'{loaded["bandleader"]} — "{loaded["track_name"]}" ({clip_start_fmt}—{clip_end_fmt}) '
                        f'\nfrom "{loaded["album_name"]}", {loaded["recording_year"]}')

    def get_explanation(self):
        explanation = self.EXPLAINED.explain_instance(
            self.piano_roll,
            self.classification_function,
            num_features=self.n_features,
            num_samples=self.n_samples,
            **self.EXPLAIN_KWS
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            **self.IMAGE_MASK_KWS
        )
        return invert(mark_boundaries(temp / 255., mask))

    def create_plot(self):
        self.masked_image = self.get_explanation()
        self._create_plot()
        self._format_ax()
        self._format_fig()

    def _create_plot(self):
        self.piano_roll[self.piano_roll == 0] = np.nan
        self.ax.imshow(
            self.masked_image,
            aspect='auto',
        )
        sns.heatmap(
            self.piano_roll,
            ax=self.ax,
            alpha=1.0,
            cbar_kws={
                'label': 'Velocity',
                'pad': 0.01,
            },
            cmap="rocket",
            vmin=0.,
            vmax=1.,
            zorder=10
        )
        cbar = self.ax.collections[0].colorbar
        cbar.set_ticks([0.0, 1.0], labels=[0, 127])

    def _format_ax(self):
        self.ax.patch.set_edgecolor('black')
        self.ax.patch.set_linewidth(1)
        self.ax.grid(**plotting.GRID_KWS)
        # Set axes limits correctly
        self.ax.set(
            title=self.parse_human_readable_trackname(),
            xticks=range(0, (utils.CLIP_LENGTH * utils.FPS + 1), utils.FPS * 5),
            xticklabels=[self.seconds_to_timedelta(i) for i in range(self.clip_start, self.clip_end + 1, 5)],
            yticks=range(0, utils.PIANO_KEYS, utils.OCTAVE),
            yticklabels=[note_number_to_name(i + utils.MIDI_OFFSET) for i in range(0, utils.PIANO_KEYS, 12)],
            xlabel='Time (seconds)',
            ylabel='Note'
        )
        self.ax.tick_params(axis='x', rotation=0)
        # self.ax.invert_yaxis()
        plt.setp(self.ax.spines.values(), linewidth=plotting.LINEWIDTH)
        self.ax.tick_params(axis='both', width=plotting.TICKWIDTH)
        plotting.fmt_heatmap_axis(self.ax)

    def _format_fig(self):
        self.fig.subplots_adjust(
            right=1.075, left=0.05, top=0.875, bottom=0.15
        )

    def save_fig(self, outpath: str):
        dirpath = os.path.join(
            utils.get_project_root(),
            'reports/figures/blackbox/lime_piano_rolls',
            self.pianist_name.lower().replace(' ', '_')
        )
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        self.fig.savefig(
            os.path.join(dirpath, f'{outpath.split("/")[1]}.png'),
            **plotting.SAVE_KWS
        )
        plt.close('all')


def get_cfg():
    # Load in the config path for the model
    cfg_path = os.path.join(utils.get_project_root(), 'config', MODEL_NAME + '.yaml')
    cfg = yaml.safe_load(open(cfg_path))
    # Replace any non-existing keys with their default value
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg.keys():
            cfg[k] = v
    return cfg


def create_plots(dataset, class_mapping, model):
    for idx in range(len(dataset)):
        _, target_class, target_dataset, clip_idx = dataset.clips[idx]
        target_pianist = class_mapping[target_class]
        ret = dataset.__getitem__(idx)
        # Skip over cases where get an ExtractorError
        if ret is None:
            continue
        roll, _, clip_path = ret
        clip_path = f'{clip_path}_clip_{str(clip_idx).zfill(3)}'
        # Update clip path with target dataset if we don't have this
        # TODO: why?
        if "jtd" not in clip_path.lower() or "pijama" not in clip_path.lower():
            clip_path = f'{"jtd" if target_dataset == 1 else "pijama"}/{clip_path}'
        logger.info(f'Processing {clip_path}')
        hm = HeatmapLIMEPianoRoll(roll[0], clip_path, model, target_pianist)
        hm.create_plot()
        hm.save_fig(clip_path)


def main():
    logger.info(f'Using model {MODEL_NAME}')
    logger.info('Getting model config...')
    cfg = get_cfg()
    # Initialise the training module here as this will create the model and automatically load the checkpoint
    tm = TrainModule(**cfg)
    # Simply grab the checkpointed model from the training module
    model: ResNet50 = tm.model.to(utils.DEVICE).eval()
    # Create the dataloader using arguments from our training module
    logger.info('Creating dataloaders...')
    validation_dataloader = tm.dataloader_cls(
        'validation',
        classify_dataset=tm.classify_dataset,
        data_split_dir=tm.data_split_dir,
        # We can just use the arguments for the test dataset
        **tm.test_dataset_cfg
    )
    test_dataloader = tm.dataloader_cls(
        'test',
        classify_dataset=tm.classify_dataset,
        data_split_dir=tm.data_split_dir,
        # We can just use the arguments for the test dataset
        **tm.test_dataset_cfg
    )
    logger.info(f'Creating plots for validation dataset split...')
    create_plots(validation_dataloader, class_mapping=tm.class_mapping, model=model)
    logger.info(f'Creating plots for test dataset split...')
    create_plots(test_dataloader, class_mapping=tm.class_mapping, model=model)


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)
    main()
