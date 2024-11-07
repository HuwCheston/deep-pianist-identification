#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates masked piano roll plots for held-out data using trained resnet and LIME"""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from lime import lime_image
from loguru import logger
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries
from skimage.util import invert

from deep_pianist_identification import utils, plotting
from deep_pianist_identification.encoders import ResNet50
from deep_pianist_identification.training import TrainModule, DEFAULT_CONFIG

MODEL_NAME = "baselines/resnet50-jtd+pijama-augment"

NUM_SAMPLES = 1000
NUM_FEATURES = 10


class HeatmapLIMEPianoRoll(plotting.BasePlot):
    EXPLAINED = lime_image.LimeImageExplainer(
        random_state=utils.SEED,
    )
    EXPLAIN_KWS = dict(
        top_labels=5,
        hide_color=0,
        num_samples=NUM_SAMPLES,
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
            classifier,
            pianist_name: str,
            n_features: int = 5
    ):
        super().__init__()
        # Don't modify the underlying piano roll
        self.piano_roll = deepcopy(piano_roll)
        self.classifier = classifier
        self.pianist_name = pianist_name
        self.n_features = n_features
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

    def get_explanation(self):
        explanation = self.EXPLAINED.explain_instance(
            self.piano_roll,
            self.classification_function,
            num_features=self.n_features,
            **self.EXPLAIN_KWS
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            **self.IMAGE_MASK_KWS
        )
        return invert(mark_boundaries(temp, mask))

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
            title=f'Pianist: {self.pianist_name}',
            xticks=list(range(0, utils.CLIP_LENGTH * utils.FPS + 500, 500)),
            yticks=range(0, utils.PIANO_KEYS, utils.OCTAVE),
            xticklabels=list(range(0, utils.CLIP_LENGTH + 5, 5)),
            yticklabels=range(0, utils.PIANO_KEYS, utils.OCTAVE),
            xlabel='Time (seconds)',
            ylabel='MIDI note number'
        )
        self.ax.tick_params(axis='x', rotation=0)
        # self.ax.invert_yaxis()
        plt.setp(self.ax.spines.values(), linewidth=plotting.LINEWIDTH)
        self.ax.tick_params(axis='both', width=plotting.TICKWIDTH)
        plotting.fmt_heatmap_axis(self.ax)

    def _format_fig(self):
        self.fig.subplots_adjust(
            right=1.075, left=0.05, top=0.9, bottom=0.15
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
            os.path.join(dirpath, f'{outpath}.png'),
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
        hm = HeatmapLIMEPianoRoll(roll[0], model, target_pianist, n_features=NUM_FEATURES)
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
    main()
