#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataloader modules for MIDI from textbooks"""

import os
from itertools import groupby
from random import shuffle

import numpy as np
from joblib import Parallel, delayed
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import get_piano_roll

__all__ = ["VoicingLoaderReal", "VoicingLoaderFake"]


class VoicingLoaderReal(Dataset):
    """Loads voicing MIDIs from unsupervised_resources/voicings/midi_final"""
    SPLIT_PATH = os.path.join(utils.get_project_root(), 'references/_unsupervised_resources/voicings/midi_final')

    def __init__(
            self,
            cav_number: int,
            n_clips: int = None,
            transpose_range: int = 6
    ):
        super().__init__()
        self.transpose_range = transpose_range
        self.n_clips = n_clips
        self.cav_midis = self.get_midi_paths(cav_number)
        self.midis = self.create_midi()

    def get_midi_paths(self, cav_number):
        cav_path = os.path.join(self.SPLIT_PATH, f"{cav_number}_cav")
        assert os.path.isdir(cav_path)
        cavs = [os.path.join(cav_path, i) for i in os.listdir(cav_path) if i.endswith('mid')]
        return cavs

    def _create_midi(self, cav_midi_path):
        pm = PrettyMIDI(cav_midi_path)
        # Replacing velocity with 1.0
        lhand = [(i.start, i.end, i.pitch, 1.0) for i in pm.instruments[1].notes]
        rhand = [(i.start, i.end, i.pitch, 1.0) for i in pm.instruments[0].notes]
        # Take top note from each left hand if we have multiple voices
        newlhand = []
        for key, group in groupby(lhand, lambda x: x[0]):
            group = list(group)
            if len(group) > 1:
                # Get the highest pitch
                sort = sorted(group, key=lambda x: x[2])
                rhand.extend(sort[1:])
                newlhand.append(sort[0])
        lhand = newlhand

        rhand = sorted(rhand, key=lambda x: (x[0], x[2]))
        rhand_grp = [list(i[1]) for i in groupby(rhand, lambda x: x[0])]

        inverted_rhands = {i: [] for i in range(len(rhand_grp[0]) - 1)}
        for inversion in inverted_rhands.keys():
            for grp in rhand_grp:
                to_invert = sorted(grp, key=lambda x: x[2])[:inversion + 1]
                inverted = [(i[0], i[1], i[2] + 12, i[3]) for i in to_invert]
                combined = inverted + grp[inversion + 1:]
                inverted_rhands[inversion].extend(sorted(combined, key=lambda x: x[2]))

        to_compute = [rhand, *list(inverted_rhands.values())]
        # Iterating over all inversions
        all_rolls = []
        for rhands in to_compute:
            for transp in range(-self.transpose_range, self.transpose_range + 1):
                for include_lh in [True, False]:
                    if include_lh:
                        comb = sorted(rhands + lhand, key=lambda x: x[0])
                    else:
                        comb = sorted(rhands, key=lambda x: x[0])
                    onset_grps = [list(i[1]) for i in groupby(comb, lambda x: x[0])]
                    spacer = np.linspace(0, utils.CLIP_LENGTH, len(onset_grps) + 1)
                    fmt = []
                    for start, end, grp in zip(spacer, spacer[1:], onset_grps):
                        fmt_grp = [(start, end, note[2] + transp, note[3]) for note in grp]
                        fmt.extend(fmt_grp)
                    all_rolls.append(get_piano_roll(fmt))
        return all_rolls

    def create_midi(self):
        with Parallel(n_jobs=-1, verbose=1) as par:
            midis = par(delayed(self._create_midi)(mid) for mid in self.cav_midis)
            midis = [m for ms in midis for m in ms]
        # midis = [list(self._create_midi(midi)) for midi in self.cav_midis]
        if self.n_clips is not None:
            shuffle(midis)
            midis = midis[:self.n_clips]
        return midis

    def __len__(self):
        return len(self.midis)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        # Return the MIDI piano roll and the class index (i.e., True)
        return np.expand_dims(self.midis[idx], 0), 1


class VoicingLoaderFake(VoicingLoaderReal):
    def __init__(
            self,
            avoid_cav_number: int,
            n_clips: int,
            transpose_range: int = 6
    ):
        super().__init__(cav_number=avoid_cav_number, n_clips=n_clips, transpose_range=transpose_range)

    def get_midi_paths(self, cav_number):
        fmt_cav_paths = [
            os.path.join(self.SPLIT_PATH, i) for i in os.listdir(self.SPLIT_PATH) if i != f"{cav_number}_cav"
        ]
        all_cavs = []
        for path in fmt_cav_paths:
            all_cavs.extend([os.path.join(self.SPLIT_PATH, path, i) for i in os.listdir(path) if i.endswith('.mid')])
        shuffle(all_cavs)
        return all_cavs

    def __len__(self):
        return self.n_clips

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        # Return the MIDI piano roll and the class index (i.e., False)
        return np.expand_dims(self.midis[idx], 0), 0
