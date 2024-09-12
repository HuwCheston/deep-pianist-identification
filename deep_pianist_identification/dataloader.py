#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIDI DataLoader module"""

import os
import warnings

import numpy as np
import pandas as pd
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset

from deep_pianist_identification.utils import get_project_root

__all__ = ["TrackLoader"]

PIANO_KEYS = 88
MIDI_OFFSET = 21
CLIP_LENGTH = 30
FPS = 100
N_FRAMES_PER_CLIP = int((CLIP_LENGTH * 1000) / FPS)


class ClipLoader:
    def __init__(self, track_path: str, normalize_velocity: bool = True):
        self.midi_fpath = os.path.join(get_project_root(), 'data', track_path)
        self.requires_norm = normalize_velocity

    def get_track_clip(self, clip_idx) -> np.ndarray:
        track_array = np.zeros((1, PIANO_KEYS, N_FRAMES_PER_CLIP), dtype=np.float32)
        pm = PrettyMIDI(os.path.join(self.midi_fpath, 'piano_midi.mid')).instruments[0]
        clip_start = CLIP_LENGTH * clip_idx
        clip_end = clip_start + CLIP_LENGTH
        clip_notes = [n for n in pm.notes if all(
            (n.start >= clip_start, n.end < clip_end, n.end - n.start > 1 / FPS)
        )]
        frame_iter = np.linspace(clip_start, clip_end, N_FRAMES_PER_CLIP + 1)
        for frame_idx, (frame_start, frame_end) in enumerate(zip(frame_iter, frame_iter[1:])):
            frame_notes = [n for n in clip_notes if n.start < frame_end and n.end >= frame_start]
            for note in frame_notes:
                track_array[0, note.pitch - MIDI_OFFSET, frame_idx] = note.velocity
        return track_array

    def load_clip_from_cache(self, clip_idx: int) -> np.ndarray:
        clip_folder = os.path.join(self.midi_fpath, 'clips')
        clip_path = os.path.join(clip_folder, f'clip_{clip_idx}.npy')
        try:
            with open(clip_path, 'rb') as inpath:
                clip = np.load(inpath)
        except FileNotFoundError:
            if not os.path.exists(clip_folder):
                os.mkdir(clip_folder)
            clip = self.get_track_clip(clip_idx)
            with open(clip_path, 'wb') as outpath:
                np.save(outpath, clip)
        return clip

    @staticmethod
    def normalize_velocity(track_array: np.ndarray) -> np.ndarray:
        return (track_array - np.min(track_array)) / (np.max(track_array) - np.min(track_array))

    def __getitem__(self, idx: int) -> np.ndarray:
        clip = self.load_clip_from_cache(idx)
        if self.requires_norm:
            clip = self.normalize_velocity(clip)
        return clip


class TrackLoader(Dataset):
    def __init__(self, split: str, n_clips: int = None, normalize_velocity: bool = True):
        super().__init__()
        csv_path = os.path.join(get_project_root(), 'references/data_splits', f'{split}_split.csv')
        split_df = pd.read_csv(csv_path, delimiter=',', index_col=0)
        self.clips = []
        self.normalize_velocity = normalize_velocity
        for idx, track in split_df.iterrows():
            track_clips = int(track['duration'] // CLIP_LENGTH)
            for clip_idx in range(track_clips):
                self.clips.append((track['track'], track['pianist'], clip_idx))
        self.clips = self.clips[:n_clips]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        track_path, target_class, clip_idx = self.clips[idx]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                cl = ClipLoader(track_path)
                loaded_clip = cl.__getitem__(clip_idx)
            except Warning as e:
                print('error found:', e)
                print(track_path, target_class, clip_idx)
                raise
        return loaded_clip, target_class
