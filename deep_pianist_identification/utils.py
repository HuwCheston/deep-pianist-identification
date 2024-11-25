#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used across the entire pipeline"""

import json
import os
import random
import string
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from time import time
from typing import ContextManager

import numpy as np
import pandas as pd
import torch
from pretty_midi import PrettyMIDI

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

PIANO_KEYS = 88
FPS = 100
MIDI_OFFSET = 21
MAX_VELOCITY = 127
MIDDLE_C = 60
OCTAVE = 12

CLIP_LENGTH = 30  # Each clip will start at CLIP_LENGTH + PADDING size, then will be cropped to CLIP_SIZE later
HOP_SIZE = 30
CLIP_PADDING = 30  # Added to the end of each clip to facilitate random cropping/time dilation


def get_class_mapping(dataset_name: str) -> dict:
    """With a given dataset, returns a dictionary of class_idx: class_name"""
    csv_loc = os.path.join(get_project_root(), 'references/data_splits', dataset_name, 'pianist_mapping.csv')
    return {i: row.iloc[0] for i, row in pd.read_csv(csv_loc, index_col=1).iterrows()}


def seed_everything(seed: int = SEED) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def total_parameters(layer) -> int:
    return sum(p.numel() for p in layer.parameters())


@contextmanager
def timer(name: str) -> ContextManager[None]:
    """Print out how long it takes to execute the provided block."""
    from loguru import logger
    start = time()
    try:
        yield
    except Exception as e:
        end = time()
        logger.warning(f"Took {end - start:.2f} seconds to {name} and raised {e}.")
        raise e
    else:
        end = time()
        logger.debug(f"Took {end - start:.2f} seconds to {name}.")


def get_project_root() -> Path:
    """Returns the root directory of the project"""
    return Path(__file__).absolute().parent.parent


def remove_punctuation(s: str) -> str:
    """Removes punctuation from a given input string `s`"""
    return ''.join(
        c for c in str(s).translate(str.maketrans('', '', string.punctuation)).replace('’', '')
        if c in string.printable
    )


@cache
def get_pianist_names(datasets: tuple = ('jtd', 'pijama')) -> set:
    """Get the names of the pianists who have at least one recording in all `datasets`"""
    res = {dat: [] for dat in datasets}
    # Iterate through each dataset
    for database in datasets:
        database_path = os.path.join(get_project_root(), 'data/raw', database)
        # Iterate through the folders corresponding to each track in the database
        for f in os.listdir(database_path):
            if not os.path.isdir(os.path.join(database_path, f)):
                continue
            # Open up the metadata file
            meta = os.path.join(database_path, f, 'metadata.json')
            reader = json.load(open(meta, 'r', encoding='utf-8'))
            # Get the name of the pianist
            res[database].append(reader['pianist'])
        # Remove all duplicates
        res[database] = set(res[database])
    # Getting the overlap
    return set.intersection(*res.values())


def get_track_metadata(pianist_names: set) -> pd.DataFrame:
    """Gets metadata (dataset name, recording duration) for all tracks by performers in `pianist_names"""
    res = []
    # Iterate over all datasets
    for dataset in ['jtd', 'pijama']:
        # Get the root path for this dataset
        root = os.path.join(get_project_root(), 'data/raw', dataset)
        # Iterate through all tracks in the dataset
        for track in os.listdir(root):
            # Get the paths for both the raw MIDI and metadata JSON
            mid = os.path.join(root, track, 'piano_midi.mid')
            meta = os.path.join(root, track, 'metadata.json')
            # If we don't have the files for whatever reason, skip to the next file
            if not os.path.isfile(meta) or not os.path.isfile(mid):
                continue
            # Load the metadata JSON and get the name of the pianist
            loaded = json.load(open(meta, 'r'))
            pianist = loaded['pianist']
            # If this pianist is in our list
            if pianist in pianist_names:
                # For pijama, get the duration from the midi file
                if dataset == 'pijama':
                    pm = PrettyMIDI(mid)
                    dur = pm.get_end_time()
                # Otherwise, get the duration from the metadata file (quicker)
                else:
                    minutes, seconds = loaded['excerpt_duration'].split(':')
                    dur = (int(minutes) * 60) + int(seconds)
                res.append({'pianist': pianist, 'dataset': dataset, 'duration': dur})
    # Create the dataframe and return
    return pd.DataFrame(res)
