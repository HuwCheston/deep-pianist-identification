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
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

PIANO_KEYS = 88
FPS = 100
MIDI_OFFSET = 21

CLIP_LENGTH = 30  # Each clip will start at CLIP_LENGTH + PADDING size, then will be cropped to CLIP_SIZE later
HOP_SIZE = 30
CLIP_PADDING = 30  # Added to the end of each clip to facilitate random cropping/time dilation


# Converts class indices into string names
CLASS_MAPPING = {
    0: 'Abdullah Ibrahim',
    1: 'Ahmad Jamal',
    2: 'Benny Green',
    3: 'Bill Evans',
    4: 'Bobby Timmons',
    5: 'Brad Mehldau',
    6: 'Cedar Walton',
    7: 'Chick Corea',
    8: 'Gene Harris',
    9: 'Geoffrey Keezer',
    10: 'Geri Allen',
    11: 'Hank Jones',
    12: 'John Hicks',
    13: 'Junior Mance',
    14: 'Keith Jarrett',
    15: 'Kenny Barron',
    16: 'Kenny Drew',
    17: 'Lennie Tristano',
    18: 'McCoy Tyner',
    19: 'Oscar Peterson',
    20: 'Red Garland',
    21: 'Stanley Cowell',
    22: 'Teddy Wilson',
    23: 'Thelonious Monk',
    24: 'Tommy Flanagan'
}
PIANIST_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}  # Inverted, maps string names to class indexes
N_CLASSES = 25


def seed_everything(seed: int = 42) -> None:
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
