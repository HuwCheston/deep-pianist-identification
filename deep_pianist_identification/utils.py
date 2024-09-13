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

CLIP_LENGTH = 30
N_CLASSES = 25


def seed_everything(seed: int = 42) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timer(name: str) -> ContextManager[None]:
    """Print out how long it takes to execute the provided block."""
    try:
        start = time()
        yield
    except Exception as e:
        end = time()
        raise e
    else:
        end = time()
        print(f"Took {end - start:.2f} seconds to {name}.")


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
        database_path = os.path.join(get_project_root(), 'data', database)
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
