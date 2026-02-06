#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for stratified data splits"""

import os
import math

import pandas as pd
import pytest

from deep_pianist_identification import utils


MAGIC_NUMBER = 1629
DESIRED_PIANISTS = [
    "Abdullah Ibrahim",
    "Ahmad Jamal",
    "Bill Evans",
    "Brad Mehldau",
    "Cedar Walton",
    "Chick Corea",
    "Gene Harris",
    "Geri Allen",
    "Hank Jones",
    "John Hicks",
    "Junior Mance",
    "Keith Jarrett",
    "Kenny Barron",
    "Kenny Drew",
    "McCoy Tyner",
    "Oscar Peterson",
    "Stanley Cowell",
    "Teddy Wilson",
    "Thelonious Monk",
    "Tommy Flanagan",
]


def load_csv(stratify_type: str) -> pd.DataFrame:
    loaded = []
    for split_name in ["test", "train", "validation"]:
        split_path = os.path.join(
            utils.get_project_root(),
            "references/data_splits",
            f"20class_80min_stratified_{stratify_type}",
            f"{split_name}_split.csv"
        )
        df = pd.read_csv(split_path, index_col=0)
        df["split"] = split_name
        loaded.append(df)
    return pd.concat(loaded, axis=0).reset_index(drop=True)


@pytest.mark.parametrize("stratify_type", ["album", "composition"])
def test_pianist_in_each_split(stratify_type):
    df_ = load_csv(stratify_type)
    # Each pianist should have one recording in every split
    for pianist in DESIRED_PIANISTS:
        for split in ["train", "validation", "test"]:
            got = df_[(df_["pianist"] == pianist) & (df_["split"] == split)]
            assert len(got) >= 1, f"Pianist {pianist} does not appear in split {split}!"


@pytest.mark.parametrize("stratify_type", ["album", "composition"])
def test_split_proportion(stratify_type):
    # Splits should be approximately 8/1/1
    df_ = load_csv(stratify_type)
    total_tracks = len(df_)
    for split, desired_proportion in zip(["train", "validation", "test"], [0.8, 0.1, 0.1]):
        actual_proportion = len(df_[df_["split"] == split]) / total_tracks
        assert math.isclose(actual_proportion, desired_proportion, abs_tol=0.01)


@pytest.mark.parametrize("stratify_type", ["album", "composition"])
def test_no_leaks(stratify_type):
    # No leaks of compositions/albums between splits
    df_ = load_csv(stratify_type)
    for split_a in ["train", "validation", "test"]:
        split_a_df = df_[df_["split"] == split_a]
        split_a_comps = set(split_a_df[stratify_type].unique().tolist())
        other_splits = [i for i in ["train", "validation", "test"] if i != split_a]
        for split_b in other_splits:
            split_b_df = df_[df_["split"] == split_b]
            split_b_comps = set(split_b_df[stratify_type].unique().tolist())
            shared = set.intersection(split_a_comps, split_b_comps)
            assert len(shared) == 0


@pytest.mark.parametrize("stratify_type", ["album", "composition"])
def test_no_overlap_tracks(stratify_type):
    # No leak of tracks between splits
    df_ = load_csv(stratify_type)
    for split_a in ["train", "val", "test"]:
        split_a_df = df_[df_["split"] == split_a]
        split_a_tracks = set(split_a_df["track"].unique().tolist())
        other_splits = [i for i in ["train", "validation", "test"] if i != split_a]
        for split_b in other_splits:
            split_b_df = df_[df_["split"] == split_b]
            split_b_tracks = set(split_b_df["track"].unique().tolist())
            shared = set.intersection(split_a_tracks, split_b_tracks)
            assert len(shared) == 0


@pytest.mark.parametrize("stratify_type", ["album", "composition"])
def test_dataset_size(stratify_type):
    df_ = load_csv(stratify_type)
    assert len(df_) == MAGIC_NUMBER
