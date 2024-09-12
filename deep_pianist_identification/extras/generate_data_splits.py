#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates data splits and stores as .csv files in `./references/data_splits`"""

import json
import os

import pandas as pd
from pretty_midi import PrettyMIDI
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deep_pianist_identification.utils import get_pianist_names, get_project_root, SEED

PIANIST_MAPPING = {name: idx for idx, name in enumerate(sorted(get_pianist_names()))}
TRAIN_SIZE, TEST_SIZE = 0.8, 0.2


def get_tracks(databases: tuple = ('pijama', 'jtd')):
    for database in databases:
        database_path = os.path.join(get_project_root(), 'data', database)
        for track in tqdm(os.listdir(database_path), desc=f'Loading tracks from {database} ...'):
            track_path = os.path.join(database_path, track)
            metadata_path = os.path.join(track_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                continue
            metadata = json.load(open(metadata_path, 'r'))
            pianist = metadata['pianist']
            if pianist in PIANIST_MAPPING.keys():
                track_duration = get_track_duration(track_path)
                yield database, PIANIST_MAPPING[pianist], track, track_duration


def get_track_duration(track_name: str) -> float:
    pm = PrettyMIDI(os.path.join(track_name, 'piano_midi.mid')).instruments[0]
    return max([i.end for i in pm.notes])


def generate_split(track_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_tracks, test_tracks, train_pianist, test_pianist = train_test_split(
        track_df['track'],
        track_df['pianist'],
        test_size=TEST_SIZE,
        train_size=TRAIN_SIZE,
        stratify=track_df['database'],
        random_state=SEED,
        shuffle=True
    )
    return (
        pd.concat([
            train_tracks,
            train_pianist,
            track_df.iloc[train_tracks.index]['database'],
            track_df.iloc[train_tracks.index]['duration'],
        ], axis=1),
        pd.concat([
            test_tracks,
            test_pianist,
            track_df.iloc[test_tracks.index]['database'],
            track_df.iloc[test_tracks.index]['duration']
        ], axis=1)
    )


if __name__ == "__main__":
    loaded_tracks = get_tracks()
    df = pd.DataFrame(loaded_tracks, columns=['database', 'pianist', 'track', 'duration'])
    train_df, test_df = generate_split(df)
    split_path = os.path.join(get_project_root(), 'references/data_splits')
    for df, split in zip([train_df, test_df], ['train', 'test']):
        df['track'] = df['database'] + "/" + df['track']
        df.drop(columns=['database']).reset_index(drop=True).to_csv(os.path.join(split_path, f"{split}_split.csv"))
