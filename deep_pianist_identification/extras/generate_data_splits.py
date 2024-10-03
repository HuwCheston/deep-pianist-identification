#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates data splits and stores as .csv files in `./references/data_splits`"""

import json
import os

import pandas as pd
from pretty_midi import PrettyMIDI
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deep_pianist_identification.utils import (
    get_project_root, SEED, PIANIST_MAPPING, seed_everything
)

TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE = 0.8, 0.1, 0.1


def get_tracks(databases: tuple = ('pijama', 'jtd')):
    for database in databases:
        database_path = os.path.join(get_project_root(), 'data/raw', database)
        for track in tqdm(os.listdir(database_path), desc=f'Loading tracks from {database} ...'):
            track_path = os.path.join(database_path, track)
            metadata_path = os.path.join(track_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                continue
            metadata = json.load(open(metadata_path, 'r'))
            pianist = metadata['pianist']
            if pianist in PIANIST_MAPPING.keys():
                track_duration, track_notes = get_track_duration_and_notes(track_path)
                # Is the track long enough for us to create at least one clip from it?
                # Do we have at least one MIDI note in the track?
                # # TODO: do we need to add padding here?
                # if track_duration >= CLIP_PADDING + CLIP_LENGTH and track_notes > 0:
                yield database, PIANIST_MAPPING[pianist], pianist, track, track_duration


def get_track_duration_and_notes(track_name: str) -> tuple[float, int]:
    """Returns the duration of a MIDI file and the number of notes that it contains"""
    pm = PrettyMIDI(os.path.join(track_name, 'piano_midi.mid')).instruments[0]
    return pm.get_end_time(), len(pm.notes)


def generate_split(track_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def fmt(tracks: pd.Series, pianist: pd.Series) -> pd.DataFrame:
        return pd.concat([
            tracks,
            pianist,
            track_df.iloc[tracks.index]['database'],
            track_df.iloc[tracks.index]['duration']
        ], axis=1
        )

    # Create train and temp test split
    train_tracks, temp_tracks, train_pianist, temp_pianist = train_test_split(
        track_df['track'],
        track_df['pianist'],
        test_size=TEST_SIZE + VALIDATION_SIZE,
        train_size=TRAIN_SIZE,
        stratify=track_df['database'],
        random_state=SEED,
        shuffle=True
    )
    # Format dataframes
    train_df = fmt(train_tracks, train_pianist)
    temp_df = fmt(temp_tracks, temp_pianist)
    # Test split into test and validation split
    test_tracks, valid_tracks, test_pianist, valid_pianist = train_test_split(
        temp_df['track'],
        temp_df['pianist'],
        stratify=temp_df['database'],
        test_size=0.5,  # Both should be the same size
        train_size=0.5,
        random_state=SEED,
        shuffle=True
    )
    # Format dataframes
    test_df = fmt(test_tracks, test_pianist)
    valid_df = fmt(valid_tracks, valid_pianist)
    # Quick checks to ensure that no tracks are in the same split
    assert not any(i in test_df['track'].values for i in train_df['track'].values)
    assert not any(i in valid_df['track'].values for i in train_df['track'].values)
    assert not any(i in valid_df['track'].values for i in test_df['track'].values)
    return train_df, test_df, valid_df


if __name__ == "__main__":
    seed_everything(SEED)

    # Get the valid tracks from all databases
    loaded_tracks = get_tracks()
    df = pd.DataFrame(loaded_tracks, columns=['database', 'pianist', 'name', 'track', 'duration'])
    # Split into train, test, and validation dataframes
    df_train, df_test, df_valid = generate_split(df)
    # Save the dataframes as individual CSV files
    split_path = os.path.join(get_project_root(), 'references/data_splits')
    for df, split in zip([df_train, df_test, df_valid], ['train', 'test', 'validation']):
        df['track'] = df['database'] + "/" + df['track']
        df.drop(columns=['database']).reset_index(drop=True).to_csv(os.path.join(split_path, f"{split}_split.csv"))
