#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates data splits and stores as .csv files in `./references/data_splits`"""

import json
import os

import pandas as pd
from loguru import logger
from pretty_midi import PrettyMIDI
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deep_pianist_identification.utils import get_project_root, SEED, seed_everything

# These pianists have at least one track in both datasets
IN_JTD_AND_PIJAMA = [
    'Abdullah Ibrahim',
    'Ahmad Jamal',
    'Benny Green',
    'Bill Evans',
    'Bobby Timmons',
    'Brad Mehldau',
    'Cedar Walton',
    'Chick Corea',
    'Gene Harris',
    'Geoffrey Keezer',
    'Geri Allen',
    'Hank Jones',
    'John Hicks',
    'Junior Mance',
    'Keith Jarrett',
    'Kenny Barron',
    'Kenny Drew',
    'Lennie Tristano',
    'McCoy Tyner',
    'Oscar Peterson',
    'Red Garland',
    'Stanley Cowell',
    'Teddy Wilson',
    'Thelonious Monk',
    'Tommy Flanagan'
]
MIN_PIANIST_MINUTES = 80.0  # A pianist must have at least this many minutes of audio to be included
TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE = 0.8, 0.1, 0.1


def get_tracks(databases: tuple = ('pijama', 'jtd')):
    """From the provided `databases`, gets tracks with valid pianists"""
    # Iterate over databases
    for database in databases:
        database_path = os.path.join(get_project_root(), 'data/raw', database)
        # Iterate over all tracks in the database, with a counter
        for track in tqdm(os.listdir(database_path), desc=f'Loading tracks from {database} ...'):
            # Get the track paths
            track_path = os.path.join(database_path, track)
            metadata_path = os.path.join(track_path, 'metadata.json')
            # If the track doesn't exist for whatever reason (i.e., it's a `.gitkeep` file), skip over
            if not os.path.exists(metadata_path):
                continue
            # Load in the metadata and get the name of the pianist
            metadata = json.load(open(metadata_path, 'r'))
            pianist = metadata['pianist']
            # If the pianist has both JTD and Pijama tracks
            if pianist in IN_JTD_AND_PIJAMA:
                # Get the duration
                track_duration = get_track_duration(track_path)
                # Return everything for our dataframe
                yield database, pianist, track, track_duration


def get_track_duration(track_name: str) -> float:
    """Returns the duration of a MIDI file and the number of notes that it contains"""
    pm = PrettyMIDI(os.path.join(track_name, 'piano_midi.mid')).instruments[0]
    return pm.get_end_time()


def remove_pianists_without_enough_audio(
        pianist_df: pd.DataFrame, min_minutes: float = MIN_PIANIST_MINUTES
) -> pd.DataFrame:
    """Drops pianists from a dataframe who do not have at least `min_minutes` worth of audio"""
    # Group by name, sum the duration, convert to minutes, and sort
    grp = pianist_df.groupby('pianist')['duration'].sum().apply(lambda x: x / 60).sort_values()
    # Remove pianists without enough audio
    pianists = grp[grp > min_minutes].index.tolist()
    return pianist_df[pianist_df['pianist'].isin(pianists)].reset_index(drop=True)


def generate_split(
        track_df: pd.DataFrame,
        train_size: float = TRAIN_SIZE,
        test_size: float = TEST_SIZE,
        validation_size: float = VALIDATION_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates splits from `track_df` with required `size`s"""

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
        test_size=test_size + validation_size,
        train_size=train_size,
        stratify=track_df['database'],
        random_state=SEED,
        shuffle=True
    )
    # Format dataframes
    train_df = fmt(train_tracks, train_pianist)
    temp_df = fmt(temp_tracks, temp_pianist)
    # Split temp dataframe into test and validation split
    v_size = validation_size / (test_size + validation_size)
    t_size = test_size / (test_size + validation_size)
    assert round(v_size + t_size) == 1., "Data splits do not sum to 1"
    test_tracks, valid_tracks, test_pianist, valid_pianist = train_test_split(
        temp_df['track'],
        temp_df['pianist'],
        stratify=temp_df['database'],
        test_size=t_size,
        train_size=v_size,  # Use validation as train
        random_state=SEED,
        shuffle=True
    )
    # Format dataframes
    test_df = fmt(test_tracks, test_pianist)
    valid_df = fmt(valid_tracks, valid_pianist)
    # Quick checks to ensure that no tracks are in the same split
    assert not any(i in test_df['track'].values for i in train_df['track'].values), 'Train tracks found in test set!'
    assert not any(i in valid_df['track'].values for i in train_df['track'].values), 'Train tracks found in valid set!'
    assert not any(i in valid_df['track'].values for i in test_df['track'].values), 'Test tracks found in valid set!'
    return train_df, test_df, valid_df


def generate(
        split_path: str,
        train_size: float,
        test_size: float,
        validation_size: float,
        min_pianist_minutes: float
) -> None:
    """Generates data splits with required `size`s and saves in `split_path`"""

    assert round(train_size + test_size + validation_size) == 1.0, "Size of all splits must sum to 1!"
    logger.info(f'Generating data splits with sizes train {train_size}, test {test_size}, validation {validation_size}')
    # Create the folder for the desired split
    split_path = os.path.join(get_project_root(), 'references/data_splits', split_path)
    logger.info(f'Splits will be saved as CSV files in {split_path}')
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    # Get the valid tracks from all databases
    loaded_tracks = get_tracks()
    # Convert to a dataframe
    df = pd.DataFrame(loaded_tracks, columns=['database', 'pianist', 'track', 'duration'])
    total_hours = df['duration'].sum() / 3600
    logger.info(f'Found {len(df)} tracks from {len(IN_JTD_AND_PIJAMA)} pianists, duration {total_hours:.4} hours')
    # Remove pianists who don't have enough audio
    fmt_df = remove_pianists_without_enough_audio(df, min_pianist_minutes)
    fmt_hours = fmt_df['duration'].sum() / 3600
    logger.info(f'... removed {fmt_df["pianist"].nunique()} pianists with <{min_pianist_minutes} minutes of audio')
    logger.info(f'... {len(fmt_df)} tracks remain, duration {fmt_hours:.4} hours')
    # Assign numbers for each pianist (sorted alphabetically) and replace their name with this
    mapping = {k: v for v, k in enumerate(sorted(fmt_df['pianist'].unique()))}
    _ = pd.Series(mapping).rename('idx').to_csv(os.path.join(split_path, 'pianist_mapping.csv'))  # dump CSV
    pd.options.mode.chained_assignment = None
    fmt_df['pianist'] = fmt_df['pianist'].map(mapping)
    pd.options.mode.chained_assignment = 'warn'
    # Split into train, test, and validation dataframes
    df_train, df_test, df_valid = generate_split(fmt_df, train_size, test_size, validation_size)
    assert len(df_train) + len(df_test) + len(df_valid) == len(fmt_df), "Data splits are not using all tracks"
    # Create the splits as individual CSV files inside the directory
    for split_df, split in zip([df_train, df_test, df_valid], ['train', 'test', 'validation']):
        logger.info(f'Split {split} has {len(split_df)} tracks!')
        split_df['track'] = split_df['database'] + "/" + split_df['track']
        _ = (
            split_df.drop(columns=['database'])
            .reset_index(drop=True)
            .to_csv(os.path.join(split_path, f"{split}_split.csv"))
        )
    logger.info(f'Finished, splits saved to {split_path}')


if __name__ == "__main__":
    import argparse

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description='Generate data splits for model training, testing, and validation')
    parser.add_argument(
        "-p", "--split-path", default="20class_80min", type=str,
        help="Path to save split inside `references/data_splits`"
    )
    parser.add_argument(
        "-t", "--train-size", default=TRAIN_SIZE, type=float,
        help=f"Size of train split, defaults to 0.8"
    )
    parser.add_argument(
        "-e", "--test-size", default=TEST_SIZE, type=float,
        help="Size of test split, defaults to 0.1"
    )
    parser.add_argument(
        "-v", "--validation-size", default=VALIDATION_SIZE, type=float,
        help="Size of validation split, defaults to 0.1"
    )
    parser.add_argument(
        "-m", "--min-pianist-minutes", default=MIN_PIANIST_MINUTES, type=float,
        help="Minimum minutes of audio for a pianist to be included in split, defaults to 80"
    )
    parser.add_argument(
        "-s", "--seed", default=SEED, type=int, help="Seed to use in reproducible randomization. Defaults to 42"
    )
    # Parse all arguments from the command line
    args = vars(parser.parse_args())
    seed_everything(args['seed'])
    # Create the data split
    generate(
        args['split_path'], args['train_size'], args['test_size'], args['validation_size'], args['min_pianist_minutes']
    )
