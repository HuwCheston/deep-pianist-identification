#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Converts files from PiJAMA dataset to the format used in JTD"""

import json
import os
import random
import shutil
import string

from tqdm import tqdm

from deep_pianist_identification.utils import get_project_root, remove_punctuation


def create_track_hash(seed: int = None) -> str:
    """Creates a fake track hash in the format used by MusicBrainz"""

    def hasher() -> str:
        return random.choice(string.ascii_lowercase + string.digits)

    # If we haven't provided a random seed, pick a random one
    if seed is None:
        seed = random.randint(1, 42 ** 42)
    # Set the random state for reproducible results
    random.seed(seed)
    # Return the hash, format 'XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX'
    return '-'.join((''.join(hasher() for _ in range(i)) for i in [8, 4, 4, 4, 12]))


def format_performer_name(name: str) -> str:
    """Returns a name in the format lastnamefirstinitial, i.e. `Bill Evans` -> `evansb`"""
    splitted = name.split(' ')
    if len(splitted) > 1:
        return remove_punctuation(splitted[1].lower() + splitted[0][0].lower())
    # If the person only has one word in their name, i.e. `Hiromi`
    else:
        return remove_punctuation(name).lower()


if __name__ == '__main__':
    pijama_root = os.path.join(get_project_root(), 'data/pijama')
    # Check to make sure that we have the correct files extracted
    if not os.path.exists(pijama_root):
        raise FileNotFoundError('Extract `midi_kong.zip` from https://zenodo.org/records/8354955 to `./data/pijama`')
    # We have two subfolders for PiJAMA
    for ext in ['live', 'studio']:
        # Each pianist gets their own folder
        for pianist in tqdm(os.listdir(os.path.join(pijama_root, ext)), desc=f'Formatting MIDI from {ext} folder'):
            pianist_fmt = format_performer_name(pianist)
            # Each album by the pianist gets its own folder
            albums = os.path.join(pijama_root, ext, pianist)
            for album in os.listdir(albums):
                # Each album has potentially multiple tracks
                tracks = os.path.join(albums, album)
                for track in os.listdir(tracks):
                    # We only use the first five words of each track
                    track_fmt = ''.join(remove_punctuation(track.split('.')[0]).lower().split(' ')[:6])
                    # Create a random track in MBZ format
                    track_id = create_track_hash()
                    track_id_short = track_id.split('-')[0]
                    # This is the file format used in JTD
                    new_fname = f'{pianist_fmt}-{track_fmt}-unaccompanied-xxxx-{track_id_short}'
                    # Creating a metadata dictionary in the same format used for JTD
                    meta = {
                        'track_name': remove_punctuation(track.split('.')[0]),
                        'album_name': remove_punctuation(album),
                        'recording_year': None,
                        'bandleader': remove_punctuation(pianist),
                        'pianist': remove_punctuation(pianist),
                        'mbz_id': track_id,
                        'live': ext == 'live',
                        'fname': new_fname
                    }
                    # Create a new directory for this track
                    newdir = os.path.join(pijama_root, new_fname)
                    os.makedirs(newdir, exist_ok=True)
                    # Dump the metadata JSON
                    json.dump(meta, open(os.path.join(newdir, 'metadata.json'), 'w'), ensure_ascii=False, indent=4)
                    # Copy the MIDI file and rename to something universal
                    shutil.copy(os.path.join(tracks, track), os.path.join(newdir, 'piano_midi.mid'))
        # Clean up by removing the `live` or `studio` folder
        shutil.rmtree(os.path.join(pijama_root, ext), ignore_errors=True)
