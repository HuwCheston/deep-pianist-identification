#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create plots and HTML for pianist melodic progression usage"""

import json
import os

import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from loguru import logger
from pretty_midi import PrettyMIDI, Note, Instrument
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import MelodyExtractor, ExtractorError
from deep_pianist_identification.preprocessing import m21_utils
from deep_pianist_identification.whitebox import wb_utils

DATASET = "20class_80min"
INPUT_DIR = os.path.join(utils.get_project_root(), 'reports/figures/whitebox/lr_weights')
PIANIST_MAPPING = {v: k for k, v in utils.get_class_mapping(DATASET).items()}

MAX_EXAMPLES_PER_CLIP = 3  # we'll only get up to this many examples from the same clip


def parse_input(input_fpath: str, extension: str = None) -> str:
    """Parses an input filepath with given extension and checks that it exists"""
    # Add the extension to the filepath if we've provided it
    if extension is not None:
        if not input_fpath.endswith(extension):
            input_fpath += f'.{extension}'
    path = os.path.join(INPUT_DIR, input_fpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {input_fpath} inside {INPUT_DIR}. Have you run `create_classifier.py`?"
        )
    return path


def extract_feature_timestamps(
        clip_midi: PrettyMIDI,
        desired_feature: list[int],
) -> list[tuple]:
    """Extract start and stop timestamps where a given feature appears in a clip"""
    # If we haven't been able to extract the melody for whatever reason (usually not enough notes in clip)
    if clip_midi is None:
        return []
    # Sort the notes by onset start time and calculate the intervals
    melody = sorted(clip_midi.instruments[0].notes, key=lambda x: x.start)
    intervals = [i2.pitch - i1.pitch for i1, i2 in zip(melody[0:], melody[1:])]
    # Extract all the n-grams with the same n as our desired feature
    n = len(desired_feature)
    all_features = [(intervals[i: i + n]) for i in range(len(intervals) - n + 1)]
    # Subset to get only indices of desired n-grams
    desired_feature_idxs = [i for i, ngram in enumerate(all_features) if ngram == desired_feature]
    # Get starting and stopping timestamps for every appearance of this feature
    #  add a bit of padding either side of the window to make matching easier
    return [(melody[idx].start, melody[idx + n].end) for idx in desired_feature_idxs]


def get_melody(clip_path: str) -> PrettyMIDI | None:
    """Extract melody from a given clip and return a PrettyMIDI object"""
    # Load the clip in as a PrettyMIDI object
    loaded = PrettyMIDI(clip_path)
    # Try and create the piano roll, but return None if we end up with zero notes
    try:
        return MelodyExtractor(loaded, clip_start=0).output_midi
    except ExtractorError as e:
        print(f'Errored for {clip_path}, skipping! {e}')
        return None


def note_list_to_pretty_midi(note_list: list[Note]) -> PrettyMIDI:
    """Convert a python list of pretty_midi.Note objects to a pretty_midi.PrettyMIDI instance"""
    new_midi = PrettyMIDI()
    new_instrument = Instrument(program=0)
    new_instrument.notes = note_list
    new_midi.instruments = [new_instrument]
    return new_midi


def generate_clip_metadata(clip_name: str) -> dict:
    """Generates a JSON of metadata for a pianist to be saved in ./app/assets/metadata/cav_sensitivity"""
    # Get the path for the original track metadata.json inside ./data/raw and make sure it exists
    metadata_path = os.path.join(
        os.path.sep.join(clip_name.replace('clips', 'raw').split(os.path.sep)[:-1]),
        'metadata.json'
    )
    assert os.path.exists(metadata_path)
    # Load in the metadata as a dictionary
    with open(metadata_path, 'r') as f:
        metadata_json = json.load(f)
    # Create the dictionary and append to the list for this concept
    return dict(
        track_name=metadata_json["track_name"],
        album_name=metadata_json["album_name"],
        recording_year=metadata_json["recording_year"],
    )


def truncate_midi_between_timestamps(midi: PrettyMIDI, start: float, stop: float, pad: float = 2.) -> PrettyMIDI:
    # Get notes from the MIDI object that are contained within the span of our window
    truncated = [n for n in midi.instruments[0].notes if (start - pad) <= n.start <= (stop + pad)]
    # Edit note start and end times
    try:
        first_start = min(truncated, key=lambda x: x.start).start
    except ValueError:  # why is this happening occasionally?
        logger.warning(f"... failed with start {start}, stop {stop}")
        return None
    first_end = min(truncated, key=lambda x: x.end).end
    offset = [
        Note(start=n.start - first_start, end=n.end - first_end, pitch=n.pitch, velocity=n.velocity)
        for n in truncated
    ]
    # Convert the list to a pretty MIDI object
    return note_list_to_pretty_midi(offset)


def get_feature_matches(pianist_name: str, pianist_features: dict, pianist_tracks: list) -> dict:
    """Get timestamps from a pianist's tracks that match features they are associated with"""
    # We'll store results in here
    new_dict = {}
    # Iterate over all features that this pianist is associated with
    for feature_idx, feature in enumerate(pianist_features, 1):
        count = 0  # used to track number of times we've seen this feature
        new_dict[str(feature)] = []
        # Iterate through all tracks by the pianist
        for track_fpath, n_clips, _ in tqdm(
                pianist_tracks, desc=f'Getting matches for {pianist_name}, feature {feature_idx}'
        ):
            # Iterate over all the clips associated with this track
            for clip_idx in range(n_clips):
                # Load in the clip and extract melody using the skyline
                clip_fpath = os.path.join(track_fpath, f'clip_{str(clip_idx).zfill(3)}.mid')
                # If we can't extract the melody for whatever reason, skip over this clip
                mel = get_melody(clip_fpath)
                if mel is None:
                    continue
                matches = extract_feature_timestamps(mel, feature)
                if len(matches) == 0:
                    continue
                full_midi = PrettyMIDI(clip_fpath)
                # Randomly shuffle the matches so we don't just get loads of appearances of the same pattern in a row
                np.random.shuffle(matches)
                n_clip_examples_seen = 0
                for match_start, match_end in matches:
                    clip_metadata = generate_clip_metadata(clip_fpath)
                    truncated = truncate_midi_between_timestamps(full_midi, match_start, match_end)
                    if truncated is None:
                        continue
                    # Create the filepath for this object
                    fname = f"{pianist_name.lower().replace(' ', '_')}_{str(feature_idx).zfill(3)}_{str(count).zfill(3)}"
                    fpath = os.path.join(utils.get_project_root(), f'app/assets/midi/melody_examples/{fname}.mid')
                    # Dump the MIDI object at the filepath
                    truncated.write(fpath)
                    # Add information to dictionary
                    clip_metadata["track_name"] = f'{count + 1}. {clip_metadata["track_name"]}'
                    clip_metadata["asset_path"] = fname
                    new_dict[str(feature)].append(clip_metadata)
                    count += 1  # increment counter for number of times we've seen this feature in total
                    n_clip_examples_seen += 1  # increment counter for times we've seen this feature this clip
                    # Stop getting examples from this clip once we've got enough
                    if n_clip_examples_seen > MAX_EXAMPLES_PER_CLIP:
                        break
    return new_dict


def get_top_melody_features(pianists: list[str]) -> dict:
    with open(parse_input("lr_weights.json"), 'r') as js_loc:
        weights_json = json.load(js_loc)
    return {p: [eval(i.replace('M_', '')) for i in weights_json[p]['melody']['top_names']] for p in pianists}


def convert_melody_feature_to_m21(melody_feature: list[int]):
    """From a melody feature (list of integers), convert to a music21.Score object"""
    # Here we're working with arrays
    pitches = m21_utils.intervals_to_pitches(melody_feature)
    centered = m21_utils.center_feature(pitches)
    # Now we're working with music21 objects
    measure = m21_utils.melody_feature_to_m21_measure(centered)
    score = m21_utils.part_to_score(m21_utils.measure_to_part(measure))
    m21_utils.adjust_m21_melody_to_partitura(  # Adjusts the spelling to match the output of partitura
        score,
        m21_utils.estimate_m21_score_spelling(score)
    )
    return score


def save_melody_feature_outputs(melody_feature: list[int], feature_rank: int, pianist_name: str) -> None:
    """Given a melody feature, rank, and pianist name, convert the feature to notation and save as an image"""
    # Set filenames and paths properly
    img_name = pianist_name.lower().replace(' ', '_') + "_" + str(feature_rank).zfill(3) + '.png'
    img_path = os.path.join(utils.get_project_root(), 'app/assets/images/melody_features', img_name)
    midi_name = pianist_name.lower().replace(' ', '_') + "_" + str(feature_rank).zfill(3) + '.mid'
    midi_path = os.path.join(utils.get_project_root(), 'app/assets/midi/melody_features', midi_name)
    # Convert feature to a music21.Score object
    feature_m21 = convert_melody_feature_to_m21(melody_feature)
    # If we haven't already created the image, go ahead and do this
    if not os.path.isfile(img_path):
        feature_image = m21_utils.score_to_image_array(feature_m21, dpi=800)
        # Get the original dimensions
        height, width, _ = feature_image.shape
        # Convert image array to pillow and save in desired location
        Image.fromarray(feature_image).crop((0, 50, width, height - 50)).save(img_path)
    # If we haven't already created the MIDI, do this as well
    if not os.path.isfile(midi_path):
        feature_m21.write('midi', midi_path)


def generate_html(pianist_metadata, pianist_name, tmp_pianist: str = "Abdullah Ibrahim"):
    """Creates HTML by updating template and stores in app/pages/melody/<pianist>.html"""
    # The concepts this pianist is associated with
    features = list(pianist_metadata.keys())
    # Load in the template and get the concept gallery, the element we need to modify
    template = os.path.join(utils.get_project_root(), 'app/pages/templates/melody.html')
    soup = BeautifulSoup(open(template, 'r'), features="lxml")
    gallery = soup.find_all('div', {"class": "concept-gallery"})[0]
    pianist_name_lower = pianist_name.lower().replace(' ', '_')
    # Iterating through each "row" in the concept gallery
    for feature_num, (tag, button, anchor, feat) in enumerate(zip(
            gallery.find_all('strong'),
            gallery.find_all('button'),
            gallery.find_all('a'),
            features
    )):
        # Updating all the headings with the correct concept name
        tag.string.replace_with(feat)
        # Updating all the buttons to use the correct argument
        button['onclick'] = f"playExampleMidi({feature_num}, '{pianist_name}')"
        # Updating all the anchor tags
        anchor['alt'] = feat
        feat_fmt = feat.replace("'", r"\'").replace('"', r'\"')
        anchor['onclick'] = f"""popSelect('{feat_fmt}', '{pianist_name}')"""
        # We can also grab the image from each anchor tag
        img = anchor.find('img')
        img['alt'] = feat
        img['src'] = f"../../assets/images/melody_features/{pianist_name_lower}_{str(feature_num).zfill(3)}.png"
    # Create the output path that we'll save the HTML in
    out_path = os.path.join(utils.get_project_root(), 'app/pages/melody', pianist_name_lower + '.html')
    # Write HTML and replace any reference to our placeholder pianist (Abdullah Ibrahim, default) with the actual one
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(str(soup).replace(tmp_pianist, pianist_name))


def main():
    # Get the names of the top N melody features associated with each pianist
    pianist_top_features = get_top_melody_features(PIANIST_MAPPING.keys())
    # Load in filenames, metadata etc. for all clips in the dataset regardless of their split (e.g., train, test)
    clips = [clip for data_split in wb_utils.get_all_clips(DATASET) for clip in data_split]
    # Iterate over pianists and the features associated with them
    for pianist_name, pianist_features in pianist_top_features.items():
        # Iterate over their associated features, with a counter
        for feature_rank, feature in enumerate(pianist_features):
            # Convert these features to images and MIDI and dump inside the assets directory
            save_melody_feature_outputs(feature, feature_rank, pianist_name)
        # Each clip is in the form (track name, number of clips, pianist idx)
        pianist_clips = [i for i in clips if i[2] == PIANIST_MAPPING[pianist_name]]
        # Get the tracks that match each feature associated with the pianist
        pianist_metadata = get_feature_matches(pianist_name, pianist_features, pianist_clips)
        # Dump the metadata
        metadata_save_path = os.path.join(
            utils.get_project_root(),
            'app/assets/metadata/melody_features',
            pianist_name.replace(' ', '_').lower() + '.json'
        )
        with open(metadata_save_path, 'w') as js_out:
            json.dump(pianist_metadata, js_out, indent=4, ensure_ascii=False)
        generate_html(pianist_metadata, pianist_name)


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)  # we shouldn't be doing anything random, but why not?
    main()
