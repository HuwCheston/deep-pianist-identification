#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create plots and HTML for pianist melodic progression usage"""

import json
import os

from PIL import Image
from bs4 import BeautifulSoup
from pretty_midi import PrettyMIDI, Note, Instrument
from tqdm import tqdm

from deep_pianist_identification import utils
from deep_pianist_identification.extractors import MelodyExtractor, ExtractorError
from deep_pianist_identification.preprocessing import m21_utils
from deep_pianist_identification.whitebox import wb_utils
from deep_pianist_identification.whitebox.features import MELODY_QUANTIZE_RESOLUTION

DATASET = "20class_80min"
INPUT_DIR = os.path.join(utils.get_project_root(), 'reports/figures/whitebox/lr_weights')
PIANIST_MAPPING = {v: k for k, v in utils.get_class_mapping(DATASET).items()}

MAX_EXAMPLES_PER_CLIP = 3  # we'll only get up to this many examples from the same clip

NOTE_VELOCITY = utils.MAX_VELOCITY // 2  # we'll set the velocity of all notes to this value


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


def extract_feature_matches(
        melody_midi: PrettyMIDI,
        desired_feature: list[int],
) -> list[Note]:
    """Extract appearances of a desired feature from the melody of a clip"""
    # If we haven't been able to extract the melody for whatever reason (usually not enough notes in clip)
    if melody_midi is None:
        return []
    # Sort the notes by onset start time and calculate the intervals
    melody = sorted(melody_midi.instruments[0].notes, key=lambda x: x.start)
    intervals = [i2.pitch - i1.pitch for i1, i2 in zip(melody[0:], melody[1:])]
    # Extract all the n-grams with the same n as our desired feature
    n = len(desired_feature)
    all_features = [(intervals[i: i + n]) for i in range(len(intervals) - n + 1)]
    # Subset to get only indices of desired n-grams
    desired_feature_idxs = [i for i, ngram in enumerate(all_features) if ngram == desired_feature]
    # Get starting and stopping timestamps for every appearance of this feature
    melody_matches = [[melody[idx + n_] for n_ in range(n + 1)] for idx in desired_feature_idxs]
    # Flatten the list of lists so we get all notes contributing to this feature within the clip as a single list
    return [adjust_velocity(x) for xs in melody_matches for x in xs]


def get_melody(clip_path: str) -> MelodyExtractor | None:
    """Extract melody from a given clip and return a PrettyMIDI object"""
    # Load the clip in as a PrettyMIDI object
    loaded = PrettyMIDI(clip_path)
    # Try and create the piano roll, but return None if we end up with zero notes
    try:
        return MelodyExtractor(loaded, clip_start=0, quantize_resolution=MELODY_QUANTIZE_RESOLUTION)
    except ExtractorError as e:
        print(f'Errored for {clip_path}, skipping! {e}')
        return None


def generate_clip_metadata(clip_name: str, asset_path: str) -> dict:
    """Generates a JSON of metadata for a pianist to be saved in ./app/assets/metadata/cav_sensitivity"""
    # Get the path for the original track metadata.json inside ./data/raw and make sure it exists
    metadata_path = os.path.join(
        os.path.sep.join(clip_name.replace(f'clips{os.path.sep}', f'raw{os.path.sep}').split(os.path.sep)[:-1]),
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
        asset_path=asset_path
    )


def adjust_velocity(note: Note, new_velocity: int = NOTE_VELOCITY) -> Note:
    """Adjusts the velocity of a note to `new_velocity` while keeping all other parameters the same"""
    return Note(velocity=new_velocity, pitch=note.pitch, start=note.start, end=note.end)


def combine_melody_accompaniment(
        melody: PrettyMIDI,
        accompaniment: PrettyMIDI,
        matched_melody: list[Note]
) -> PrettyMIDI:
    """Combines inputs into a single MIDI with one instrument using melody notes and other non-melody notes"""
    # These are the onset times associated with melody notes that match the feature
    matched_melody_onsets = [i.start for i in matched_melody]
    # Now we grab all melody notes that are not matching with the feature
    melody_non_match = [adjust_velocity(m) for m in melody.instruments[0].notes if m.start not in matched_melody_onsets]
    # We can also adjust all accompaniment notes
    accompaniment_notes = [adjust_velocity(a) for a in accompaniment.instruments[0].notes]
    # Combine both into a single list
    non_matching_melody_accomp = melody_non_match + accompaniment_notes
    # Create the new MIDI object:
    #  the first instrument is the melody notes matching the feature, the second instrument is every other note
    #  by setting the instruments in this way, we can assign different colors to them on the piano roll visualiser
    new_midi = PrettyMIDI()
    melody_instrument = Instrument(program=0)
    melody_instrument.notes = matched_melody
    non_melody_instrument = Instrument(program=0)
    non_melody_instrument.notes = non_matching_melody_accomp
    new_midi.instruments = [melody_instrument, non_melody_instrument]
    return new_midi


def sort_metadata_list_by_feature_appearances(metadata_list: list[dict]) -> list[dict]:
    """Given a list of dictionaries, sort by the number of times a feature appears in a track and add track number"""
    # Sort in descending order by the number of times a feature appears in a track
    sorter = sorted(metadata_list, reverse=True, key=lambda x: x['n_appearances_in_track'])
    for idx, track in enumerate(sorter, 1):
        # Add on a track number before the track name
        track['track_name'] = f'{idx}. {track["track_name"]} ({track["n_appearances_in_track"]} appearances)'
        yield track


def process_midi_and_generate_metadata(
        pianist_name: str,
        pianist_features: dict,
        pianist_tracks: list
) -> dict:
    """Process all MIDI examples and generate a dictionary of metadata"""
    # We'll store results in here
    new_dict = {}
    # Iterate over all features that this pianist is associated with
    for feature_idx, feature in enumerate(pianist_features):
        feature_results_list = []
        count = 0  # iterate every time a clip contains the feature
        # Iterate through all tracks by the pianist
        for track_fpath, n_clips, _ in tqdm(
                pianist_tracks,
                desc=f'Getting matches for {pianist_name}, feature {feature_idx + 1}'
        ):
            # Iterate over all the clips associated with this track
            for clip_idx in range(n_clips):
                # Load in the clip and extract melody using the skyline
                clip_fpath = os.path.join(track_fpath, f'clip_{str(clip_idx).zfill(3)}.mid')
                # If we can't extract the melody for whatever reason, skip over this clip
                melody_extract = get_melody(clip_fpath)
                if melody_extract is None:
                    continue
                # Subset to get the quantised melody and accompaniment as separate objects
                melody, accomp = melody_extract.skylined, melody_extract.accomp
                # Extract timestamps for where the feature occurs in the skylined melody
                melody_matches = extract_feature_matches(melody, feature)
                # If we can't find this feature in this clip, skip over
                if len(melody_matches) == 0:
                    continue
                # Combine everything into one MIDI, with instrument[0] assigned to melody notes that match the
                #  current feature and instrument[1] assigned to every other melody and accompaniment note
                fmtted_midi = combine_melody_accompaniment(melody, accomp, melody_matches)
                # Write the MIDI to disk with the required filename
                fname = f"{pianist_name.lower().replace(' ', '_')}_{str(feature_idx).zfill(3)}_{str(count).zfill(3)}"
                fpath = os.path.join(utils.get_project_root(), f'app/assets/midi/melody_examples/{fname}.mid')
                fmtted_midi.write(fpath)
                # Generate metadata for the current clip and append to the list associated with this feature
                metadata = generate_clip_metadata(clip_fpath, fname)
                metadata['n_appearances_in_track'] = len(melody_matches) // (len(feature) + 1)
                # Update the dictionary associated with this feature
                feature_results_list.append(metadata)
                count += 1
        new_dict[str(feature)] = list(sort_metadata_list_by_feature_appearances(feature_results_list))
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
        feature_image = m21_utils.score_to_image_array(feature_m21, dpi=800, right_crop=0.92)
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
        n_appearances = sum([i['n_appearances_in_track'] for i in pianist_metadata[feat]])
        n_tracks = len(pianist_metadata[feat])
        feat_pitches = m21_utils.intervals_to_pitches(feat)
        tag.string.replace_with(f'{feat_pitches}: {n_appearances} appearances in {n_tracks} clips')
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
        pianist_metadata = process_midi_and_generate_metadata(pianist_name, pianist_features, pianist_clips)
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
