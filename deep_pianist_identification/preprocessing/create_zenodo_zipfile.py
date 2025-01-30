#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Small script to create the .zip file hosted on Zenodo"""

import os
import zipfile

from loguru import logger
from tqdm import tqdm

from deep_pianist_identification import utils

OUT_PATH = os.path.join(utils.get_project_root(), 'zenodo.zip')


def get_checkpoints() -> str:
    return os.path.join(
        utils.get_project_root(),
        'checkpoints',
        "disentangle-resnet-channel",
        "disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc",
        "checkpoint_099.pth"
    )


def get_data_files(skip_extensions: tuple[str] = ('.npy', '.gitkeep')) -> str:
    for root, dirs, files in os.walk(os.path.join(utils.get_project_root(), 'data')):
        for file in files:
            if not any(file.endswith(skip) for skip in skip_extensions):
                yield os.path.join(root, file)


def get_cav_files() -> str:
    cav_dir = os.path.join(utils.get_project_root(), 'references/cav_resources/voicings/midi_final')
    for root, dirs, files in os.walk(cav_dir):
        for file in files:
            yield os.path.join(root, file)


def get_figures(skip_extensions: tuple[str] = (".gitkeep", ".p")) -> str:
    for root, dirs, files in os.walk(os.path.join(utils.get_project_root(), 'reports/figures')):
        for file in files:
            if not any(file.endswith(skip) for skip in skip_extensions):
                yield os.path.join(root, file)


def validate_files(files_to_zip: list[str]):
    for file in files_to_zip:
        assert os.path.isfile(file)


def zip_everything(files_to_zip: list[str]):
    common_path = os.path.commonpath(files_to_zip)
    with zipfile.ZipFile(OUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for zipp in tqdm(files_to_zip, desc='Zipping...'):
            arcname = os.path.relpath(zipp, start=common_path)
            # Add the file to the zip file with the relative path
            zipf.write(zipp, arcname=arcname)


def main():
    logger.info('Zipping files for Zenodo upload...')
    tozip = [get_checkpoints()]
    tozip.extend(get_data_files())
    tozip.extend(get_cav_files())
    tozip.extend(get_figures())
    logger.info(f'... found {len(tozip)} files to zip!')
    validate_files(tozip)
    logger.info('... validated all files exist!')
    zip_everything(tozip)
    logger.info('... zipped all files successfully!')


if __name__ == "__main__":
    main()
