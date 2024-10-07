#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions and variables for random forest baselines"""

import csv
import json
import os
from ast import literal_eval
from tempfile import NamedTemporaryFile

from tenacity import retry, retry_if_exception_type

# These are the parameters we'll sample from when optimizing
OPTIMIZE_PARAMS = dict(
    # The number of trees to grow in the forest
    n_estimators=[i for i in range(10, 1001, 1)],
    # Max number of features considered for splitting a node
    max_features=[None, 'sqrt', 'log2'],
    # Max number of levels in each tree
    max_depth=[None, *[i for i in range(1, 101, 1)]],
    # Minimum number of samples required to split a node
    min_samples_split=[i for i in range(2, 11)],
    # Minimum number of samples required at each leaf node
    min_samples_leaf=[i for i in range(1, 11)],
    # Whether to sample data points with our without replacement
    bootstrap=[True, False],
)
N_ITER = 10000
VALID_NGRAMS = 10
NGRAMS = [3, 4]


@retry(retry=retry_if_exception_type(json.JSONDecodeError))
def safe_load_csv(csv_path: str) -> dict:
    """Simple wrapper around `csv.load` that catches errors when working on the same file in multiple threads"""

    def eval_(i):
        try:
            return literal_eval(i)
        except (ValueError, SyntaxError) as _:
            return str(i)

    with open(csv_path, "r+") as in_file:
        return [{k: eval_(v) for k, v in row.items()} for row in csv.DictReader(in_file, skipinitialspace=True)]


def safe_save_csv(obj, fdir: str, fpath: str) -> None:
    """Simple wrapper around csv.DictWriter with protections to assist in multithreaded access"""
    # If we have an existing file with the same name, load it in and extend it with our new data
    try:
        existing_file = safe_load_csv(os.path.join(fdir, fpath))
    except FileNotFoundError:
        pass
    else:
        if isinstance(obj, dict):
            obj = [obj]
        obj = existing_file + obj

    # Create a new temporary file, in append mode
    temp_file = NamedTemporaryFile(mode='a', newline='', dir=fdir, delete=False, suffix='.csv')
    # Get our CSV header from the keys of the first dictionary, if we've passed in a list of dictionaries
    if isinstance(obj, list):
        keys = obj[0].keys()
    # Otherwise, if we've just passed in a dictionary, get the keys from it directly
    else:
        keys = obj.keys()
    # Open the temporary file and create a new dictionary writer with our given columns
    with temp_file as out_file:
        dict_writer = csv.DictWriter(out_file, keys)
        dict_writer.writeheader()
        # Write all the rows, if we've passed in a list
        if isinstance(obj, list):
            dict_writer.writerows(obj)
        # Alternatively, write a single row, if we've passed in a dictionary
        else:
            dict_writer.writerow(obj)

    @retry(retry=retry_if_exception_type(PermissionError))
    def replacer():
        os.replace(temp_file.name, os.path.join(fdir, fpath))

    replacer()
