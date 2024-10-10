#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for modules in deep_pianist_identification/training.py"""

import unittest

import torch
from torchmetrics.classification import (
    BinaryAccuracy, MulticlassAccuracy, BinaryConfusionMatrix, MulticlassConfusionMatrix
)

from deep_pianist_identification.training import NoOpScheduler, TrainModule, DEFAULT_CONFIG
from deep_pianist_identification.utils import seed_everything, SEED


class TrainTest(unittest.TestCase):
    def test_no_op_scheduler(self):
        # Define a simple model and optimizer
        model = torch.nn.Linear(in_features=1, out_features=32)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        # Define our custom scheduler
        scheduler = NoOpScheduler(optimizer=optim)
        # Iterate over some `epochs`
        for i in range(100):
            optim.zero_grad()
            # model would train here
            optim.step()
            # After stepping in the scheduler, the LR should not have changed from the initial value
            scheduler.step()
            for lr in scheduler.get_lr():
                self.assertEqual(lr, 0.001)
        # Check that we have a state dict
        self.assertTrue(hasattr(scheduler, 'state_dict'))

    def test_multiclass_parsing(self):
        tm = TrainModule(**DEFAULT_CONFIG)
        # Default configuration uses 25 classes, with Geri Allen as the 10th class
        self.assertEqual(tm.num_classes, 25)
        self.assertEqual(tm.class_mapping[10], "Geri Allen")
        # Class mapping keys should already be sorted
        self.assertEqual(sorted(tm.class_mapping.values()), list(tm.class_mapping.values()))
        # Accuracy measures should reflect the class mapping properly
        self.assertEqual(tm.acc_fn.num_classes, 25)
        self.assertEqual(tm.confusion_matrix_fn.num_classes, 25)
        self.assertIsInstance(tm.acc_fn, MulticlassAccuracy)
        self.assertIsInstance(tm.confusion_matrix_fn, MulticlassConfusionMatrix)
        # Check that classification head of the model is correct
        self.assertEqual(tm.model.fc2.fc.out_features, 25)

    def test_binary_parsing(self):
        # Default config classifies performers, not dataset, so change this
        cfg = DEFAULT_CONFIG.copy()
        cfg["classify_dataset"] = True
        tm = TrainModule(**cfg)
        # Check that class attributes update correctly
        self.assertEqual(tm.num_classes, 2)
        self.assertEqual(tm.class_mapping[0], "jtd")
        # Class mapping keys should already be sorted
        self.assertEqual(sorted(tm.class_mapping.values()), list(tm.class_mapping.values()))
        # Check that metrics update correctly
        self.assertIsInstance(tm.acc_fn, BinaryAccuracy)
        self.assertIsInstance(tm.confusion_matrix_fn, BinaryConfusionMatrix)
        # Check that classification head of the model is correct
        self.assertEqual(tm.model.fc2.fc.out_features, 2)

    def test_alt_split(self):
        # Change data split directory
        cfg = DEFAULT_CONFIG.copy()
        cfg["data_split_dir"] = "20class_80min"
        tm = TrainModule(**cfg)
        # Check that class attributes update correctly
        self.assertEqual(tm.num_classes, 20)
        self.assertEqual(tm.class_mapping[10], "Junior Mance")
        # Class mapping keys should already be sorted
        self.assertEqual(sorted(tm.class_mapping.values()), list(tm.class_mapping.values()))
        # Accuracy measures should reflect the class mapping properly
        self.assertEqual(tm.acc_fn.num_classes, 20)
        self.assertEqual(tm.confusion_matrix_fn.num_classes, 20)
        self.assertIsInstance(tm.acc_fn, MulticlassAccuracy)
        self.assertIsInstance(tm.confusion_matrix_fn, MulticlassConfusionMatrix)
        # Check that classification head of the model is correct
        self.assertEqual(tm.model.fc2.fc.out_features, 20)


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
