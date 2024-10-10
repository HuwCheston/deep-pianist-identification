#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for modules in deep_pianist_identification/training.py"""

import unittest

import torch
from torchmetrics.classification import (
    BinaryAccuracy, MulticlassAccuracy, BinaryConfusionMatrix, MulticlassConfusionMatrix
)
from torchmetrics.functional import accuracy

from deep_pianist_identification.training import NoOpScheduler, TrainModule, DEFAULT_CONFIG, groupby_tracks
from deep_pianist_identification.utils import seed_everything, SEED, DEVICE


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

    def test_vars_change(self):
        # Iterate through all the encoder modules we want to train
        for encoder in ['cnn', 'crnn', 'resnet50', 'disentangle']:
            # Set the encoder module correctly in our configuration dictionary
            cfg = DEFAULT_CONFIG.copy()
            cfg['batch_size'] = 1
            cfg["encoder_module"] = encoder
            cfg['checkpoint_cfg']['load_checkpoints'] = False
            cfg['checkpoint_cfg']['save_checkpoints'] = False
            # Set this argument correctly for the disentangle encoder
            if encoder == 'disentangle':
                cfg['train_dataset_cfg']['multichannel'] = True
                cfg['test_dataset_cfg']['multichannel'] = True
            # Create the training module
            tm = TrainModule(**cfg)
            tm.model.train()
            # Get model parameters
            params = [np for np in tm.model.named_parameters() if np[1].requires_grad]
            # Make a copy for later comparison
            initial_params = [(name, p.clone()) for (name, p) in params]
            # Take a batch from our dataloader and put on the correct device
            features, targets, _ = next(iter(tm.train_loader))
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            # Forwards and backwards pass through the model
            loss, _, __ = tm.step(features, targets)
            tm.optimizer.zero_grad()
            loss.backward()
            tm.optimizer.step()
            # Iterate through all the parameters in the model: they should have updated
            for (_, p0), (name, p1) in zip(initial_params, params):
                self.assertFalse(torch.equal(p0.to(DEVICE), p1.to(DEVICE)))

    def test_track_groupby(self):
        # Fake track names
        track_names = ['T1', 'T1', 'T2', 'T3', 'T3', 'T3', 'T4', 'T4', 'T5']
        # Track predictions; these can be thought of as the probabilities per clip, per class
        # Scale doesn't matter here, but the actual values will be softmaxed (i.e., sum to 1.0)
        batched_track_predictions = [
            [[0.5, 0.3, 0.9]],  # [  0.4,  0.5,  0.6  ] -> [2] predicted class index
            [[0.3, 0.7, 0.3]],
            [[0.7, 0.5, 0.1]],  # [  0.7,  0.5,  0.1  ] -> [0]
            [[0.1, 0.1, 0.9]],  # [  0.5,  0.5,  0.53 ] -> [2]
            [[0.9, 0.9, 0.1]],
            [[0.5, 0.5, 0.6]],
            [[0.4, 0.4, 0.7]],  # [  0.65, 0.6,  0.7  ] -> [2]
            [[0.9, 0.8, 0.7]],
            [[0.3, 0.5, 0.1]],  # [  0.3,  0.5,  0.1  ] -> [1]
        ]
        # Function expects a list of tensors
        batched_track_predictions = [torch.tensor(b).to(DEVICE) for b in batched_track_predictions]
        # These are the target classes of each clip
        batched_track_targets = [
            [0],
            [0],
            [1],
            [2],
            [2],
            [2],
            [0],
            [0],
            [1]
        ]
        # Function expects a list of tensors
        batched_track_targets = [torch.tensor(b).to(DEVICE) for b in batched_track_targets]
        expected_preds = torch.tensor([2, 0, 2, 2, 1]).to(DEVICE)
        expected_targets = torch.tensor([0, 1, 2, 0, 1]).to(DEVICE)
        actual_preds, actual_targets = groupby_tracks(track_names, batched_track_predictions, batched_track_targets)
        # We should return the expected predicted and target class indexes
        self.assertEqual(actual_preds.cpu().tolist(), expected_preds.cpu().tolist())
        self.assertEqual(actual_targets.cpu().tolist(), expected_targets.cpu().tolist())
        # Check that the accuracy is correct
        expected_acc = 0.4  # We could calculate this automatically...
        actual_acc = accuracy(actual_preds, actual_targets, task="multiclass", num_classes=3)
        self.assertEqual(expected_acc, actual_acc)


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
