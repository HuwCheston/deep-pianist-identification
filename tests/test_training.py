#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for modules in deep_pianist_identification/training.py"""

import os
import unittest

import torch
import yaml
from torchmetrics.classification import (
    BinaryAccuracy, MulticlassAccuracy, BinaryConfusionMatrix, MulticlassConfusionMatrix
)
from torchmetrics.functional import accuracy

from deep_pianist_identification.encoders import DisentangleNet
from deep_pianist_identification.training import (
    TrainModule, DEFAULT_CONFIG, groupby_tracks, parse_config_yaml, NoOpScheduler
)
from deep_pianist_identification.utils import seed_everything, SEED, DEVICE, get_project_root


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

    def test_classification_head_size(self):
        # Iterate through all the encoder modules we want to train
        for encoder in ['cnn', 'crnn', 'disentangle']:
            for dataset, n_classes in zip(['25class_0min', '20class_80min'], [25, 20]):
                # Set the encoder module and datset correctly in our configuration dictionary
                cfg = DEFAULT_CONFIG.copy()
                cfg['data_split_dir'] = dataset
                cfg['encoder_module'] = encoder
                # Create the train module and assert that the classification head output size is correct
                tm = TrainModule(**cfg)
                self.assertEqual(tm.num_classes, n_classes)
                self.assertEqual(tm.model.fc2.fc.out_features, n_classes)
                # Testing binary classification with the same dataset
                cfg = DEFAULT_CONFIG.copy()
                cfg['data_split_dir'] = dataset
                cfg['encoder_module'] = encoder
                cfg['classify_dataset'] = True
                # Create the train module and assert that the classification head output size is correct
                tm = TrainModule(**cfg)
                self.assertEqual(tm.num_classes, 2)
                self.assertEqual(tm.model.fc2.fc.out_features, 2)

    def test_targets_range(self):
        for encoder in ['cnn', 'crnn', 'disentangle']:
            for dataset, n_classes in zip(['25class_0min', '20class_80min'], [25, 20]):
                # Set the encoder module and datset correctly in our configuration dictionary
                cfg = DEFAULT_CONFIG.copy()
                cfg['data_split_dir'] = dataset
                cfg['encoder_module'] = encoder
                # Set batch size and number of clips to process correctly
                cfg['batch_size'] = 5
                cfg['train_dataset_cfg']['n_clips'] = 50
                cfg['test_dataset_cfg']['n_clips'] = 50
                # Create the train module
                tm = TrainModule(**cfg)
                # Iterate through a few batches in the training and test loader
                n_batches = 0
                for loader in [tm.train_loader, tm.test_loader]:
                    for batch, targets, _ in loader:
                        # Range of target values should be within range supplied
                        self.assertTrue(all([i in range(n_classes) for i in targets.tolist()]))
                        self.assertFalse(any([i not in range(n_classes) for i in targets.tolist()]))
                        # Break out once we've checked ten batches
                        n_batches += 1
                        if n_batches > 10:
                            break

    def test_target_mapping(self):
        # IDXs for first and last pianist in the dataset
        test_mapping = [
            {0: "Abdullah Ibrahim", 24: "Tommy Flanagan"},
            {0: "Abdullah Ibrahim", 19: "Tommy Flanagan"},
        ]
        for dataset, mapper in zip(['25class_0min', '20class_80min'], test_mapping):
            # Set the encoder module and datset correctly in our configuration dictionary
            cfg = DEFAULT_CONFIG.copy()
            cfg['data_split_dir'] = dataset
            # Set the batch size
            cfg['batch_size'] = 5
            cfg['train_dataset_cfg']['n_clips'] = None
            cfg['test_dataset_cfg']['n_clips'] = None
            # Create the train module
            tm = TrainModule(**cfg)
            for loader in [tm.train_loader, tm.test_loader]:
                for batch, targets, tracks in loader:
                    targets = targets.tolist()
                    # If we have any of our two test pianists in the batch
                    if any([i in targets for i in list(mapper.keys())]):
                        # Check that the surname of the pianist is in any one of our returned track names
                        surnames = [i.lower().split(' ')[1] for i in mapper.values()]
                        self.assertTrue(any([any([s_ in t for t in tracks]) for s_ in surnames]))
                        # Break out to test the next part of the loop
                        break

    def test_arguments_parsing(self):
        # Load up a YAML file: no masking, 4 conv layers, 3 pool layers, no attention, average pooling
        cfg_path = "disentangle-jtd+pijama-4conv3pool-nomask-augment50-noattention-avgpool.yaml"
        tester = yaml.safe_load(os.path.join(get_project_root(), 'config', 'disentangle-mask', cfg_path))
        parsed = dict(config=tester)
        parse_config_yaml(parsed)
        # Pass the config through into the training module
        trainer = TrainModule(**parsed)

        # Check the training dataloader
        train_loader = trainer.train_loader.dataset
        # Should be using data augmentation with 50% probability
        self.assertTrue(train_loader.data_augmentation)
        self.assertEqual(train_loader.augmentation_probability, 0.5)
        # Should be using jitter
        self.assertTrue(train_loader.jitter_start)
        # Should be using multichannel piano rolls
        self.assertTrue(train_loader.multichannel)
        self.assertEqual(train_loader.creator_func, train_loader.get_multichannel_piano_roll)
        # Should not be classifying dataset
        self.assertFalse(train_loader.classify_dataset)

        # Check the training dataloader
        test_loader = trainer.test_loader.dataset
        # Should not be using data augmentation
        self.assertFalse(test_loader.data_augmentation)
        # Should not be using jitter
        self.assertFalse(test_loader.jitter_start)
        # Should be using multichannel piano rolls
        self.assertTrue(test_loader.multichannel)
        self.assertEqual(test_loader.creator_func, test_loader.get_multichannel_piano_roll)
        # Should not be classifying dataset
        self.assertFalse(test_loader.classify_dataset)

        # Check the model
        model = trainer.model
        # Should be a disentangled encoder
        self.assertIsInstance(model, DisentangleNet)
        # Should not use masking
        self.assertFalse(model.use_masking)
        # Should have four convolution layers
        self.assertEqual(len(model.melody_concept.layers), 4)
        # Should have three pooling layers
        pools = [i for i in model.melody_concept.layers if hasattr(i, "avgpool")]
        self.assertEqual(len(pools), 3)
        # Should not use attention
        self.assertFalse(hasattr(model, "self_attention"))
        self.assertFalse(model.use_attention)
        # Should use global AVERAGE pooling
        self.assertIsInstance(model.pooling, torch.nn.AdaptiveAvgPool1d)

        # Check the optimizer
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        self.assertIsInstance(trainer.scheduler, NoOpScheduler)


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
