#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for modules in deep_pianist_identification/training.py"""

import unittest

import torch

from deep_pianist_identification.training import NoOpScheduler
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


if __name__ == '__main__':
    seed_everything(SEED)
    unittest.main()
