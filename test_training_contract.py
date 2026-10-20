"""Regression checks for validation protocol, cache identity, and failure handling."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import train_complexes as train


class Batch:
    def __init__(self, value=2.):
        self.affinity = torch.tensor([value])
    def to(self, device):
        return self


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.))
    def forward(self, data):
        return torch.tensor([float('nan')]), self.bias.reshape(1)


class Contract(unittest.TestCase):
    def test_labels_reject_invalid_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'labels.csv'
            for rows in ['', 'a,1\na,2\n', ',1\n', '../a,1\n', 'a,nan\n', 'a,inf\n']:
                p.write_text('complex_id,y_true\n' + rows)
                with self.assertRaises(ValueError):
                    train.read_labels(p)
            p.write_text('complex_id,y_true\na,0\nb,2\n')
            self.assertEqual(train.read_labels(p), {'a': 0., 'b': 2.})

    def test_cache_tracks_protocol_labels_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a').mkdir()
            source = root / 'a/a_protein.pdb'
            source.write_text('first coordinates')
            key = lambda validation=False, seed=0, y=1.: train.cache_identity(
                root, {'a': y}, validation, seed)
            first = key()
            self.assertEqual(first, key(seed=5))  # no conformer in crystal training
            self.assertNotEqual(first, key(validation=True))
            self.assertNotEqual(key(True), key(True, 2))
            self.assertNotEqual(first, key(y=2.))
            source.write_text('new coordinates')
            self.assertNotEqual(first, key())

    def test_cache_reuse_and_role_separation(self):
        # Simulate PyG storage without executing pickle on the host or building proteins.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / 'labels.csv'
            labels.write_text('complex_id,y_true\na,1\n')
            calls = []
            def dataset(path, **kwargs):
                calls.append((path, kwargs))
                processed = Path(path) / 'processed'
                processed.mkdir(parents=True, exist_ok=True)
                (processed / 'data.pt').touch()
                return [1]
            with patch.object(train, 'TankBindDataSet', side_effect=dataset), patch.object(
                    train, 'featurise', return_value=(pd.DataFrame({'id': ['a']}), {}, {})) as features:
                for val in [False, False, True]:
                    train.build_dataset(str(root / 'cache'), str(root), str(labels), validation=val)
                self.assertEqual(features.call_count, 2)
                self.assertEqual(calls[0][0], calls[1][0])
                self.assertEqual(calls[1][1], {})
                self.assertNotEqual(calls[0][0], calls[2][0])

    def test_validation_uses_affinity_without_distance_target(self):
        model = Model()
        result = train.run_epoch(model, [Batch(2.)], 'cpu', None, .01, False)
        self.assertEqual(result['loss'], 1.)
        self.assertEqual(result['rmse'], 1.)
        self.assertIsNone(result['contact'])
        self.assertIsNone(model.bias.grad)

    def test_training_retains_two_losses_and_rejects_bad_gradients(self):
        batch = Batch(2.)
        batch.real_y_mask = torch.tensor([True])
        batch.dis_map = torch.tensor([3.])
        model = Model()
        model.forward = lambda data: (model.bias.reshape(1), model.bias.reshape(1))
        optimiser = torch.optim.SGD(model.parameters(), lr=.1)
        result = train.run_epoch(model, [batch], 'cpu', optimiser, .5, False)
        # d[(w-3)^2 + .5*(w-2)^2]/dw at w=1 is -5.
        self.assertAlmostEqual(model.bias.item(), 1.5)
        self.assertEqual(result['contact'], 4.)
        model.bias.data.zero_()
        model.forward = lambda data: (model.bias.sqrt().reshape(1), model.bias.reshape(1))
        with self.assertRaisesRegex(ValueError, 'nonfinite gradient'):
            train.run_epoch(model, [batch], 'cpu', optimiser, .5, False)
        self.assertEqual(model.bias.item(), 0.)

    def test_invalid_validation_fails(self):
        for batches in [[], [Batch(float('nan'))]]:
            with self.assertRaises(ValueError):
                train.run_epoch(Model(), batches, 'cpu', None, .01, False)
        model = Model()
        model.bias.data.fill_(float('inf'))
        with self.assertRaises(ValueError):
            train.run_epoch(model, [Batch()], 'cpu', None, .01, False)


if __name__ == '__main__':
    unittest.main()
