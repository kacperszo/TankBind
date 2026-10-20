"""Check actual model inputs and checkpoint selection against the inference CLI.

Run after the two-complex smoke train and inference, inside the isolated image.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_complexes import build_dataset
from data import TankBind_prediction


def main():
    val = build_dataset('/outputs/cache/labels', '/data', '/data/labels.csv', validation=True)
    inference = TankBind_prediction('/outputs/trained_work')
    assert len(val) == len(inference) == 2
    for i in range(len(val)):
        a, b = val[i], inference[i]
        # The extra affinity/mask metadata must not change any inference tensor.
        for key, expected in b.to_dict().items():
            actual = a.to_dict()[key]
            if isinstance(expected, dict):
                for name, tensor in expected.items():
                    torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    summary = json.loads(Path('/outputs/trained/summary.json').read_text())
    predictions = pd.read_csv('/outputs/trained_predict.csv')
    truth = pd.read_csv('/data/labels.csv')
    joined = predictions.merge(truth, on='complex_id', validate='one_to_one')
    mse = float(np.square(joined.y_pred - joined.y_true).mean())
    np.testing.assert_allclose(summary['best_score'], mse, rtol=1e-6, atol=1e-6)
    assert summary['training_ligand_conformer'] == 'crystal'
    assert summary['validation_ligand_conformer'] == 'rdkit'
    assert summary['validation_conformer_seed'] == 0
    Path('/outputs/parity.json').write_text(json.dumps({
        'complexes': 2, 'input_tensor_comparison': 'exact',
        'selected_val_mse': summary['best_score'], 'inference_mse': mse,
        'scope': 'one-epoch plumbing smoke, overlapping train/val fixtures; not scientific performance'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
