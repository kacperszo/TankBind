# Harness training repair — 2026-10-20

Validation now uses fresh RDKit ligands (seed 0 by default), the crystal pocket
centre rounded as in inference, and affinity MSE alone. Training still uses
crystal ligands and the original contact-plus-affinity objectives and warm-up.
This aligns checkpoint selection with the harness's deployed inference protocol;
it does not claim the authors' validation inputs were wrong. The authors' local
get_data path uses crystal validation features, and their external test uses RDKit.

A versioned cache namespace hashes source contents, labels and conformer policy.
Old cache directories remain intact. Invalid/duplicate labels and preprocessing
failures raise instead of silently reducing the split. Nonfinite objectives and
gradients fail before an optimiser update. Empty validation cannot select a model.

Run test_training_contract.py inside the modern image. check_validation_parity.py
is an integration assertion for the harness two-complex train/infer fixture; it
compares every inference tensor exactly and validation MSE against reloaded CLI
predictions. These are plumbing checks, not scientific performance measurements.
