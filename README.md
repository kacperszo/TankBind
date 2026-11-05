# TankBind — a complex modelled as an interaction, not a fused graph

## Upstream

A fork of the authors' TankBind repository (Wei Lu et al., Galixir Technologies; MIT licence, see
[LICENSE](LICENSE)), the code for the NeurIPS 2022 paper
[TANKBind: Trigonometry-Aware Neural NetworKs for Drug-Protein Binding Structure Prediction](https://biorxiv.org/cgi/content/short/2022.06.06.495043v1).
Commits up to `ff85f51` are the authors'; everything after is this fork's. The authors' README is
kept verbatim as [README.upstream.md](README.upstream.md); **its build and run instructions are not
current for this fork.** The published checkpoints are in `saved_models/`, as in the authors'
repository.

## What it is

The two sides stay separate and the complex is their **interaction**: a tensor of
shape (residues, atoms, 128) refined by five rounds of triangle attention. Protein features come from
GVP local frames, the ligand carries a pair-distance matrix. Two heads read the refined tensor — a
distance map and an affinity — so the trained model holds no complex vector, and the embedding uses
the authors' own `readout_mode == 1` formula, a masked mean.

Two images are built here. The reference one (`Containerfile`) is the original environment
(torch 1.13, python 3.8, torchdrug). The clean one (`Containerfile.gpu`) removes torchdrug and every
compiled PyG extension and runs torch 2.5.1.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.852**, RMSE 1.141, n=285 |
| CASF-2016 ranking | **rho 0.714** |
| embedding | 128d, the authors' formula, probe R 0.781 — 92% of its own head |
| agreement with stored reference embeddings | reference image 0.0; clean image 1.1e-02 embed (max abs difference), 0.0 predict, against a tolerance of 0.02 |
| ligand input | a fresh RDKit conformer, as in the authors' test protocol; the crystal pose gave R 0.821 |
| invariance | rotation and translation, to 1.6e-05 |

**The affinity is scored against a native pocket** — the ligand centroid, no p2rank — which is an
easier question than the published protocol asks.

## Build

```bash
podman build --format=docker -t tankbind:latest .                              # reference: torchdrug, py3.8
podman build --format=docker -f Containerfile.gpu -t tankbind-torch:latest .   # clean stack
```

## Run it

Every command below runs with `--network=none` and a read-only root filesystem. The weights
(`saved_models/self_dock.pt`) are in this repository. `finetune` additionally needs an
encoder-only file (the checkpoint with the two head prefixes listed in `transfer.py` removed).

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

**p2rank is not needed.** It exists to find pockets when you do not know where they are; every evaluation complex ships a bound ligand, so the pocket centre is the ligand centroid and the segmentation step falls away. Point this at apo structures and you will need p2rank back.

```bash
# reference image — localhost/tankbind:latest

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/saved_models:/ckpt:ro" \
    localhost/tankbind:latest \
    sh -c 'cd /tmp && python /work/embed_complexes.py --complexes /data --model /ckpt/self_dock.pt --out /outputs/embeddings.npz --workdir /outputs/work --device cpu'
```

```bash
# clean image — localhost/tankbind-torch:latest

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/saved_models:/ckpt:ro" \
    localhost/tankbind-torch:latest \
    sh -c 'cd /tmp && python /work/embed_complexes.py --complexes /data --model /ckpt/self_dock.pt --out /outputs/embeddings.npz --workdir /outputs/work --predict /outputs/predictions.csv --device cpu'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/saved_models:/ckpt:ro" \
    localhost/tankbind-torch:latest \
    sh -c 'cd /tmp && python /work/embed_complexes.py --complexes /data --model /ckpt/self_dock.pt --out /outputs/embeddings.npz --workdir /outputs/work --device cpu'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/saved_models:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/tankbind-torch:latest \
    sh -c 'cd /tmp && python /work/train_complexes.py --complexes /data --labels /splits/train.csv --out /outputs --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/saved_models:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/tankbind-torch:latest \
    sh -c 'cd /tmp && python /work/train_complexes.py --complexes /data --labels /splits/train.csv --out /outputs --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder'
```

## What comes out

| file | holds |
|---|---|
| `embeddings.npz` | `ids` and `vectors`, 128-dim — a masked mean over the interaction tensor, the authors' own `readout_mode == 1` formula |
| `predictions.csv` | `complex_id,y_pred` from the affinity head, with `--predict` |
| `work/` | the processed dataset PyG caches; reuse it between runs |

## Before you trust the numbers

**The clean tier is not bit-identical to the reference one, and the difference is measured.** `torch_cluster`'s `knn_graph` returns each node's neighbours in neither distance nor index order — a heap artefact with no rule to reproduce — so `tankbind/torch_ops.py` returns a canonical order instead and prefers the extension wherever it is installed. `compare_knn_graph.py` shows the edge *set* is identical on every backbone tested; only a handful of rows are ordered differently. Five rounds of triangle attention turn that into a median embedding difference of 4.5e-08 and a maximum of 1.1e-02, with CASF Pearson R unchanged to six decimal places.

Affinity comes from a native pocket here, which is an easier question than the published protocol asks.

## Training and validation

`train_complexes.py` trains the model on the authors' two objectives (contact distance map and
affinity) or fine-tunes it from a transferred encoder. Choices that differ from the authors' own
`main.py`, and why:

- **Validation uses a fresh RDKit conformer** (seed 0, independent of the training seed), the
  crystal pocket centre rounded to three decimals, batch size 1, and scores affinity MSE only.
  This makes checkpoint selection match what `embed_complexes.py` does at inference. It is a
  different protocol from the authors' crystal-ligand validation, not a defect in theirs.
- **Training still uses crystal ligands** and the original contact-plus-affinity objectives and
  warm-up. Crystal coordinates supply the distance targets: RDKit coordinates do not define a valid
  native distance target and can reorder atoms through SMILES, so RDKit distance maps are never
  treated as supervision.
- **Feature caches are keyed by content**: labels, input-file contents, conformer policy and seed.
  A split filename alone is not enough to identify a cached graph; the `v2` namespace is separate
  and older caches are never overwritten.
- **Failures raise instead of shrinking the split**: invalid or duplicate labels, preprocessing
  failures, non-finite objectives or gradients, and an empty validation set all stop the run.
- Native pockets remain privileged information; this is not p2rank or decoy-pocket training.

`test_training_contract.py` checks these properties inside the clean image.
`check_validation_parity.py` compares the validation inputs and MSE against a reloaded run of the
inference script on a two-complex fixture. They are plumbing checks, not performance measurements.

## Porting notes

**Where the embedding comes from.** TankBind keeps protein and ligand separate and models the
complex as their interaction: a tensor `z` of shape (batch, residues, atoms, 128), refined by five
rounds of triangle attention, so "the complex embedding" is a choice rather than a layer output. The
published checkpoints run `readout_mode=2`, which computes a per-pair energy and sums scalars and
leaves no vector to take. The authors' own `readout_mode == 1` branch (`model.py:119`) defines the
needed aggregation, a masked mean over the interaction tensor feeding a `linear_energy` of width 128:

```python
valid_interaction_z = (z * z_mask).sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2))
```

It comes from a branch the released weights do not take, so the trained head does not use it, but it
is the authors' formula and not one invented here.

**RDKit widened the bond features, not torchdrug.** The forward first died on `mat1 and mat2 shapes
cannot be multiplied (26x21 and 19x56)`: ligand bond features were 21-dimensional where the
published weights expect 19. Node features matched exactly, so it was bonds alone:

    bond type (4) + direction (7) + stereo (6 -> 8) + conjugated (1) + length (1)

RDKit 2023.09 added `STEREOATROPCW` and `STEREOATROPCCW` to the `BondStereo` enum, taking it from six
values to eight, and torchdrug derives its stereo one-hot from that enum's length. A newer RDKit
therefore adds two columns at indices 17 and 18 without saying anything. Both are identically zero
across 2132 bonds in 40 CASF-2016 complexes (atropisomerism does not occur in these ligands), so
dropping them restores the exact layout the weights were trained on. `trim_bond_features` refuses to
drop them if a ligand ever does use that stereo, because then the columns carry information and the
answer is retraining. The git history rules out the obvious suspect: the weights were uploaded
2022-05-25 with `conda install torchdrug` unpinned and `df47d7f` pinned 0.1.2 two weeks later, so the
pin was never the problem; RDKit moved underneath both. The clean image therefore pins
`rdkit=2024.03.5` (21-dimensional bond features before trimming, 19 after).

**The clean stack and its tolerance.** `torch_scatter`, `torch_sparse` and `torch_cluster` each build
against one torch ABI, so the wheel index was the torch pin; PyG dropped the hard dependency in 2.4.
`tankbind/torch_ops.py` replaces the four functions actually used (`scatter_mean/add/max` in
`gvp/__init__.py`, `GATv2.py`, `GINv2.py`, and `knn_graph` in `gvp/data.py`) and prefers the extension
wherever it is installed, so the reference image stays bit-for-bit. `scatter` is bit-identical.
`knn_graph` is not: `torch_cluster` returns each node's neighbours in neither distance nor index
order, a heap artefact with no rule to reproduce, so `torch_ops` returns a canonical order. The edge
set is identical on every backbone tested (`compare_knn_graph.py`); only a few rows are ordered
differently, and five rounds of triangle attention turn that into a median embedding difference of
4.5e-08, a maximum of 1.1e-02, and a CASF Pearson R unchanged at 0.820826 (that figure is for the
crystal-ligand input; see State). `test_invariance.py` checks what the rewrite could have broken
beyond the stored reference: rotating and translating a complex leaves the embedding alone to 1.6e-05
on vectors of norm 25-35.

```bash
python test_invariance.py --complexes /data --model /ckpt/self_dock.pt --n 2 --trials 2
python compare_knn_graph.py --complexes /data --check knn_reference.npz
```

**Environment facts.**

- `mkl<2024` is pinned on purpose: MKL 2024.1+ removes the symbol `iJIT_NotifyEvent`, which breaks
  `import torch` outright.
- `freesasa`, `h5py` and `python-lmdb` must come from conda, not pip: the slim image has no
  compiler, so pip builds them from source and fails. `atom3d` pulls freesasa.
- `atom3d` is a real dependency the authors' README omits (`from atom3d.datasets import LMDBDataset`);
  it installs via pip. The README lists `mlcrate` and `pyarrow`, which the code never imports.
- PyTorch resolves to 1.13.1 in the reference image, not the README's 1.12; the solver bumps it and
  the PyG builds are for 1.13.
- torchdrug compiles C++ extensions on import and creates its lock file at a relative path, so a
  read-only working directory fails before any model code runs. Run from a writable cwd (the
  commands above `cd /tmp`).
- `get_model` needs a real logger, not `None`; it logs its configuration on construction.
- micromamba images require `podman build --format=docker`, and `ARG MAMBA_DOCKERFILE_ACTIVATE=1`
  must precede any `RUN` that needs python.
- Weights load through `torch.load`, which is pickle and so code execution on deserialisation. The
  checkpoints here scan clean (only a plain state dict), but run them network-off regardless.
