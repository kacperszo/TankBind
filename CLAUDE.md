# CLAUDE.md — TankBind

## What this is

A protein-ligand model whose published task is binding-structure prediction with an affinity
head. What is used here is the **encoder**: a complex-level embedding, with the affinity head
left off.

## Current state

Done for `embed`: 285 of 285 CASF-2016 core-set complexes, **probe R 0.762** against a
heavy-atom-count baseline of 0.470.

`predict` is not registered — the affinity path was not pursued.

Environment: `localhost/tankbind:latest`, micromamba base, CPU-only. `environment.yml` builds
from scratch and every import the code makes at module level works inside the image.

## Where the embedding comes from, and why it is not ours

TankBind does not build one fused protein-ligand graph. It keeps the two sides separate and
models the complex as their **interaction**: a tensor `z` of shape (batch, residues, atoms, 128),
refined by five rounds of triangle attention. So "the complex embedding" is a choice rather than
a layer output.

**The published checkpoints run `readout_mode=2`**, which computes a per-pair energy and sums
the scalars — `head → pool`, leaving no vector to take. But the authors' own `readout_mode == 1`
branch (`model.py:119`) defines exactly the aggregation needed:

```python
valid_interaction_z = (z * z_mask).sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2))
```

a masked mean over the interaction tensor, feeding a `linear_energy` of width 128. That formula
is theirs, from a sibling branch the released weights do not take, and `git blame` puts every
line of it in `18314e96` — the upload commit that also carries the weights, in a file with
exactly one commit in its history.

Recorded as `defined` rather than `native`, because the trained head does not use it. But it is
the authors' formula, not one invented here, and that distinction has mattered: poolings chosen
by us land eight to ten points lower on the same probe.

## p2rank is not needed

It exists to find pockets when their location is unknown. Every benchmark complex ships a bound
ligand, so the pocket centre is the ligand centroid and the whole segmentation step falls away.

## RDKit widened the bond features, not torchdrug

The forward first died on `mat1 and mat2 shapes cannot be multiplied (26x21 and 19x56)` — ligand
bond features 21-dimensional where the published weights expect 19. Node features matched
exactly, so it was bonds alone.

    bond type (4) + direction (7) + stereo (6 -> 8) + conjugated (1) + length (1)

RDKit 2023.09 added `STEREOATROPCW` and `STEREOATROPCCW` to the `BondStereo` enum, taking it from
six values to eight, and torchdrug derives its stereo one-hot from that enum's length. A newer
RDKit therefore adds two columns at indices 17 and 18 without saying anything.

**Both are identically zero** across 2132 bonds in 40 CASF-2016 complexes — atropisomerism does
not occur in these ligands — so dropping them restores the exact layout the weights were trained
on rather than approximating it. `trim_bond_features` refuses to drop them if a ligand ever does
use that stereo, because then the columns carry information and the answer is retraining.

The git history rules out the obvious suspect: weights uploaded 2022-05-25 with
`conda install torchdrug` unpinned, and `df47d7f` pinned 0.1.2 two weeks later. The pin was never
the problem — RDKit moved underneath both.

## Hard-won environment facts (do NOT regress these)

- **`mkl<2024` is pinned on purpose.** MKL 2024.1+ removes the symbol `iJIT_NotifyEvent`, which
  breaks `import torch` outright. Do not loosen it.
- **`freesasa`, `h5py` and `python-lmdb` must come from conda, not pip.** The slim image has no
  compiler, so pip builds them from source and fails. atom3d pulls freesasa.
- **`atom3d` is a real dependency the README omits** — the code does
  `from atom3d.datasets import LMDBDataset`. It installs via pip.
- **The README lists `mlcrate` and `pyarrow`; the code imports neither.** Left out deliberately.
- PyTorch resolves to **1.13.1**, not the README's 1.12 — the solver bumps it, and the pyg builds
  are for 1.13, so it is consistent.
- **torchdrug compiles C++ extensions on import** and creates its lock file at a *relative* path,
  so a read-only working directory fails before any of our code runs. Run from a writable cwd.
- **`get_model` needs a real logger**, not `None` — it logs its configuration on construction.
- micromamba requires `--format=docker` under Podman, and `ARG MAMBA_DOCKERFILE_ACTIVATE=1` must
  precede any `RUN` that needs python.

## Build & run

```bash
podman build --format=docker -t tankbind:latest .
gnnb run --variant tankbind.reference --capability embed --dataset <complexes>
```

Weights load through `torch.load`, which is pickle and therefore code execution on
deserialisation. They come from this repository rather than a runtime download, and they scan
clean — but the run is still network-off, read-only, with no host home mounted.
