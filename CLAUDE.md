# CLAUDE.md — TankBind

## What this is

A protein-ligand model whose published task is binding-structure prediction with an affinity
head. What is used here is the **encoder**: a complex-level embedding, with the affinity head
left off.

## Current state

Done for `embed`: 285 of 285 CASF-2016 core-set complexes, **probe R 0.762** against a
heavy-atom-count baseline of 0.470.

`predict` is not registered — the affinity path was not pursued.

Two tiers, and the modern one is now the default place to run this:

| variant | stack | check |
|---|---|---|
| `tankbind.reference` | torch 1.13.1, torchdrug, python 3.8, CPU only | the golden's source |
| `tankbind.torch` | torch 2.1.2+cu121, pyg 2.5.3, **no torchdrug** | `verify` 285/285, max|Δ| 2.9e-06 |

One image serves both processors: the cu121 build runs on the CPU when no device is passed,
so the CPU tier costs nothing beyond a registry line. The reference tier declares no `gpu`
image on purpose — torch 1.13's cu117 build has no sm_89 kernels, and the image that does run
on the card is the torchdrug-free port, not this code with a CUDA wheel. Pointing `--gpu` at
it would have swapped the implementation while the variant id stayed the same.

`verify` compares embeddings here, not predictions: the golden is
`reference/tankbind/casf2016_embeddings.npz`, and the harness compares per complex on the ids
they share. That is what makes an embed-only model checkable at all — before it, a rewrite of
the featuriser could have changed every vector with nothing to notice.

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

## The clean stack, and the one number it moved

Settled 2026-09-11. This tier ran torch 2.1.2 because `torch_scatter`, `torch_sparse` and
`torch_cluster` came from a per-torch wheel index — each links against one torch ABI, so the
index was the pin. `tankbind/torch_ops.py` replaces the four functions actually used and
**prefers the extension wherever it is installed**, so the reference tier stays bit-for-bit
while this one runs torch 2.5.1 with nothing compiled.

| what was used | where | replacement | cost |
|---|---|---|---|
| `scatter_mean/add/max` | `gvp/__init__.py`, `GATv2.py`, `GINv2.py` | `torch_geometric.utils.scatter` | none, bit-identical |
| `knn_graph` | `gvp/data.py` | dense `cdist` + stable `topk` | see below |
| `torch_sparse`, `torch_cluster` elsewhere | pulled in by PyG, never by this port | none needed | none |

`model.py` also imported `scatter_mean` and never used it.

**`knn_graph` is the one thing that could not be made identical, and the reason is worth
knowing.** `torch_cluster`'s version returns each node's neighbours in neither distance nor
index order — on 1bcu node 185 it puts a neighbour at distance 3.811748 *ahead* of one at
3.811731. That is a heap artefact, not a rule, so there is nothing to reproduce. It also drops
non-finite points from both the query and the candidate set, which matters because
`gvp/data.py` sets incomplete backbones to infinity first; missing that gave 1gpk 15871 edges
against 15840.

`compare_knn_graph.py` settles the rest: on all eight CASF backbones tested the **edge set is
identical** and only a handful of rows are ordered differently — 2 of 8250 on 1bcu, 8 of 31920
on 1e66. Float summation is not associative, and five rounds of triangle attention turn that
into:

| | |
|---|---|
| embeddings, median abs diff | 4.5e-08 |
| embeddings, p99 / max | 2.9e-05 / **1.1e-02** |
| affinity, median / max abs diff | 0 / 1.7e-03 |
| CASF Pearson R | **0.820826 either way** |

Isolated by changing one thing at a time: the same source on the old stack reproduces the same
residual, so it is not the torch bump; reverting `knn_graph` alone returns the pre-existing
2.9e-06. `tankbind.torch` declares `tolerance = 0.02` and the registry entry explains why.

`test_invariance.py` checks the property the rewrite could have broken rather than only the
golden: rotating and translating a complex leaves the embedding alone to **1.6e-05** on vectors
of norm 25-35. Every feature is a distance, a dihedral or a local frame, so invariance is the
model's own property — which is what makes it a check on the port.

```bash
python test_invariance.py --complexes /data --model /ckpt/self_dock.pt --n 2 --trials 2
python compare_knn_graph.py --complexes /data --check knn_reference.npz
```

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
