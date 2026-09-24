# TankBind — a complex modelled as an interaction, not a fused graph

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

The two sides stay separate and the complex is their **interaction**: a tensor of
shape (residues, atoms, 128) refined by five rounds of triangle attention. Protein features come from
GVP local frames, the ligand carries a pair-distance matrix. Two heads read the refined tensor — a
distance map and an affinity — so the trained model holds no complex vector, and the embedding uses
the authors' own `readout_mode == 1` formula, a masked mean.

`tankbind.reference` is the original environment (torch 1.13, python 3.8, torchdrug).
`tankbind.torch` removes torchdrug and every compiled PyG extension and runs torch 2.5.1.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.852**, RMSE 1.141, n=285 — highest in the roster |
| CASF-2016 ranking | **rho 0.714** |
| embedding | 128d, the authors' formula, probe R 0.781 — 92% of its own head |
| `gnnb verify` | reference 0.0; port 1.1e-02 embed, 0.0 predict (tolerance 0.02) |
| ligand input | a fresh RDKit conformer, as in the authors' test protocol; the crystal pose gave R 0.821 |
| invariance | rotation and translation, to 1.6e-05 |

**The affinity is scored against a native pocket** — the ligand centroid, no p2rank — which is an
easier question than the published protocol asks.

## Build

```bash
podman build --format=docker -t tankbind:latest .                              # reference: torchdrug, py3.8
podman build --format=docker -f Containerfile.gpu -t tankbind-torch:latest .   # clean stack
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

**p2rank is not needed.** It exists to find pockets when you do not know where they are; every benchmark complex ships a bound ligand, so the pocket centre is the ligand centroid and the segmentation step falls away. Point this at apo structures and you will need p2rank back.

```bash
# tankbind.reference — localhost/tankbind:latest
# source: models/tankbind

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
# tankbind.torch — localhost/tankbind-torch:latest
# source: models/tankbind

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

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
