<!-- gnn-benchmark:begin -->
# Running this in gnn-benchmark

What is used here is the **encoder** — a complex-level embedding, with the affinity head left
off. **Use `tankbind.torch`**: torchdrug is gone, which is what welded python 3.8 and torch
1.13 together.

| variant | stack | `gnnb verify` on CASF-2016 |
|---|---|---|
| `tankbind.reference` | torch 1.13.1, torchdrug, python 3.8, CPU only | the golden's source |
| `tankbind.torch` | torch 2.1.2+cu121, PyG 2.5.3, **no torchdrug** | 285/285, max abs diff 2.9e-06 |

One image serves both processors: the cu121 build runs on the CPU when no device is passed.
The reference tier declares no GPU image on purpose — torch 1.13's cu117 build has no sm_89
kernels, and the image that does run on the card is this port, not that code with a CUDA wheel.

`verify` compares embeddings here rather than predictions, against
`reference/tankbind/casf2016_embeddings.npz`.

```bash
podman build --format=docker -f Containerfile.gpu -t tankbind-torch:latest .

gnnb verify --variant tankbind.torch --dataset data/CASF-2016/coreset
gnnb run --variant tankbind.torch --capability embed --dataset <complexes> --gpu
```

TankBind keeps protein and ligand separate and models the complex as their interaction, so
"the complex embedding" is a choice rather than a layer output. The one used is the authors'
own `readout_mode == 1` formula. Why, and what the trained `readout_mode=2` leaves behind, is
in [CLAUDE.md](CLAUDE.md).

## Running it without the harness

This fork runs on its own; the benchmark adds bookkeeping, not capability. Every
command below is generated from the adapter by `gnnb howto`, so it cannot drift from
what the harness actually runs — regenerate with `python tools/sync_model_readmes.py`.

All of them run with `--network=none` and a read-only root filesystem. Nothing is
fetched at run time; dependencies are resolved when the image is built.

### What it eats

One directory per complex, named after it:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

**p2rank is not needed.** It exists to find pockets when you do not know where they are; every benchmark complex ships a bound ligand, so the pocket centre is the ligand centroid and the segmentation step falls away. Point this at apo structures and you will need p2rank back.

### Build

```bash
podman build --format=docker -t tankbind:latest .                              # reference: torchdrug, py3.8
podman build --format=docker -f Containerfile.gpu -t tankbind-torch:latest .   # clean stack
```

### Run

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

### What comes out

| file | holds |
|---|---|
| `embeddings.npz` | `ids` and `vectors`, 128-dim — a masked mean over the interaction tensor, the authors' own `readout_mode == 1` formula |
| `predictions.csv` | `complex_id,y_pred` from the affinity head, with `--predict` |
| `work/` | the processed dataset PyG caches; reuse it between runs |

### Before you trust the numbers

**The clean tier is not bit-identical to the reference one, and the difference is measured.** `torch_cluster`'s `knn_graph` returns each node's neighbours in neither distance nor index order — a heap artefact with no rule to reproduce — so `tankbind/torch_ops.py` returns a canonical order instead and prefers the extension wherever it is installed. `compare_knn_graph.py` shows the edge *set* is identical on every backbone tested; only a handful of rows are ordered differently. Five rounds of triangle attention turn that into a median embedding difference of 4.5e-08 and a maximum of 1.1e-02, with CASF Pearson R unchanged to six decimal places.

Affinity comes from a native pocket here, which is an easier question than the published protocol asks.

<!-- gnn-benchmark:end -->

---

![header](imgs/header.png)

Source code for the *NeurIPS 2022* paper [TANKBind: Trigonometry-Aware Neural NetworKs for Drug-Protein Binding Structure Prediction](https://biorxiv.org/cgi/content/short/2022.06.06.495043v1)

# TankBind
TankBind could predict both the protein-ligand binding structure and their affinity.

The primary purpose of this repository is to enable the reproduction of the results reported in the paper, as well as to facilitate the work of others who wish to build upon it.
To experience the latest version, which includes various improvements made to the model, simply create an account at https://m1.galixir.com/public/login_en/index.html.

If you have any question or suggestion, please feel free to open an issue or email me at [wei.lu@galixir.com](wei.lu@galixir.com) or shuangjia zheng at [shuangjia.zheng@galixir.com](shuangjia.zheng@galixir.com).

## Installation
````
conda create -n tankbind_py38 python=3.8
conda activate tankbind_py38
````
You might want to change the cudatoolkit version based on the GPU you are using.:
````
conda install pytorch cudatoolkit=11.3 -c pytorch
````

````
conda install torchdrug=0.1.2 pyg=2.1.0 biopython nglview jupyterlab -c milagraph -c conda-forge -c pytorch -c pyg
pip install torchmetrics tqdm mlcrate pyarrow
rdkit version used: 2021.03.4
````

p2rank v2.3 could be downloaded from here:

https://github.com/rdk/p2rank/releases/download/2.3/p2rank_2.3.tar.gz


## Test set evaluation
We include the script for reproducing the self-dock result in

    examples/testset_evaluation_cleaned.ipynb

The test_dataset is constructed using the notebook in "Dataset construction" section.


## Prediction
We use the prediction of the structure of protein ABL1 in complex with two drugs, Imatinib and compound6 (PDB: 6HD6) as an example for predicting the drug-protein binding structure. 

    examples/prediction_example_using_PDB_6hd6.ipynb

<img src="imgs/example_6hd6.png" width="200">


## Dataset construction
Scripts for training/test dataset construction is provided in:

    examples/construction_PDBbind_training_and_test_dataset.ipynb.ipynb

The Script I used to train the model is 

    python main.py -d 0 -m 0 --batch_size 5 --label baseline --addNoise 5 --use_equivalent_native_y_mask


## High-throughput virtual screening
TankBind also support virtual screening. In our example here, for the WDR domain of LRRK2 protein, we can screen 10,000 drug candidates in 2 minutes (or 1M in around 3 hours) with a single GPU. Check out

    examples/high_throughput_virtual_screening_LRRK2_WDR.ipynb


## Citation
    @article{lu2022tankbind,
    	title={Tankbind: Trigonometry-aware neural networks for drug-protein binding structure prediction},
    	author={Lu, Wei and Wu, Qifeng and Zhang, Jixian and Rao, Jiahua and Li, Chengtao and Zheng, Shuangjia},
    	journal={Advances in Neural Information Processing Systems},
    	year={2022}
    }
