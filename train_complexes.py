"""Train TankBind on its two published objectives, or fine-tune it from a transferred encoder.

The authors' `tankbind/main.py` trains this model and stays the reference for what training
means here. What it wants is their preprocessed dataset — decoy pockets from p2rank, a sampled
subset per epoch, warm-up phases — and what the harness hands a trainer is a directory of
complexes and a csv of labels. So the *data* comes from `embed_complexes.py`'s featurisation
and the *loss* comes from theirs, unchanged:

    contact_loss  = MSE(y_pred, dis_map)                      main.py:197, pred_dis=1
    affinity_loss = relative_k * MSE(affinity_pred, affinity) main.py:215
    loss          = contact_loss + affinity_loss

with `relative_k` ramping exponentially to `--relative-k` over `--warm-up-epochs`, exactly as
`relative_k_mode=0` does. Their default `affinity_loss_mode=1` uses `my_affinity_criterion`,
which penalises decoy pockets asymmetrically — with `mask` all true it *is* MSE
(`utils.py:124`), and every pocket here is the native one, so the two coincide. Recorded in
`summary.json` rather than left as a silent equivalence.

**Every pocket is native, and that is a real difference from their training.** p2rank is not
run: each complex ships its bound ligand, so the pocket centre is the ligand centroid and
`use_compound_com` is true throughout. TankBind's affinity head was trained to rank the native
pocket above decoys, and nothing here presents it with a decoy. A model trained this way is
being asked an easier question than the published one.

usage:
    python train_complexes.py --complexes /data --labels /splits/train.csv \
        --val-labels /splits/val.csv --out /outputs --cache /cache --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tankbind"))

from data import TankBindDataSet  # noqa: E402
from feature_utils import (extract_torchdrug_feature_from_mol,  # noqa: E402
                           get_clean_res_list, get_protein_feature)
from model import get_model  # noqa: E402

from embed_complexes import read_ligand, trim_bond_features  # noqa: E402
from transfer import set_training_mode, transfer_encoder  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def featurise(complexes: str, labels: dict[str, float]):
    """Complexes to the two dictionaries and the index frame `TankBindDataSet` reads.

    The same featurisation `embed_complexes.py` uses, so a model trained here and a vector read
    there describe the same input. Divergence between the two would look like a modest,
    plausible loss of accuracy rather than a bug.
    """
    parser = PDBParser(QUIET=True)
    protein_dict, compound_dict, rows, failed = {}, {}, [], []

    for cid in tqdm(sorted(labels), desc="Featurising"):
        d = os.path.join(complexes, cid)
        if not os.path.isdir(d):
            failed.append(f"{cid}: no directory")
            continue
        try:
            structure = parser.get_structure(cid, os.path.join(d, f"{cid}_protein.pdb"))
            res_list = get_clean_res_list(structure.get_residues(), verbose=False,
                                          ensure_ca_exist=True)
            protein_dict[cid] = get_protein_feature(res_list)

            mol = read_ligand(d, cid)
            coords, node_f, edge_list, edge_f, pair_dis = extract_torchdrug_feature_from_mol(
                mol, has_LAS_mask=True)
            compound_dict[cid] = (coords, node_f, edge_list, trim_bond_features(edge_f), pair_dis)

            centre = mol.GetConformer().GetPositions().mean(axis=0)
            rows.append({"protein_name": cid, "compound_name": cid, "pdb": cid,
                         "affinity": labels[cid], "pocket_com": centre.reshape(1, 3),
                         "use_compound_com": True, "group": "train"})
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")

    if failed:
        print(f"{len(failed)} complexes failed to featurise:")
        for line in failed[:10]:
            print("  ", line)
    return pd.DataFrame(rows), protein_dict, compound_dict


def build_dataset(root: str, complexes: str, labels_path: str):
    """A `TankBindDataSet` under `root`, reusing what is already processed there.

    `TankBindDataSet` is a PyG `Dataset`, so it caches into `root/processed` on first use and
    reloads from there afterwards — the caching is theirs, and pointing `root` at /cache is all
    that is needed to keep it between runs.
    """
    with open(labels_path) as f:
        labels = {r["complex_id"]: float(r["y_true"]) for r in csv.DictReader(f)}

    processed = Path(root) / "processed" / "data.pt"
    if processed.exists():
        dataset = TankBindDataSet(root)
        print(f"reusing {processed}: {len(dataset)} complexes")
        return dataset

    info, protein_dict, compound_dict = featurise(complexes, labels)
    if info.empty:
        raise SystemExit(f"nothing featurised for {labels_path}")
    return TankBindDataSet(root, data=info, protein_dict=protein_dict,
                           compound_dict=compound_dict)


def run_epoch(model, loader, device, optimiser, relative_k, frozen) -> dict:
    """One pass, with the authors' two losses. `optimiser=None` means validation."""
    training = optimiser is not None
    set_training_mode(model, training, frozen_encoder=frozen)
    contact_criterion = nn.MSELoss()
    affinity_criterion = nn.MSELoss()

    total_contact, total_affinity, n = 0.0, 0.0, 0
    preds, trues = [], []

    for data in loader:
        data = data.to(device)
        with torch.set_grad_enabled(training):
            y_pred, affinity_pred = model(data)
            # real_y_mask is all-true here — every pocket is the native one — but the mask is
            # applied anyway so this line stays the authors' line rather than a simplification
            # that would quietly be wrong the moment decoys are introduced.
            mask = data.real_y_mask
            dis_map, y_masked = data.dis_map[mask], y_pred[mask]
            contact_loss = (contact_criterion(y_masked, dis_map) if len(dis_map)
                            else torch.zeros((), device=device))
            affinity_loss = relative_k * affinity_criterion(affinity_pred, data.affinity)
            objective = contact_loss + affinity_loss
        if training:
            optimiser.zero_grad()
            objective.backward()
            optimiser.step()

        batch_n = int(data.affinity.shape[0])
        total_contact += float(contact_loss) * batch_n
        total_affinity += float(affinity_criterion(affinity_pred, data.affinity)) * batch_n
        n += batch_n
        preds.append(affinity_pred.detach().cpu().numpy().ravel())
        trues.append(data.affinity.detach().cpu().numpy().ravel())

    p, t = np.concatenate(preds), np.concatenate(trues)
    r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and p.std() and t.std() else float("nan")
    return {"loss": total_affinity / max(n, 1), "contact": total_contact / max(n, 1),
            "r": r, "rmse": float(np.sqrt(((p - t) ** 2).mean()))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--labels", required=True, help="csv with complex_id,y_true")
    p.add_argument("--val-labels", dest="val_labels", default=None)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--workdir", default="/tmp/tankbind-train",
                   help="where datasets are processed when --cache is not given")
    p.add_argument("--cache", default=None)
    p.add_argument("--init-weights", dest="init_weights", default=None,
                   help="start from a published checkpoint rather than from scratch")
    p.add_argument("--init-encoder", dest="init_encoder", default=None,
                   help="encoder .pt from `gnnb encoder split`")
    p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8, help="the authors' own")
    p.add_argument("--lr", type=float, default=1e-4, help="the authors' own, main.py:140")
    p.add_argument("--relative-k", dest="relative_k", type=float, default=0.01,
                   help="weight on the affinity loss once warmed up; the authors' default")
    p.add_argument("--warm-up-epochs", dest="warm_up_epochs", type=int, default=15,
                   help="epochs over which relative_k reaches its full value, doubling each "
                        "epoch as relative_k_mode=0 does")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    root = args.cache or args.workdir
    train_ds = build_dataset(os.path.join(root, Path(args.labels).stem),
                             args.complexes, args.labels)
    val_ds = (build_dataset(os.path.join(root, Path(args.val_labels).stem),
                            args.complexes, args.val_labels) if args.val_labels else None)
    print(f"{len(train_ds)} training complexes"
          + (f", {len(val_ds)} validation" if val_ds else ""))

    follow = ["x", "compound_pair"]
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, follow_batch=follow)
    val_dl = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, follow_batch=follow)
              if val_ds else None)

    model = get_model(0, logging.getLogger("tankbind"), device)
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device,
                                         weights_only=True))
        print(f"started from {args.init_weights}")

    provenance: dict = {"init_encoder": None, "frozen_params": 0}
    if args.init_encoder:
        provenance = transfer_encoder(model, args.init_encoder, args.freeze_encoder)
    elif args.freeze_encoder:
        raise SystemExit("--freeze-encoder without --init-encoder would fit the heads against a "
                         "random representation; that is not an experiment")
    model.to(device)

    optimiser = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)

    out = Path(args.out)
    out_dir = out.parent if out.suffix == ".pt" else out
    model_path = out if out.suffix == ".pt" else out / "model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best, best_epoch = [], float("inf"), -1
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        # relative_k_mode=0: double each epoch until the base value is reached (main.py:203)
        relative_k = min(args.relative_k * (2 ** (epoch - 1)) / (2 ** args.warm_up_epochs),
                         args.relative_k)
        tr = run_epoch(model, train_dl, device, optimiser, relative_k, args.freeze_encoder)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_r": tr["r"],
               "val_loss": "", "val_r": "",
               "train_rmse": tr["rmse"], "train_contact": tr["contact"],
               "relative_k": relative_k, "val_rmse": ""}
        line = (f"epoch {epoch:4d}  k {relative_k:.2e}  train affinity mse {tr['loss']:7.4f}  "
                f"contact {tr['contact']:7.4f}  R {tr['r']:+.3f}")
        if val_dl:
            va = run_epoch(model, val_dl, device, None, relative_k, args.freeze_encoder)
            row.update(val_loss=va["loss"], val_r=va["r"], val_rmse=va["rmse"])
            line += f"  |  val mse {va['loss']:7.4f}  R {va['r']:+.3f}"
            score = va["loss"]
        else:
            score = tr["loss"]
        history.append(row)
        print(line, flush=True)

        if score < best:
            best, best_epoch = score, epoch
            # a plain state dict, so it reloads under weights_only=True — which is also what
            # embed_complexes.py expects
            torch.save(model.state_dict(), model_path)

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "tankbind",
        "objectives": "interaction distance map + affinity, the authors' two losses",
        "affinity_loss_mode": "0; their default 1 penalises decoy pockets asymmetrically and "
                              "reduces to MSE when every pocket is native, which it is here",
        "pockets": "native only — p2rank is not run, the pocket is the ligand centroid",
        "epochs": len(history), "best_epoch": best_epoch, "best_score": best,
        "selected_on": "val_affinity_mse" if val_dl else "train_affinity_mse",
        "train_complexes": len(train_ds), "val_complexes": len(val_ds) if val_ds else 0,
        "seed": args.seed, "freeze_encoder": args.freeze_encoder,
        "lr": args.lr, "batch_size": args.batch_size,
        "relative_k": args.relative_k, "warm_up_epochs": args.warm_up_epochs,
        "seconds": round(time.time() - started, 1),
        **provenance,
    }, indent=2) + "\n")

    print(f"\nbest {best:.4f} at epoch {best_epoch} -> {model_path}  "
          f"({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
