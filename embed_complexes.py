"""Extract complex embeddings from TankBind, with the affinity head removed.

TankBind never builds one fused protein-ligand graph. It keeps the two sides separate and
models the complex as their **interaction**: a tensor `z` of shape (batch, residues, atoms,
128), refined by five rounds of triangle attention. So "the complex embedding" is a choice,
and the analysis behind this one is in CLAUDE.md.

**We do not invent the pooling.** The published checkpoints use `readout_mode=2`, which is
`head -> pool` — per-pair energies summed to a scalar, leaving no vector. But the authors'
own `readout_mode == 1` branch (`model.py:119`) defines exactly the aggregation we want:

    valid_interaction_z = (z * z_mask).sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2))

a masked mean over the interaction tensor, feeding a `linear_energy` of shape (1, 128). That
is the authors' formula from a sibling branch, not our construction, and 128 is their own
choice of width. This is still recorded as `defined` rather than `native`, because the trained
head does not use it — but it is the most defensible aggregation available.

**p2rank is not needed here.** It exists to find pockets when you do not know where they are.
Every benchmark complex ships a bound ligand, so the pocket centre is the ligand centroid and
the whole segmentation step falls away.

usage:
    python embed_complexes.py --complexes /data --model /ckpt/self_dock.pt \
        --out /outputs/embeddings.npz --workdir /outputs/work
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tankbind"))

from data import TankBind_prediction  # noqa: E402
from feature_utils import (extract_torchdrug_feature_from_mol,  # noqa: E402
                           get_clean_res_list, get_protein_feature)
from model import get_model  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def read_ligand(complex_dir: str, cid: str):
    """PDBbind ships both; the sdf fails to sanitise for some complexes and the mol2 does not."""
    for name, reader in ((f"{cid}_ligand.mol2", Chem.MolFromMol2File),
                         (f"{cid}_ligand.sdf", None)):
        path = os.path.join(complex_dir, name)
        if not os.path.exists(path):
            continue
        mol = reader(path, sanitize=True) if reader else next(
            (m for m in Chem.SDMolSupplier(path, sanitize=True) if m is not None), None)
        if mol is not None:
            return mol
    raise ValueError("no readable ligand")


# RDKit 2023.09 added STEREOATROPCW and STEREOATROPCCW to BondStereo, taking the enum from
# six values to eight. torchdrug builds its stereo one-hot from that enum's length, so a newer
# RDKit silently widens every bond feature vector by two — 21 columns where these weights were
# trained on 19. The layout is bond type (4) + direction (7) + stereo (6 then, 8 now) +
# conjugated (1) + length (1), which puts the two new columns at indices 17 and 18.
#
# They describe atropisomerism, which none of these ligands has: across 2132 bonds in 40
# CASF-2016 complexes both columns are identically zero. Dropping them therefore restores the
# exact feature layout the weights were trained on rather than approximating it.
STEREO_ATROP_COLUMNS = (17, 18)
EXPECTED_EDGE_FEATURES = 19


def trim_bond_features(edge_feature: torch.Tensor) -> torch.Tensor:
    """Drop the atropisomer stereo columns a newer RDKit adds, if they are present."""
    if edge_feature.shape[1] == EXPECTED_EDGE_FEATURES:
        return edge_feature
    if edge_feature.shape[1] != EXPECTED_EDGE_FEATURES + len(STEREO_ATROP_COLUMNS):
        raise ValueError(
            f"bond features are {edge_feature.shape[1]}-dimensional; expected "
            f"{EXPECTED_EDGE_FEATURES} or {EXPECTED_EDGE_FEATURES + 2}. The RDKit stereo "
            f"vocabulary has changed again and the column layout must be rechecked."
        )
    extra = edge_feature[:, list(STEREO_ATROP_COLUMNS)]
    if extra.abs().sum() != 0:
        raise ValueError(
            "a ligand actually uses atropisomer stereo, so these columns carry information "
            "and cannot be dropped. This needs the weights retrained, not a workaround."
        )
    keep = [i for i in range(edge_feature.shape[1]) if i not in STEREO_ATROP_COLUMNS]
    return edge_feature[:, keep]


def pooled(z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
    """The authors' readout_mode 1 aggregation, model.py:119."""
    masked = z * z_mask.unsqueeze(-1)
    return masked.sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2)).unsqueeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed complexes with the head removed")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    parser_pdb = PDBParser(QUIET=True)

    ids = sorted(d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d)))

    protein_dict, compound_dict, rows, failed = {}, {}, [], []
    for cid in tqdm(ids, desc="Featurising"):
        d = os.path.join(args.complexes, cid)
        try:
            structure = parser_pdb.get_structure(cid, os.path.join(d, f"{cid}_protein.pdb"))
            res_list = get_clean_res_list(structure.get_residues(), verbose=False, ensure_ca_exist=True)
            protein_dict[cid] = get_protein_feature(res_list)

            mol = read_ligand(d, cid)
            coords, node_f, edge_list, edge_f, pair_dis = extract_torchdrug_feature_from_mol(
                mol, has_LAS_mask=True)
            compound_dict[cid] = (coords, node_f, edge_list,
                                  trim_bond_features(edge_f), pair_dis)

            # the pocket is known: it is where the bound ligand sits, so p2rank is unnecessary
            centre = mol.GetConformer().GetPositions().mean(axis=0)
            rows.append([cid, cid, f"{cid}_pocket", ",".join(f"{v:.3f}" for v in centre)])
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")

    if failed:
        print(f"\n{len(failed)} complexes could not be featurised:")
        for f in failed[:10]:
            print("  ", f)
    if not rows:
        raise SystemExit("nothing to embed")

    info = pd.DataFrame(rows, columns=["protein_name", "compound_name", "pocket_name", "pocket_com"])
    dataset = TankBind_prediction(args.workdir, data=info,
                                  protein_dict=protein_dict, compound_dict=compound_dict)

    # get_model logs its configuration, so it needs a real logger rather than None
    import logging as _logging
    model = get_model(0, _logging.getLogger("tankbind"), device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    # A hook on the transition module captures z after the last triangle round, which is
    # exactly what the readout would have consumed. Their forward() stays untouched, so the
    # same checkpoint keeps producing the same affinities.
    captured: list[torch.Tensor] = []
    handle = model.tranistion.register_forward_hook(lambda m, i, o: captured.append(o.detach()))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=["x", "compound_pair"])
    ids_out, vectors = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Embedding")):
            captured.clear()
            model(data.to(device))
            z = captured[-1]  # last transition = last refinement round
            mask = torch.ones(z.shape[:3], device=z.device)
            vectors.append(pooled(z, mask).cpu().numpy())
            ids_out.append(info.iloc[i]["protein_name"])
    handle.remove()

    matrix = np.concatenate(vectors, axis=0)
    np.savez(args.out, ids=np.array(ids_out), vectors=matrix)
    print(f"\n{len(ids_out)} embeddings of dimension {matrix.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
