"""TankBind's embedding depends on geometry, so moving the complex must not change it.

Every quantity the featuriser computes is a distance, a dihedral or a local frame: GVP protein
features are built from backbone orientations, the ligand carries a pair-distance matrix, and
the interaction tensor is refined from both. All of that is invariant under rotation and
translation, and under relabelling the residues — which is a property of the *model*, and
therefore a useful check on the *port*. If a coordinate leaked into a feature directly, or an
axis got transposed while every shape stayed right, the embedding would move here and nowhere
else.

This matters more for TankBind than for its neighbours. Removing `torch_cluster` meant
reimplementing `knn_graph`, and that rewrite is the one change in the roster that moved a number
against the golden file. A bit-identical golden would have settled the question on its own; a
moved one cannot, so the property gets checked directly.

Expect small deviations rather than zero, and the reason is worth knowing: the pocket is selected
by a distance cutoff and the neighbour graph by a `topk`, so a residue sitting exactly on either
boundary can fall the other way once the coordinates have been through a rotation matrix. That is
the featuriser being discontinuous at a threshold, not the model being non-invariant.

usage, inside the image:
    python test_invariance.py --complexes /data --model /ckpt/self_dock.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import RDLogger
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tankbind"))

from data import TankBind_prediction  # noqa: E402
from feature_utils import (extract_torchdrug_feature_from_mol,  # noqa: E402
                           get_clean_res_list, get_protein_feature)
from model import get_model  # noqa: E402

from embed_complexes import pooled, read_ligand, trim_bond_features  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def embed(model, workdir: str, cid: str, res_list, mol, device: str) -> np.ndarray:
    """One complex through the model, pooled exactly as embed_complexes.py pools it."""
    protein_dict = {cid: get_protein_feature(res_list)}
    coords, node_f, edge_list, edge_f, pair_dis = extract_torchdrug_feature_from_mol(
        mol, has_LAS_mask=True)
    compound_dict = {cid: (coords, node_f, edge_list, trim_bond_features(edge_f), pair_dis)}
    centre = mol.GetConformer().GetPositions().mean(axis=0)
    info = pd.DataFrame([[cid, cid, f"{cid}_pocket", ",".join(f"{v:.3f}" for v in centre)]],
                        columns=["protein_name", "compound_name", "pocket_name", "pocket_com"])

    dataset = TankBind_prediction(workdir, data=info, protein_dict=protein_dict,
                                 compound_dict=compound_dict)
    captured: list[torch.Tensor] = []
    handle = model.tranistion.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=["x", "compound_pair"])
    with torch.no_grad():
        for data in loader:
            model(data.to(device))
    handle.remove()
    z = captured[-1]
    return pooled(z, torch.ones(z.shape[:3], device=z.device)).cpu().numpy().ravel()


def moved(res_list, mol, rotation, translation):
    """The same complex somewhere else. Both sides move together or the pocket changes."""
    import copy

    res_list = copy.deepcopy(res_list)
    for res in res_list:
        for atom in res:
            atom.set_coord((rotation @ atom.coord) + translation)
    mol = copy.deepcopy(mol)
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = np.asarray(conformer.GetAtomPosition(i))
        conformer.SetAtomPosition(i, (rotation @ p) + translation)
    return res_list, mol


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=2, help="how many complexes")
    p.add_argument("--trials", type=int, default=2, help="random placements each")
    p.add_argument("--tolerance", type=float, default=1e-2,
                   help="the pocket cutoff and the knn topk are both discontinuous at a "
                        "threshold, so a residue on the boundary can fall either way")
    p.add_argument("--workdir", default="/tmp/tankbind-invariance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    model = get_model(0, logging.getLogger("tankbind"), args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=True))
    model.eval()

    parser = PDBParser(QUIET=True)
    ids = sorted(d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d)))[: args.n]

    worst, failures, run = 0.0, 0, 0
    for cid in ids:
        directory = os.path.join(args.complexes, cid)
        structure = parser.get_structure(cid, os.path.join(directory, f"{cid}_protein.pdb"))
        res_list = get_clean_res_list(structure.get_residues(), verbose=False, ensure_ca_exist=True)
        mol = read_ligand(directory, cid)

        base = embed(model, f"{args.workdir}/{cid}-base-{run}", cid, res_list, mol, args.device)
        run += 1
        print(f"  {cid}: |embedding| = {np.linalg.norm(base):.6f}")

        for trial in range(args.trials):
            rotation = Rotation.random(random_state=rng.randint(2 ** 31)).as_matrix()
            translation = rng.randn(3) * 50.0
            shifted_res, shifted_mol = moved(res_list, mol, rotation, translation)
            other = embed(model, f"{args.workdir}/{cid}-{trial}-{run}", cid,
                          shifted_res, shifted_mol, args.device)
            run += 1
            deviation = float(np.abs(base - other).max())
            worst = max(worst, deviation)
            ok = deviation <= args.tolerance
            failures += 0 if ok else 1
            print(f"      rotation+translation {trial}: max|Δ| {deviation:.2e}"
                  f"  {'ok' if ok else 'FAILED'}")

    print(f"\nworst deviation {worst:.2e} (tolerance {args.tolerance:g})")
    if failures:
        print(f"{failures} placement(s) moved the embedding — the featuriser is reading "
              f"coordinates somewhere it should be reading distances")
        return 1
    print("invariant under rotation and translation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
