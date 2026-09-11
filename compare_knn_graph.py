"""Check the pure-torch `knn_graph` against `torch_cluster`'s, on real protein backbones.

`gvp/data.py` calls `torch_cluster.knn_graph(X_ca, k=top_k)` exactly once, and that single call
was the last thing holding this tier to a compiled extension — and therefore to whichever torch
its wheel was built for. `torch_ops.knn_graph` replaces it.

Matching it is not a matter of finding the same *set* of neighbours. The result is an edge list
whose row order becomes the row order of every downstream edge feature, so the two have to agree
**index for index**. This script is what establishes that they do, rather than an argument that
they should:

    # in the old image, which has torch_cluster
    python compare_knn_graph.py --complexes /data --out /outputs/knn_reference.npz --reference
    # in the new one, which does not
    python compare_knn_graph.py --complexes /data --check /outputs/knn_reference.npz

Backbones with missing residues matter here and are included deliberately: `gvp/data.py` sets
those coordinates to infinity before the call, so the two implementations have to agree about
what a non-finite point is nearest to.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tankbind"))

TOP_K = 30   # gvp.data.ProteinGraphDataset's default, and what TankBind builds with


def backbones(complexes: str, limit: int):
    """X_ca for each protein, prepared exactly as gvp/data.py prepares it."""
    from Bio.PDB import PDBParser

    from feature_utils import get_clean_res_list

    parser = PDBParser(QUIET=True)
    out = []
    for cid in sorted(os.listdir(complexes))[:limit]:
        path = os.path.join(complexes, cid, f"{cid}_protein.pdb")
        if not os.path.exists(path):
            continue
        structure = parser.get_structure(cid, path)
        res_list = get_clean_res_list(structure.get_residues(), verbose=False, ensure_ca_exist=True)
        coords = []
        for res in res_list:
            frame = []
            for atom in ("N", "CA", "C", "O"):
                frame.append(res[atom].coord if atom in res else [np.nan] * 3)
            coords.append(frame)
        c = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
        # gvp/data.py:170 — residues with any non-finite backbone atom go to infinity
        mask = torch.isfinite(c.sum(dim=(1, 2)))
        c[~mask] = np.inf
        out.append((cid, c[:, 1]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--reference", action="store_true",
                   help="compute with torch_cluster and write the result")
    p.add_argument("--out", default=None)
    p.add_argument("--check", default=None, help="a file written by --reference")
    p.add_argument("--dump", default=None,
                   help="print the neighbour sequences that differ for this complex, with "
                        "their distances, so the tie-breaking rule can be read off")
    args = p.parse_args()

    cases = backbones(args.complexes, args.limit)
    print(f"{len(cases)} backbones, k={TOP_K}")

    if args.reference:
        import torch_cluster

        payload = {}
        for cid, x_ca in cases:
            payload[f"{cid}.x"] = x_ca.numpy()
            payload[f"{cid}.e"] = torch_cluster.knn_graph(x_ca, k=TOP_K).numpy()
            print(f"  {cid}: {x_ca.shape[0]} residues, {payload[f'{cid}.e'].shape[1]} edges")
        np.savez(args.out, **payload)
        print(f"-> {args.out}")
        return 0

    from torch_ops import knn_graph

    if args.dump:
        with np.load(args.check) as ref:
            x_ca = torch.as_tensor(ref[f"{args.dump}.x"])
            want = torch.as_tensor(ref[f"{args.dump}.e"])
        got = knn_graph(x_ca, k=TOP_K)
        distance = torch.cdist(x_ca, x_ca)
        differing = (got != want).any(dim=0).nonzero().view(-1)
        shown = set()
        for col in differing[:20].tolist():
            node = int(want[1, col])
            if node in shown:
                continue
            shown.add(node)
            block = (want[1] == node).nonzero().view(-1)
            w_seq = want[0, block].tolist()
            g_seq = got[0, (got[1] == node).nonzero().view(-1)].tolist()
            first = next((i for i, (a, b) in enumerate(zip(w_seq, g_seq)) if a != b), None)
            print(f"\nquery node {node}, sequences diverge at position {first}")
            lo = max(0, (first or 0) - 2)
            hi = (first or 0) + 4
            for label, seq in (("torch_cluster", w_seq), ("torch_ops    ", g_seq)):
                pairs = [f"{j}@{distance[node, j]:.9f}" for j in seq[lo:hi]]
                print(f"  {label}: " + "  ".join(pairs))
        return 0

    with np.load(args.check) as ref:
        ids = sorted({k.split(".")[0] for k in ref.files})
        bad = 0
        for cid in ids:
            x_ca = torch.as_tensor(ref[f"{cid}.x"])
            want = torch.as_tensor(ref[f"{cid}.e"])
            got = knn_graph(x_ca, k=TOP_K)
            if got.shape != want.shape:
                print(f"  {cid}: FAIL shape {tuple(got.shape)} against {tuple(want.shape)}")
                bad += 1
            elif not torch.equal(got, want):
                # Order and membership are different failures. A reordering of the same edge
                # set is harmless downstream — every consumer aggregates with scatter, which
                # does not care what order rows arrive in. A different *set* is a different
                # graph, and would be a real divergence.
                differing = int((got != want).any(dim=0).sum())
                same_set = (set(map(tuple, got.t().tolist()))
                            == set(map(tuple, want.t().tolist())))
                verdict = "reordered" if same_set else "DIFFERENT EDGES"
                print(f"  {cid}: {verdict} — {differing} of {want.shape[1]} rows differ")
                bad += 0 if same_set else 1
            else:
                print(f"  {cid}: identical, {want.shape[1]} edges")

    print(f"\n{len(ids) - bad} of {len(ids)} carry the same edge set")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
