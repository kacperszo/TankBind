"""Check a torchdrug-free featuriser against torchdrug's, array by array.

TankBind reaches torchdrug for exactly one thing on the embedding path:

    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol), node_feature='property_prediction')
    -> molstd.node_feature, molstd.edge_list, molstd.edge_weight, molstd.edge_feature

and torchdrug contributes no chemistry of its own there. Both feature functions are plain
RDKit getters wrapped in one-hot encodings over fixed vocabularies (`torchdrug/data/feature.py`,
`atom_property_prediction` and `bond_default`). That makes the dependency replaceable — and it
is worth replacing, because torchdrug 0.1.2 is unmaintained, declares `python_requires <3.9`,
and 0.2.1 is not a drop-in: it yields 18/67-dimensional features where 0.1.2 yields 21/56.

Replacing it silently is the danger. A wrong column ordering, or a mis-sized one-hot, produces
arrays of the right shape that the model happily consumes, and the embeddings come out
plausible and wrong. So the replacement is not trusted until it reproduces torchdrug's arrays
exactly, on many molecules, including the awkward ones.

Run inside the reference image, which has torchdrug 0.1.2:

    python compare_featurizer.py --sdf-dir /data
    python compare_featurizer.py --dump-vocab /outputs/vocab.json   # extract, don't guess
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def dump_vocab(path: str) -> int:
    """Write torchdrug's feature vocabularies out verbatim.

    These must be copied, never reconstructed from the feature widths: several plausible
    guesses at the element list produce the right total dimension while placing atoms in
    different columns.
    """
    from torchdrug.data import feature as F

    names = ["atom_vocab", "degree_vocab", "num_hs_vocab", "total_valence_vocab",
             "formal_charge_vocab", "chiral_tag_vocab", "hybridization_vocab",
             "bond_type_vocab", "bond_dir_vocab", "bond_stereo_vocab"]
    out = {}
    for n in names:
        v = getattr(F, n, None)
        if v is None:
            continue
        # vocabularies are lists or dicts of rdkit enum values; store something JSON-safe
        # while keeping order, since order *is* the column layout
        if isinstance(v, dict):
            out[n] = {"kind": "dict", "keys": [str(k) for k in v], "values": list(v.values())}
        else:
            out[n] = {"kind": "list", "values": [str(x) for x in v]}
        out[n]["len"] = len(v)
    Path(path).write_text(json.dumps(out, indent=2))
    for n, d in out.items():
        print(f"{n:<22} {d['len']:>3}  {str(d.get('values') or d.get('keys'))[:90]}")
    print(f"\n-> {path}")
    return 0


def torchdrug_features(mol):
    """The reference: exactly the call TankBind's feature_utils.py makes."""
    from rdkit import Chem
    from torchdrug import data as td

    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol), node_feature="property_prediction")
    return (np.asarray(molstd.node_feature),
            np.asarray(molstd.edge_list),
            np.asarray(molstd.edge_weight),
            np.asarray(molstd.edge_feature))


def replacement_features(mol):
    """The candidate, imported lazily so this script still runs before it exists."""
    from rdkit import Chem

    from tankbind_features import molecule_features  # noqa: E402

    return molecule_features(Chem.MolToSmiles(mol))


def compare(name: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"    {name:<16} SHAPE  reference {a.shape} vs replacement {b.shape}")
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        d = float(np.abs(a.astype(float) - b.astype(float)).max()) if a.size else 0.0
        # bond length rides on Compute2DCoords, so allow float noise but nothing structural
        same = d < 1e-5
        if not same:
            print(f"    {name:<16} DIFF   max|Δ|={d:.3e}")
        return same
    if not (a == b).all():
        bad = ([i for i in range(a.shape[1]) if not (a[:, i] == b[:, i]).all()]
               if a.ndim == 2 else [])
        print(f"    {name:<16} DIFF   {int((a != b).sum())} cells; columns: {bad}")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sdf-dir", default=None, help="directory of <id>/<id>_ligand.sdf")
    p.add_argument("--dump-vocab", default=None)
    p.add_argument("--limit", type=int, default=40)
    args = p.parse_args()

    if args.dump_vocab:
        return dump_vocab(args.dump_vocab)
    if not args.sdf_dir:
        raise SystemExit("pass --sdf-dir or --dump-vocab")

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Read ligands exactly as production does. This matters more than it looks: production
    # tries the mol2 first and only falls back to the sdf, and the two give different SMILES
    # on round-trip. Reading the sdf alone yields explicit-hydrogen SMILES that torchdrug's
    # parser rejects for 11 of 25 core-set ligands — which looks like a property of TankBind
    # and is really a property of how the comparison loaded the file.
    from embed_complexes import read_ligand

    complexes = sorted(d for d in Path(args.sdf_dir).iterdir() if d.is_dir())[: args.limit]
    if not complexes:
        raise SystemExit(f"no complexes under {args.sdf_dir}")

    ok = bad = skipped = 0
    rejected: list[str] = []
    for lig in complexes:
        try:
            mol = read_ligand(str(lig), lig.name)
        except Exception:
            skipped += 1
            continue
        try:
            ref = torchdrug_features(mol)
        except Exception as e:
            # torchdrug rejects some PDBbind ligands outright — 1a30's carboxylates round-trip
            # through SMILES as [C-](=O)=O, which its parser refuses. Production drops those
            # complexes too, so they are counted, not treated as a failure of the replacement.
            rejected.append(f"{lig.name}: {type(e).__name__}")
            continue
        try:
            got = replacement_features(mol)
        except ImportError as e:
            raise SystemExit(f"replacement not importable: {e}")
        except Exception as e:
            print(f"  {lig.name}: replacement raised {type(e).__name__}: {e}")
            bad += 1
            continue
        names = ["node_feature", "edge_list", "edge_weight", "edge_feature"]
        if all(compare(n, r, g) for n, r, g in zip(names, ref, got)):
            ok += 1
        else:
            print(f"  {lig.name}: DIVERGES")
            bad += 1

    print(f"\n{ok} identical, {bad} divergent, {skipped} unreadable, "
          f"{len(rejected)} rejected by torchdrug itself (of {len(complexes)} complexes)")
    if rejected:
        print(f"  torchdrug rejected: {', '.join(rejected[:6])}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
