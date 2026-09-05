"""torchdrug-free molecule featurisation for TankBind.

TankBind reaches torchdrug for one call, in `feature_utils.extract_torchdrug_feature_from_mol`:

    td.Molecule.from_smiles(Chem.MolToSmiles(mol), node_feature='property_prediction')

and uses four things off the result: `node_feature`, `edge_list`, `edge_weight`,
`edge_feature`. torchdrug contributes no chemistry there — both feature functions are RDKit
getters wrapped in one-hot encodings over fixed vocabularies. This module reproduces them.

Why bother: torchdrug 0.1.2 is unmaintained, declares `python_requires <3.9`, references
`torch._six` (removed in torch 2.0), and 0.2.1 is not a drop-in — it yields 18/67-dimensional
features where 0.1.2 yields 21/56. Keeping it pinned works but welds the whole stack in place.

Correctness is not argued, it is measured: `compare_featurizer.py` runs this against torchdrug
on real ligands and requires the arrays to match exactly.

The vocabularies below were read out of `torchdrug.data.feature` rather than reconstructed.
That matters — several plausible guesses at the element list give the right total width while
putting atoms in different columns, which no shape check would catch.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
# Imported explicitly, not reached through Chem: rdDepictor is a submodule that only
# appears as `Chem.rdDepictor` once something has imported it. torchdrug used to do
# that incidentally, so dropping torchdrug removed the import this depended on.
from rdkit.Chem import rdDepictor

# --- vocabularies, verbatim from torchdrug.data.feature -----------------------------------
ATOM_VOCAB = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S",
              "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]                    # 17
DEGREE_VOCAB = list(range(7))                                             # 0..6
NUM_HS_VOCAB = list(range(7))                                             # 0..6
TOTAL_VALENCE_VOCAB = list(range(8))                                      # 0..7
FORMAL_CHARGE_VOCAB = list(range(-5, 6))                                  # -5..5

BOND_TYPE_VOCAB = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                   Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]          # 4
BOND_DIR_VOCAB = list(range(7))                                           # rdkit BondDir ints
BOND_STEREO_VOCAB = list(range(8))                                        # rdkit BondStereo ints

# 18 + 8 + 8 + 9 + 12 + 1 = 56 ; 4 + 7 + 8 + 1 + 1 = 21
NODE_FEATURE_DIM = 56
EDGE_FEATURE_DIM = 21


def onehot(value, vocab: list, allow_unknown: bool = False) -> list[float]:
    """torchdrug's `onehot`, including the quirk that decides where unknowns land.

    With `allow_unknown` the vector is one longer than the vocabulary and an unrecognised
    value sets index **-1** — Python's negative index, i.e. the *last* slot. Reimplementing
    this as "append an unknown column and set it" happens to agree, but only because -1 and
    len(vocab) coincide there; writing it any other way silently shifts a column.
    """
    try:
        index = vocab.index(value)
    except ValueError:
        index = -1
    if allow_unknown:
        feature = [0.0] * (len(vocab) + 1)
    else:
        if index == -1:
            raise ValueError(f"unknown value {value!r} for vocabulary of {len(vocab)}")
        feature = [0.0] * len(vocab)
    feature[index] = 1.0
    return feature


def atom_features(atom: Chem.Atom) -> list[float]:
    """torchdrug's `atom_property_prediction`."""
    return (onehot(atom.GetSymbol(), ATOM_VOCAB, allow_unknown=True)
            + onehot(atom.GetDegree(), DEGREE_VOCAB, allow_unknown=True)
            + onehot(atom.GetTotalNumHs(), NUM_HS_VOCAB, allow_unknown=True)
            + onehot(atom.GetTotalValence(), TOTAL_VALENCE_VOCAB, allow_unknown=True)
            + onehot(atom.GetFormalCharge(), FORMAL_CHARGE_VOCAB, allow_unknown=True)
            + [float(atom.GetIsAromatic())])


def bond_length(bond: Chem.Bond) -> list[float]:
    """torchdrug's `bond_length`, quirk included.

    It measures the distance in the owning molecule's conformer, and *generates a 2-D
    depiction* when there is none. TankBind always hits that branch, because it rebuilds the
    molecule from SMILES and a SMILES molecule carries no conformer — so this feature is a
    distance in a generated drawing, not in the real structure, even though the model receives
    the true 3-D coordinates separately. Reproduced rather than fixed: the point here is to
    match the reference exactly.
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    h = conf.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conf.GetAtomPosition(bond.GetEndAtomIdx())
    return [float(h.Distance(t))]


def bond_features(bond: Chem.Bond) -> list[float]:
    """torchdrug's `bond_default`."""
    return (onehot(bond.GetBondType(), BOND_TYPE_VOCAB)
            + onehot(int(bond.GetBondDir()), BOND_DIR_VOCAB)
            + onehot(int(bond.GetStereo()), BOND_STEREO_VOCAB)
            + [float(bond.GetIsConjugated())]
            + bond_length(bond))


def molecule_features(smiles: str):
    """Replace `td.Molecule.from_smiles(smiles, node_feature='property_prediction')`.

    Returns (node_feature, edge_list, edge_weight, edge_feature) with torchdrug's shapes:
    edge_list is (E, 3) of [source, target, bond type index] and every bond contributes both
    directions, which is why E is twice the bond count.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")

    node_feature = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    edges, feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = BOND_TYPE_VOCAB.index(bond.GetBondType())
        f = bond_features(bond)
        # both directions share one feature vector; torchdrug emits them adjacently
        edges += [[i, j, btype], [j, i, btype]]
        feats += [f, f]

    edge_list = np.array(edges, dtype=np.int64).reshape(-1, 3)
    edge_feature = np.array(feats, dtype=np.float32).reshape(-1, EDGE_FEATURE_DIM)
    edge_weight = np.ones(len(edge_list), dtype=np.float32)
    return node_feature, edge_list, edge_weight, edge_feature
