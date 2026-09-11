"""The compiled-extension operations this port uses, reimplemented on torch and PyG.

Three functions came from `torch_scatter` and one from `torch_cluster`. Both are extensions
built against a single torch ABI, so requiring them pinned this tier to whichever torch their
wheels were published for — which is what held it at 2.1.2 while the rest of the benchmark
moved to 2.5.1.

## The scatter operations

`torch_geometric.utils.scatter` covers everything used here; PyG dropped the hard dependency on
the extensions in 2.4.

Two differences between the two APIs, both of which would be silent:

* **`torch_scatter.scatter` defaults to `dim=-1`, PyG's to `dim=0`.** Every call site in this
  port passes `dim=` explicitly, which is why they could be left alone — but a future one
  written from memory of the other library would scatter along the wrong axis and still run.
* **`scatter_max` returns `(values, argmax)`; PyG's returns values only.** `GATv2.py:99` takes
  `[0]`, so the tuple shape is reproduced here rather than the call site being rewritten.
  Nothing in this port reads the indices, so the second element is None rather than a
  recomputed argmax that would cost a second pass to produce and never be looked at.

Empty groups differ too — `torch_scatter.scatter_max` fills them with the dtype minimum while
PyG's leaves them at zero — and it does not matter here: the only `scatter_max` result is
immediately indexed by `node_out`, so a group with no members is never read.

## knn_graph

`gvp/data.py` calls `torch_cluster.knn_graph(X_ca, k=top_k)` once, on a protein backbone. The
replacement below is a dense pairwise distance and a `topk`, which is exact and, at a few
hundred to a few thousand residues, cheaper than the index the extension builds.

Matching it means matching the **edge order**, not just the neighbour set: the returned list
becomes the row order of every downstream edge feature. `compare_knn_graph.py` is what
establishes that on real backbones, against the extension itself, rather than by argument.
"""

from __future__ import annotations

import torch

# **The extension wins where it is installed.** One source tree serves both tiers, and the
# reference tier is pinned to torch 1.13 with torch_scatter and torch_cluster present *on
# purpose* — it is the fidelity baseline the torch tier is checked against. Preferring the
# extension there keeps it bit-for-bit what it was, and the replacements below take effect only
# in the image that has no extensions to fall back on.
#
# This is the same shape as IGN's source serving three tiers through fallbacks: pick the
# behaviour per environment, not per source tree.
try:
    from torch_scatter import scatter_add as _ext_scatter_add
    from torch_scatter import scatter_max as _ext_scatter_max
    from torch_scatter import scatter_mean as _ext_scatter_mean

    HAVE_TORCH_SCATTER = True
except ImportError:
    from torch_geometric.utils import scatter as _scatter

    HAVE_TORCH_SCATTER = False


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None) -> torch.Tensor:
    if HAVE_TORCH_SCATTER:
        return _ext_scatter_add(src, index, dim=dim, dim_size=dim_size)
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: int | None = None) -> torch.Tensor:
    if HAVE_TORCH_SCATTER:
        return _ext_scatter_mean(src, index, dim=dim, dim_size=dim_size)
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None):
    """Values and argmax, as torch_scatter returns them — the indices are None on the PyG path.

    Returning the pair keeps `scatter_max(...)[0]` working at the call site. If something ever
    needs `[1]`, it will get None and fail immediately rather than compute a wrong answer.
    """
    if HAVE_TORCH_SCATTER:
        return _ext_scatter_max(src, index, dim=dim, dim_size=dim_size)
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="max"), None


try:
    from torch_cluster import knn_graph as _ext_knn_graph

    HAVE_TORCH_CLUSTER = True
except ImportError:
    HAVE_TORCH_CLUSTER = False


def knn_graph(x: torch.Tensor, k: int, loop: bool = False,
              flow: str = "source_to_target") -> torch.Tensor:
    """`torch_cluster.knn_graph`, in plain torch where the extension is absent.

    **This one is not edge-for-edge identical, and the difference is measured rather than
    assumed.** `torch_cluster`'s knn returns each node's neighbours in neither distance nor
    index order — for 1bcu node 185 it puts 184 at distance 3.811748 ahead of 186 at 3.811731 —
    which is a heap artefact rather than a rule. Reproducing it would mean reimplementing that
    heap. So this returns a canonical order instead: ascending distance, ties by index.

    `compare_knn_graph.py` establishes what that costs. On all eight CASF backbones tested the
    **edge set is identical**; only a handful of rows are ordered differently (2 of 8250 on
    1bcu, 8 of 31920 on 1e66). Everything downstream aggregates with scatter, so the graph is
    the same graph — but float summation is not associative, and five rounds of triangle
    attention amplify the reordering into a median embedding difference of 4.5e-08, a 99th
    percentile of 2.9e-05 and a maximum of 1.1e-02 on values whose own median magnitude is 1.6.
    Affinity moves by a median of exactly zero and at most 1.7e-03, and the CASF Pearson R is
    unchanged to six decimal places at 0.820826.

    That residual is why `tankbind.torch` declares a tolerance in the registry, and why the
    extension is preferred where it exists: the reference tier stays exact.

    Two details had to be measured rather than assumed, and `compare_knn_graph.py` is what
    measured them against the extension on real backbones:

    * **Non-finite points get no edges at all, and are never anyone's neighbour.**
      `gvp/data.py` sets residues with an incomplete backbone to infinity before calling this,
      and the extension drops them from the query *and* the candidate set. A naive version
      gives them `k + 1` edges instead of zero, because their self-distance is NaN rather than
      zero so the self match is not removed — 1gpk came back with 15871 edges against 15840.
    * **Ties break by index.** Equidistant neighbours do occur — 2 edges in 1bcu, 12 in 1e66 —
      and `torch.topk` does not promise a stable order. A stable full sort does, and at a few
      thousand residues it costs nothing worth measuring.

    Returns `[neighbour, query]` for `flow='source_to_target'`, the convention `gvp/data.py`
    reads: `X_ca[edge_index[0]] - X_ca[edge_index[1]]` is neighbour minus self.
    """
    if HAVE_TORCH_CLUSTER:
        return _ext_knn_graph(x, k=k, loop=loop, flow=flow)
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError(f"flow must be source_to_target or target_to_source, got {flow!r}")

    usable = torch.isfinite(x).all(dim=1).nonzero(as_tuple=False).view(-1)
    points = x[usable]
    count = points.size(0)
    if count == 0:
        return x.new_empty((2, 0), dtype=torch.long)

    wanted = min(k if loop else k + 1, count)
    distance = torch.cdist(points, points)
    # stable, so equal distances come back in index order — see the note above
    nearest = distance.sort(dim=1, stable=True).indices[:, :wanted]

    query = torch.arange(count, device=x.device).view(-1, 1).expand_as(nearest)
    row, col = nearest.reshape(-1), query.reshape(-1)
    if not loop:
        keep = row != col
        row, col = row[keep], col[keep]

    row, col = usable[row], usable[col]
    if flow == "source_to_target":
        return torch.stack([row, col], dim=0)
    return torch.stack([col, row], dim=0)
