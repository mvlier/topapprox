"""Reference "link and reduce" kernel in pure Python.

Correctness reference for the numba and C++ kernels: same total order, same
merge tree, no compilation required.  The union-find runs over Python lists
rather than NumPy arrays because scalar indexing of an ``ndarray`` costs roughly
an order of magnitude more than indexing a ``list``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["reduce_edges"]


def reduce_edges(
    birth: NDArray[np.float64], edges: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """Build a merge tree from edges in sweep order.

    Row ``i`` of ``edges`` is ``(passive, active)``, ``active`` being the vertex
    whose arrival creates the edge; rows sharing an ``active`` endpoint must be
    consecutive, since they form a single merge event.  See
    :func:`topapprox.graph.sweep_order`.

    Returns ``(parent, linking_vertex, root)`` with ``parent[root] == root`` and
    ``linking_vertex[root] == -1``.
    """
    value = birth.tolist()
    n_vertices = len(value)
    parent = list(range(n_vertices))
    ancestor = list(range(n_vertices))
    linking_vertex = [-1] * n_vertices
    root = 0

    passive_of, active_of = edges[:, 0].tolist(), edges[:, 1].tolist()
    n_edges = len(active_of)

    def find(vertex: int) -> int:
        while ancestor[vertex] != vertex:
            ancestor[vertex] = ancestor[ancestor[vertex]]
            vertex = ancestor[vertex]
        return vertex

    first = 0
    while first < n_edges:
        witness = active_of[first]
        last = first
        while last < n_edges and active_of[last] == witness:
            last += 1

        # Every component meeting at `witness` joins the oldest of them, ties
        # broken by index; `witness` itself joins last, so a basin's subtree is
        # exactly the component it had just before merging.
        survivor = find(witness)
        for k in range(first, last):
            candidate = find(passive_of[k])
            if value[candidate] < value[survivor] or (
                value[candidate] == value[survivor] and candidate < survivor
            ):
                survivor = candidate

        for vertex in [*passive_of[first:last], witness]:
            dying = find(vertex)
            if dying == survivor:
                continue
            parent[dying] = survivor
            ancestor[dying] = survivor
            linking_vertex[dying] = witness
            root = survivor
        first = last

    return (
        np.asarray(parent, dtype=np.int64),
        np.asarray(linking_vertex, dtype=np.int64),
        root,
    )
