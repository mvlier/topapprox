"""JIT-compiled "link and reduce" kernel.

Same algorithm and total order as :mod:`topapprox._reduce_python`, compiled by
numba.  Importing this module triggers the numba import, so it is only loaded on
demand by :mod:`topapprox.backends`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["reduce_edges"]


@njit(cache=True, nogil=True)
def _find(vertex, ancestor):
    while ancestor[vertex] != vertex:
        ancestor[vertex] = ancestor[ancestor[vertex]]
        vertex = ancestor[vertex]
    return vertex


@njit(cache=True, nogil=True)
def _reduce_edges(birth, edges, parent, ancestor, linking_vertex):
    n_edges = edges.shape[0]
    root = 0
    first = 0
    while first < n_edges:
        witness = edges[first, 1]
        last = first
        while last < n_edges and edges[last, 1] == witness:
            last += 1

        # Every component meeting at `witness` joins the oldest of them, ties
        # broken by index; `witness` itself joins last, so a basin's subtree is
        # exactly the component it had just before merging.
        survivor = _find(witness, ancestor)
        for k in range(first, last):
            candidate = _find(edges[k, 0], ancestor)
            if birth[candidate] < birth[survivor] or (
                birth[candidate] == birth[survivor] and candidate < survivor
            ):
                survivor = candidate

        for k in range(first, last + 1):
            if k == last:
                vertex = witness
            else:
                vertex = edges[k, 0]
            dying = _find(vertex, ancestor)
            if dying == survivor:
                continue
            parent[dying] = survivor
            ancestor[dying] = survivor
            linking_vertex[dying] = witness
            root = survivor
        first = last
    return root


def reduce_edges(birth, edges):
    """Build a merge tree from edges in sweep order.

    See :func:`topapprox._reduce_python.reduce_edges` for the input contract.
    """
    n_vertices = birth.shape[0]
    parent = np.arange(n_vertices, dtype=np.int64)
    ancestor = np.arange(n_vertices, dtype=np.int64)
    linking_vertex = np.full(n_vertices, -1, dtype=np.int64)
    root = _reduce_edges(birth, edges, parent, ancestor, linking_vertex)
    return parent, linking_vertex, int(root)
