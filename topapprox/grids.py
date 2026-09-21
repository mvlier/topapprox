"""Regular-grid geometry: neighbourhoods, sweep edges and cell complexes.

A signal sampled on a ``d``-dimensional grid is filtered through the merge tree
of its *primal* graph (the ``2d`` axis-aligned neighbours).  Filtering the
``1``-dimensional features of the same signal amounts to filtering the ``0``-th
features of ``-signal`` on the *dual* graph: full ``3^d - 1`` connectivity plus a
single point at infinity that all boundary cells attach to.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

__all__ = ["neighbour_steps", "sweep_edges", "grid_faces", "grid_boundary"]

Shape = tuple[int, ...]


def neighbour_steps(ndim: int, *, dual: bool) -> tuple[tuple[int, ...], ...]:
    """Signed per-axis steps to the neighbours of a grid cell.

    The order is part of the contract between the backends: every backend must
    visit the edges of a vertex in this order so that all of them build the very
    same merge tree on plateaus.  The C++ kernel mirrors this enumeration in
    ``neighbour_steps`` (see ``_reduce_cpp.cpp``).
    """
    if dual:
        return tuple(step for step in product((-1, 0, 1), repeat=ndim) if any(step))
    steps = []
    for axis in range(ndim):
        for direction in (-1, 1):
            step = [0] * ndim
            step[axis] = direction
            steps.append(tuple(step))
    return tuple(steps)


def sweep_edges(birth: NDArray[np.float64], shape: Shape, *, dual: bool) -> NDArray[np.int64]:
    """Grid edges in the order a vertex sweep over ``birth`` would visit them.

    Row ``i`` of the result is ``(passive, active)``: ``active`` is the endpoint
    whose arrival in the sub-level set creates the edge.  Feeding these rows to
    an edge kernel reproduces, merge for merge, what the C++ grid kernel does
    without materialising an edge list.

    ``birth`` holds one value per cell, plus a trailing value for the point at
    infinity when ``dual`` is true.
    """
    n_cells = int(np.prod(shape))
    order = np.argsort(birth[:n_cells], kind="stable")
    rank = np.empty(n_cells, dtype=np.int64)
    rank[order] = np.arange(n_cells, dtype=np.int64)

    coords = np.unravel_index(np.arange(n_cells, dtype=np.int64), shape)
    strides = np.ones(len(shape), dtype=np.int64)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]

    steps = neighbour_steps(len(shape), dual=dual)

    actives: list[NDArray[np.int64]] = []
    passives: list[NDArray[np.int64]] = []
    ranks: list[NDArray[np.int64]] = []

    for step_index, step in enumerate(steps):
        inside = np.ones(n_cells, dtype=bool)
        offset = 0
        for axis, delta in enumerate(step):
            if delta < 0:
                inside &= coords[axis] > 0
            elif delta > 0:
                inside &= coords[axis] < shape[axis] - 1
            offset += delta * int(strides[axis])
        active = np.flatnonzero(inside)
        if active.size == 0:
            continue
        passive = active + offset
        # Keep the orientation in which `active` is the vertex that creates the
        # edge; the opposite orientation would merge two identical components.
        later = birth[passive] < birth[active]
        tied = (birth[passive] == birth[active]) & (rank[active] < rank[passive])
        keep = later | tied
        actives.append(active[keep])
        passives.append(passive[keep])
        ranks.append(np.full(int(keep.sum()), step_index, dtype=np.int64))

    if dual:
        on_boundary = np.zeros(n_cells, dtype=bool)
        for axis, extent in enumerate(shape):
            on_boundary |= (coords[axis] == 0) | (coords[axis] == extent - 1)
        active = np.flatnonzero(on_boundary)
        actives.append(active)
        passives.append(np.full(active.size, n_cells, dtype=np.int64))
        ranks.append(np.full(active.size, len(steps), dtype=np.int64))

    active = np.concatenate(actives) if actives else np.empty(0, dtype=np.int64)
    passive = np.concatenate(passives) if passives else np.empty(0, dtype=np.int64)
    step_index = np.concatenate(ranks) if ranks else np.empty(0, dtype=np.int64)

    visit = np.lexsort((step_index, rank[active]))
    return np.stack((passive[visit], active[visit]), axis=1)


def grid_faces(shape: Shape) -> list[list[int]]:
    """Cells of a 1-D or 2-D array seen as a graph with faces."""
    if len(shape) == 1:
        return [[i, i + 1] for i in range(shape[0] - 1)]
    if len(shape) != 2:
        raise ValueError("Only 1-D and 2-D arrays have a canonical face structure.")
    n_rows, n_cols = shape
    return [
        [
            row * n_cols + col,
            (row + 1) * n_cols + col,
            (row + 1) * n_cols + col + 1,
            row * n_cols + col + 1,
        ]
        for row in range(n_rows - 1)
        for col in range(n_cols - 1)
    ]


def grid_boundary(shape: Shape) -> list[int]:
    """Outer boundary cycle of a 1-D or 2-D array seen as a graph with faces."""
    if len(shape) == 1:
        shape = (1, shape[0])
    if len(shape) != 2:
        raise ValueError("Only 1-D and 2-D arrays have a canonical face structure.")

    n_rows, n_cols = shape
    if min(n_rows, n_cols) == 1:
        # A single row or column: the boundary runs to the far end and back.
        line = list(range(n_rows * n_cols))
        return line if len(line) <= 2 else line + line[-2:0:-1]

    top = list(range(n_cols))
    right = [row * n_cols + n_cols - 1 for row in range(1, n_rows)]
    bottom = [(n_rows - 1) * n_cols + col for col in range(n_cols - 2, -1, -1)]
    left = [row * n_cols for row in range(n_rows - 2, 0, -1)]
    return top + right + bottom + left
