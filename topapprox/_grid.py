"""Grid helpers shared across image and graph interfaces."""

from __future__ import annotations

import numpy as np


def sorted_grid_edges(birth: np.ndarray, shape: tuple[int, int], *, dual: bool) -> np.ndarray:
    """Return 2D grid edges sorted by their edge birth value."""

    n_rows, n_cols = shape
    edges = [
        (row * n_cols + col, (row + 1) * n_cols + col)
        for row in range(n_rows - 1)
        for col in range(n_cols)
    ]
    edges.extend(
        (row * n_cols + col, row * n_cols + col + 1)
        for row in range(n_rows)
        for col in range(n_cols - 1)
    )

    if dual:
        extra_vertex = n_rows * n_cols
        edges.extend(
            (row * n_cols + col, (row + 1) * n_cols + col + 1)
            for row in range(n_rows - 1)
            for col in range(n_cols - 1)
        )
        edges.extend(
            (row * n_cols + col, (row - 1) * n_cols + col + 1)
            for row in range(1, n_rows)
            for col in range(n_cols - 1)
        )
        edges.extend((col, extra_vertex) for col in range(n_cols))
        edges.extend(((n_rows - 1) * n_cols + col, extra_vertex) for col in range(n_cols))
        edges.extend((row * n_cols, extra_vertex) for row in range(1, n_rows - 1))
        edges.extend((row * n_cols + n_cols - 1, extra_vertex) for row in range(1, n_rows - 1))

    edge_array = np.asarray(edges, dtype=np.int64)
    edge_birth = np.maximum(birth[edge_array[:, 0]], birth[edge_array[:, 1]])
    return edge_array[np.argsort(edge_birth, kind="stable")]


def faces_from_array_shape(shape: tuple[int, ...]) -> list[list[int]]:
    """Build a graph-with-faces embedding for a 1D/2D array."""

    if len(shape) == 1:
        return [[index, index + 1] for index in range(shape[0] - 1)]
    if len(shape) != 2:
        raise ValueError("Only 1D and 2D arrays can be converted into GraphWithFaces.")

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


def boundary_from_array_shape(shape: tuple[int, ...]) -> list[int]:
    """Return the outer boundary cycle of a 1D/2D array embedding."""

    if len(shape) == 1:
        if shape[0] <= 1:
            return [0]
        forward = list(range(shape[0]))
        backward = list(range(shape[0] - 2, 0, -1))
        return forward + backward

    if len(shape) != 2:
        raise ValueError("Only 1D and 2D arrays can be converted into GraphWithFaces.")

    n_rows, n_cols = shape
    top = list(range(n_cols))
    right = [row * n_cols + n_cols - 1 for row in range(1, n_rows)]
    bottom = [(n_rows - 1) * n_cols + n_cols - 1 - offset for offset in range(1, n_cols + 1)]
    left = [n_cols * (n_rows - offset) for offset in range(2, n_rows)]
    return top + right + bottom + left
