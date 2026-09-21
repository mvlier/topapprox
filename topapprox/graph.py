"""Graphs with faces, and the two filtrations they carry.

A :class:`GraphWithFaces` is an embedded graph together with the 2-cells that
fill it: ``faces`` are filled, ``holes`` are not.  It exposes the two edge lists
the filters need:

``edges``
    the 1-skeleton, whose merge tree gives the connected components (``H_0``).
``dual_edges``
    the 1-skeleton plus one apex per face and hole, cone-connected to the cell's
    boundary.  Running the same merge-tree algorithm on ``-values`` there gives
    the cycles (``H_1``).
"""

from __future__ import annotations

import random
from functools import cached_property
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["GraphWithFaces", "random_graph_with_faces", "sweep_order"]

Polygon = Sequence[int]


def sweep_order(values: NDArray[np.float64], edges: NDArray[np.int64]) -> NDArray[np.int64]:
    """Order undirected edges the way a vertex sweep over ``values`` visits them.

    Returns rows ``(passive, active)``, ``active`` being the endpoint whose
    arrival in the sub-level set creates the edge.  This is the input contract of
    every ``reduce_edges`` kernel; see :func:`topapprox.grids.sweep_edges` for
    the grid counterpart.
    """
    if edges.shape[0] == 0:
        return edges.reshape(0, 2)

    rank = np.empty(values.shape[0], dtype=np.int64)
    rank[np.argsort(values, kind="stable")] = np.arange(values.shape[0], dtype=np.int64)

    left, right = edges[:, 0], edges[:, 1]
    tied = values[left] == values[right]
    left_is_active = (values[left] > values[right]) | (tied & (rank[left] < rank[right]))
    active = np.where(left_is_active, left, right)
    passive = np.where(left_is_active, right, left)

    visit = np.argsort(rank[active], kind="stable")
    return np.ascontiguousarray(np.stack((passive[visit], active[visit]), axis=1))


def _flatten(polygons: Sequence[Polygon]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Concatenated vertex indices of ``polygons`` plus their lengths."""
    sizes = (len(polygon) for polygon in polygons)
    lengths = np.fromiter(sizes, dtype=np.int64, count=len(polygons))
    if lengths.size == 0:
        return np.empty(0, dtype=np.int64), lengths
    flat = np.concatenate([np.asarray(polygon, dtype=np.int64).ravel() for polygon in polygons])
    return flat, lengths


def _cycle_edges(polygons: Sequence[Polygon]) -> NDArray[np.int64]:
    """Boundary edges of every polygon, as consecutive pairs around each cycle."""
    flat, lengths = _flatten(polygons)
    if flat.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    starts = np.cumsum(lengths) - lengths
    following = np.arange(1, flat.size + 1, dtype=np.int64)
    following[starts + lengths - 1] = starts  # wrap each cycle back to its start
    return np.stack((flat, flat[following]), axis=1)


def _cone_edges(polygons: Sequence[Polygon], apexes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Edges joining each polygon's boundary to its own apex vertex."""
    flat, lengths = _flatten(polygons)
    if flat.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack((flat, np.repeat(apexes, lengths)), axis=1)


def _unique_edges(pairs: NDArray[np.int64], n_vertices: int) -> NDArray[np.int64]:
    """Deduplicate undirected edges and drop self-loops."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    keep = low != high
    codes = np.unique(low[keep] * n_vertices + high[keep])
    return np.stack((codes // n_vertices, codes % n_vertices), axis=1)


class GraphWithFaces:
    """An embedded graph, its filled faces and its holes.

    Parameters
    ----------
    faces
        Vertex cycles bounding the filled 2-cells.
    holes
        Vertex cycles bounding the unfilled 2-cells, the outer boundary included.
    signal
        One value per vertex.
    edges
        Optional explicit 1-skeleton.  Defaults to the boundary edges of
        ``faces`` and ``holes``.
    is_triangulated
        Set for a triangulated surface: its faces need no apex, since the three
        vertices of a triangle are already mutually adjacent.
    """

    def __init__(
        self,
        faces: Iterable[Polygon] = (),
        holes: Iterable[Polygon] = (),
        signal: ArrayLike | None = None,
        *,
        edges: ArrayLike | None = None,
        is_triangulated: bool = False,
    ) -> None:
        if signal is None:
            raise ValueError("signal must be provided.")

        self.faces = [list(face) for face in faces]
        self.holes = [list(hole) for hole in holes]
        self.is_triangulated = bool(is_triangulated)
        self.signal = np.ascontiguousarray(signal, dtype=np.float64).ravel()
        self.n_vertices = int(self.signal.shape[0])

        skeleton = (
            _unique_edges(np.asarray(edges, dtype=np.int64), self.n_vertices)
            if edges is not None
            else _unique_edges(_cycle_edges(self.faces + self.holes), self.n_vertices)
        )
        if skeleton.size and int(skeleton.max()) >= self.n_vertices:
            raise ValueError("Edges reference vertices outside the signal.")
        self._skeleton = skeleton

    def __repr__(self) -> str:
        return (
            f"GraphWithFaces(n_vertices={self.n_vertices}, n_edges={len(self._skeleton)}, "
            f"n_faces={len(self.faces)}, n_holes={len(self.holes)}, "
            f"is_triangulated={self.is_triangulated})"
        )

    # ------------------------------------------------------------------
    # Filtrations
    # ------------------------------------------------------------------

    @cached_property
    def edges(self) -> NDArray[np.int64]:
        """1-skeleton edges, ordered for a sweep over :attr:`signal`."""
        return sweep_order(self.signal, self._skeleton)

    @cached_property
    def _apex_cells(self) -> tuple[list[Polygon], NDArray[np.float64]]:
        """Cells that receive an apex vertex, and the value assigned to each apex."""
        cells: list[Polygon] = [] if self.is_triangulated else list(self.faces)
        values = [
            float(self.signal[np.asarray(face, dtype=np.int64)].max()) for face in cells
        ]
        cells += self.holes
        values += [np.inf] * len(self.holes)
        return cells, np.asarray(values, dtype=np.float64)

    @cached_property
    def dual_values(self) -> NDArray[np.float64]:
        """Vertex values extended by one entry per apex; holes sit at ``+inf``."""
        _, apex_values = self._apex_cells
        return np.concatenate((self.signal, apex_values))

    @cached_property
    def dual_edges(self) -> NDArray[np.int64]:
        """Skeleton plus apex cones, ordered for a sweep over ``-dual_values``."""
        cells, apex_values = self._apex_cells
        apexes = self.n_vertices + np.arange(len(cells), dtype=np.int64)
        pairs = np.concatenate((self._skeleton, _cone_edges(cells, apexes)))
        return sweep_order(-self.dual_values, _unique_edges(pairs, self.dual_values.shape[0]))

    def with_signal(self, signal: ArrayLike) -> "GraphWithFaces":
        """The same cells and 1-skeleton, carrying a different vertex signal."""
        return GraphWithFaces(
            self.faces,
            self.holes,
            signal,
            edges=self._skeleton,
            is_triangulated=self.is_triangulated,
        )

    def draw(self, **kwargs):
        """Plot the graph; see :func:`topapprox.plotting.draw_graph`."""
        from .plotting import draw_graph

        return draw_graph(self, **kwargs)


def random_graph_with_faces(
    n_faces: int,
    *,
    face_size_range: tuple[int, int] = (3, 12),
    n_holes: int = 1,
    unique_faces: bool = True,
    rng: random.Random | None = None,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Generate a random planar graph with faces.

    Returns ``(faces, holes, n_vertices)``; useful for randomised testing.
    """
    rng = rng or random
    min_size, max_size = face_size_range
    n_vertices = rng.randint(min_size, max_size)
    boundary = list(range(n_vertices))
    faces = [boundary.copy()]
    min_new = 1 if unique_faces else 0

    for _ in range(n_faces - 1):
        start = rng.randint(0, len(boundary) - 1)
        increment = rng.randint(2, min(len(boundary) - 1, max_size - min_new))
        end = start + increment
        new_count = rng.randint(max(min_new, min_size - increment), max_size - increment)
        segment = list(range(n_vertices, n_vertices + new_count))

        faces.append(_wrap(boundary, start, end) + segment)
        n_vertices += new_count
        boundary = _wrap(boundary, end - 1, start + 1) + list(reversed(segment))

    faces.append(boundary.copy())
    holes = [faces.pop(rng.randrange(len(faces))) for _ in range(min(n_holes, len(faces)))]
    return faces, holes, n_vertices


def _wrap(values: list[int], start: int, end: int) -> list[int]:
    """Cyclic slice ``values[start:end]``."""
    size = len(values)
    start %= size
    end %= size
    return values[start:end] if start < end else values[start:] + values[:end]
