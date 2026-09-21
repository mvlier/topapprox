"""The merge tree (basin hierarchy tree) of a scalar signal.

The tree is stored in the smallest form that determines it -- ``birth``,
``parent``, ``linking_vertex`` and ``root`` -- and every derived quantity is
computed lazily with vectorised NumPy.

Two structural facts drive the implementation:

* ``birth`` never decreases from a vertex towards its children, and ``death``
  never increases.  A query such as "the topmost ancestor whose basin is filled"
  therefore reduces to a *maximum along the path to the root*.
* That path maximum is computed by pointer doubling: ``ceil(log2(height))``
  fully vectorised passes.  No Python-level tree walk is needed, so filtering a
  megapixel image costs a handful of array operations.
"""

from __future__ import annotations

from functools import cached_property
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MergeTree", "DIAGRAM_COLUMNS"]

#: Columns of the arrays returned by :meth:`MergeTree.diagram`.
DIAGRAM_COLUMNS = (
    "birth",  # value at which the basin appears
    "death",  # value at which it merges into an older basin
    "birth_vertex",  # the local minimum that started it
    "death_vertex",  # the saddle that merged it away (-1 for the essential pair)
    "basin_size",  # number of vertices in the basin just before it merges
    "depth",  # distance from the root of the merge tree
    "height",  # distance to the deepest leaf of the positive-persistence subtree
)

WEIGHTS = ("basin_size", "persistence", "basin_size_x_persistence")


def _concatenate_ranges(start: NDArray[np.int64], length: NDArray[np.int64]) -> NDArray[np.int64]:
    """``np.concatenate([np.arange(s, s + n) for s, n in zip(start, length)])``."""
    total = int(length.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    shifted = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(length)[:-1]))
    return np.arange(total, dtype=np.int64) + np.repeat(start - shifted, length)


class MergeTree:
    """Merge tree of a scalar signal on a graph.

    Parameters
    ----------
    birth
        Value of the signal at each vertex.
    parent
        Merge-tree parent of each vertex; the root is its own parent.
    linking_vertex
        Vertex whose arrival merged each basin into its parent, ``-1`` at the root.
    root
        Index of the root, i.e. of the global minimum.
    """

    def __init__(self, birth: ArrayLike, parent: ArrayLike, linking_vertex: ArrayLike, root: int):
        self.birth = np.ascontiguousarray(birth, dtype=np.float64)
        self.parent = np.ascontiguousarray(parent, dtype=np.int64)
        self.linking_vertex = np.ascontiguousarray(linking_vertex, dtype=np.int64)
        self.root = int(root)
        if not self.birth.shape == self.parent.shape == self.linking_vertex.shape:
            raise ValueError("birth, parent and linking_vertex must have the same length.")
        if self.birth.ndim != 1:
            raise ValueError("birth must be one-dimensional.")

    def __repr__(self) -> str:
        return f"MergeTree(n_vertices={self.n_vertices}, root={self.root})"

    @property
    def n_vertices(self) -> int:
        return int(self.birth.shape[0])

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @cached_property
    def death(self) -> NDArray[np.float64]:
        """Value at which each basin merges into an older one; ``inf`` at the root."""
        values = np.full(self.n_vertices, np.inf)
        merged = self.linking_vertex >= 0
        values[merged] = self.birth[self.linking_vertex[merged]]
        return values

    @cached_property
    def persistence(self) -> NDArray[np.float64]:
        """``death - birth``; ``0`` for basins that never existed, ``inf`` at the root."""
        return self.death - self.birth

    @cached_property
    def depth(self) -> NDArray[np.int64]:
        """Number of edges from each vertex to the root."""
        steps = (self.parent != np.arange(self.n_vertices)).astype(np.int64)
        for ancestor in self._doubling_ancestors():
            steps = steps + steps[ancestor]
        return steps

    @cached_property
    def subtree_size(self) -> NDArray[np.int64]:
        """Number of vertices in each basin, the vertex itself included."""
        size = np.ones(self.n_vertices, dtype=np.int64)
        depth = self.depth
        reach = 1
        for ancestor in self._doubling_ancestors():
            # A vertex contributes its own (partial) count to the ancestor
            # `reach` levels up, provided that ancestor is not the saturated root.
            deep_enough = depth >= reach
            if not deep_enough.any():
                break
            size = size + np.bincount(
                ancestor[deep_enough],
                weights=size[deep_enough],
                minlength=self.n_vertices,
            ).astype(np.int64)
            reach *= 2
        return size

    @cached_property
    def positive_height(self) -> NDArray[np.int64]:
        """Height of each vertex in the forest spanned by positive-persistence basins."""
        height = np.zeros(self.n_vertices, dtype=np.int64)
        basins = np.flatnonzero(self._is_basin)
        if basins.size == 0:
            return height

        by_depth = basins[np.argsort(self.depth[basins], kind="stable")]
        levels = self.depth[by_depth]
        bounds = np.searchsorted(levels, np.arange(int(levels[-1]) + 2))
        for level in range(len(bounds) - 2, 0, -1):
            block = by_depth[bounds[level] : bounds[level + 1]]
            if block.size:
                np.maximum.at(height, self.parent[block], height[block] + 1)
        return height

    @cached_property
    def _is_basin(self) -> NDArray[np.bool_]:
        """Vertices that start a basin of strictly positive, finite persistence."""
        return (self.persistence > 0) & (self.linking_vertex >= 0)

    @cached_property
    def _children(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Children in compressed form: ``(offset, index)`` with CSR semantics."""
        vertices = np.arange(self.n_vertices, dtype=np.int64)
        vertices = vertices[vertices != self.root]
        index = vertices[np.argsort(self.parent[vertices], kind="stable")]

        counts = np.bincount(self.parent, minlength=self.n_vertices)
        counts[self.root] -= 1  # the root is listed as its own parent
        offset = np.zeros(self.n_vertices + 1, dtype=np.int64)
        np.cumsum(counts, out=offset[1:])
        return offset, index

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def _doubling_ancestors(self):
        """Yield ``parent``, ``parent^2``, ``parent^4``, ... until it saturates.

        The last array yielded maps every vertex to the root, so a reduction run
        over the whole sequence covers each vertex's entire root path.
        """
        ancestor = self.parent
        while True:
            yield ancestor
            nxt = ancestor[ancestor]
            if np.array_equal(nxt, ancestor):
                return
            ancestor = nxt

    def _path_max(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Maximum of ``values`` over each vertex's ancestors, itself included."""
        best = np.array(values, dtype=np.float64, copy=True)
        for ancestor in self._doubling_ancestors():
            np.maximum(best, best[ancestor], out=best)
        return best

    def _path_argmax(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        """Deepest ancestor (itself included) attaining the maximum of ``values``."""
        best = np.array(values, dtype=np.float64, copy=True)
        argmax = np.arange(self.n_vertices, dtype=np.int64)
        for ancestor in self._doubling_ancestors():
            candidate = best[ancestor]
            better = candidate > best
            argmax = np.where(better, argmax[ancestor], argmax)
            best = np.where(better, candidate, best)
        return argmax

    def children(self, vertex: int) -> NDArray[np.int64]:
        """Direct children of ``vertex``."""
        offset, index = self._children
        return index[offset[vertex] : offset[vertex + 1]]

    def descendants(self, vertex: int) -> NDArray[np.int64]:
        """Every vertex in the basin of ``vertex``, itself included."""
        offset, index = self._children
        frontier = np.array([int(vertex)], dtype=np.int64)
        levels = []
        while frontier.size:  # one vectorised step per level of the subtree
            levels.append(frontier)
            start = offset[frontier]
            frontier = index[_concatenate_ranges(start, offset[frontier + 1] - start)]
        return np.concatenate(levels)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def low_pers_filter(
        self, epsilon: float, *, size_range: Sequence[float] | None = None
    ) -> NDArray[np.float64]:
        """Fill every basin whose persistence is below ``epsilon``.

        Each vertex of a filled basin takes the value at which that basin merges
        into its parent, so the result stays within ``epsilon`` of ``birth`` in
        the supremum norm.

        Parameters
        ----------
        epsilon
            Persistence threshold; basins with ``0 < persistence < epsilon`` go.
        size_range
            Optional ``(min_size, max_size)``.  Basins outside the range are kept
            even when their persistence is small; basins that are too large are
            descended into, so their small sub-basins can still be filled.
        """
        epsilon = float(epsilon)
        if size_range is not None:
            return self._filter_by_size(epsilon, size_range)

        doomed = (self.persistence > 0) & (self.persistence < epsilon)
        # Since `death` grows towards the root, the maximum over the root path
        # is the death value of the *topmost* doomed ancestor -- exactly the
        # basin a top-down traversal would have stopped at.
        fill = self._path_max(np.where(doomed, self.death, -np.inf))
        return np.where(np.isfinite(fill), fill, self.birth)

    def _filter_by_size(self, epsilon: float, size_range: Sequence[float]) -> NDArray[np.float64]:
        low, high = float(size_range[0]), float(size_range[1])
        persistence, death, size = self.persistence, self.death, self.subtree_size
        filtered = self.birth.copy()

        stack = [self.root]
        while stack:
            for child in self.children(stack.pop()):
                if persistence[child] <= 0:
                    continue  # a basin that never existed, and neither did its subtree
                if persistence[child] >= epsilon or size[child] > high:
                    stack.append(int(child))  # survives, or is too big to fill wholesale
                elif size[child] >= low:
                    filtered[self.descendants(child)] = death[child]
        return filtered

    # ------------------------------------------------------------------
    # Basins
    # ------------------------------------------------------------------

    def basin_representatives(self, epsilon: float = 0.0) -> NDArray[np.int64]:
        """Index of the basin each vertex belongs to at threshold ``epsilon``.

        A vertex belongs to the deepest basin that both survives the threshold
        and is reachable from the root through surviving basins only.  Depth
        strictly increases away from the root, so that vertex is the one
        maximising ``depth`` among the unblocked ancestors.
        """
        blocked = self._blocked(epsilon)
        return self._path_argmax(np.where(blocked, -1.0, self.depth.astype(float)))

    def basin_map(self, epsilon: float = 0.0) -> NDArray[np.float64]:
        """Birth value of the basin each vertex belongs to at threshold ``epsilon``.

        Parents are born before their children, so feeding the result to a
        sequential colour map shows the hierarchy directly: the colour gap
        between a basin and its parent is its persistence.
        """
        return self.birth[self.basin_representatives(epsilon)]

    def _blocked(self, epsilon: float) -> NDArray[np.bool_]:
        """Vertices cut off from the root by a basin that does not survive."""
        surviving = (self.persistence > 0) & (self.persistence >= float(epsilon))
        surviving[self.root] = True
        return self._path_max(np.where(surviving, 0.0, 1.0)) > 0

    # ------------------------------------------------------------------
    # Persistence diagram
    # ------------------------------------------------------------------

    def diagram(self, *, reduced: bool = True) -> NDArray[np.float64]:
        """Persistence pairs, one row per basin, with columns :data:`DIAGRAM_COLUMNS`.

        ``reduced=False`` appends the essential pair of the global minimum.
        """
        basins = np.flatnonzero(self._is_basin)
        rows = np.column_stack(
            (
                self.birth[basins],
                self.death[basins],
                basins,
                self.linking_vertex[basins],
                self.subtree_size[basins],
                self.depth[basins],
                self.positive_height[basins],
            )
        ).astype(float)
        if reduced:
            return rows

        essential = np.array(
            [
                [
                    self.birth[self.root],
                    np.inf,
                    self.root,
                    -1.0,
                    np.inf,
                    self.depth[self.root],
                    self.positive_height[self.root],
                ]
            ]
        )
        return np.concatenate((rows, essential))

    def _diagram_weights(self, diagram: NDArray[np.float64], weight) -> NDArray[np.float64] | float:
        if weight is None:
            return 1.0
        if isinstance(weight, str):
            lifetime = diagram[:, 1] - diagram[:, 0]
            if weight == "basin_size":
                return diagram[:, 4]
            if weight == "persistence":
                return lifetime
            if weight == "basin_size_x_persistence":
                return diagram[:, 4] * lifetime
            raise ValueError(f"Unknown weight {weight!r}; expected one of {WEIGHTS} or an array.")
        return np.asarray(weight, dtype=float)

    def _distribution(self, column: int, value_range, ratio: bool, weight) -> NDArray[np.float64]:
        diagram = self.diagram()
        if diagram.shape[0] == 0:
            return np.zeros(0)
        weights = self._diagram_weights(diagram, weight)
        values = diagram[:, column].astype(np.int64)
        low, high = (int(values.min()), int(values.max())) if value_range is None else value_range
        counts = np.array(
            [np.sum((values == level) * weights) for level in range(int(low), int(high) + 1)],
            dtype=float,
        )
        total = counts.sum() if ratio else 1.0
        return counts / total if total else counts

    def depth_distribution(self, *, depth_range=None, ratio=False, weight=None):
        """Weighted number of basins per merge-tree depth."""
        return self._distribution(DIAGRAM_COLUMNS.index("depth"), depth_range, ratio, weight)

    def height_distribution(self, *, height_range=None, ratio=False, weight=None):
        """Weighted number of basins per merge-tree height."""
        return self._distribution(DIAGRAM_COLUMNS.index("height"), height_range, ratio, weight)
