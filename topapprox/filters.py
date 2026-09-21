"""User-facing filters for arrays and graph signals.

All of them do the same three things: turn the input into a filtration (values
plus edges), hand it to a backend to obtain a :class:`~topapprox.merge_tree.MergeTree`,
and map the tree's answers back to the caller's layout.

Filtering ``1``-dimensional features is filtering ``0``-dimensional features of
the negated signal on the dual complex, so a single ``dual`` flag covers both.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import resolve_backend
from .graph import GraphWithFaces
from .grids import grid_boundary, grid_faces
from .merge_tree import MergeTree

__all__ = [
    "ImageFilter",
    "GraphFilter",
    "TopologicalFilter",
    "low_pers_filter",
    "persistence_diagram",
]


class TopologicalFilter:
    """Shared front end: build the merge tree once, then answer queries about it."""

    def __init__(self, *, dual: bool = False, method: str = "auto") -> None:
        self.dual = bool(dual)
        self.backend = resolve_backend(method)
        self._tree: MergeTree | None = None

    @property
    def method(self) -> str:
        return self.backend.name

    @property
    def tree(self) -> MergeTree:
        """The merge tree of the filtration, built on first access."""
        if self._tree is None:
            self._tree = self._build_tree()
        return self._tree

    def _build_tree(self) -> MergeTree:
        raise NotImplementedError

    def _restore(self, values: NDArray[np.float64]) -> np.ndarray:
        """Map per-tree-vertex values back to the caller's signal layout."""
        raise NotImplementedError

    def low_pers_filter(self, epsilon: float, *, size_range: Sequence[float] | None = None):
        """Signal with every feature of persistence below ``epsilon`` removed."""
        return self._restore(self.tree.low_pers_filter(epsilon, size_range=size_range))

    def basin_map(self, epsilon: float = 0.0):
        """Birth value of the basin each sample belongs to at threshold ``epsilon``."""
        return self._restore(self.tree.basin_map(epsilon))

    def diagram(self, *, reduced: bool = True) -> NDArray[np.float64]:
        """Persistence pairs with columns :data:`~topapprox.merge_tree.DIAGRAM_COLUMNS`."""
        rows = self.tree.diagram(reduced=reduced)
        if not self.dual or rows.shape[0] == 0:
            return rows
        # The dual tree filters -signal, so its pairs read backwards.
        flipped = rows.copy()
        flipped[:, 0], flipped[:, 1] = -rows[:, 1], -rows[:, 0]
        flipped[:, 2], flipped[:, 3] = rows[:, 3], rows[:, 2]
        return flipped


class ImageFilter(TopologicalFilter):
    """Low-persistence filtering of a signal sampled on a 1-D, 2-D or 3-D grid.

    Parameters
    ----------
    array
        The signal.  ``dual=False`` treats it with the ``2d`` axis-aligned
        neighbourhood (connected components), ``dual=True`` with full ``3^d - 1``
        connectivity plus a point at infinity (cycles / cavities).
    method
        ``"auto"``, ``"cpp"``, ``"numba"`` or ``"python"``.
    """

    def __init__(self, array: ArrayLike, *, dual: bool = False, method: str = "auto") -> None:
        signal = np.ascontiguousarray(array, dtype=np.float64)
        if signal.ndim not in (1, 2, 3):
            raise ValueError(f"Expected a 1-D, 2-D or 3-D array, got shape {signal.shape}.")
        super().__init__(dual=dual, method=method)
        self.signal = signal
        self.shape = signal.shape
        self.n_cells = signal.size

    def __repr__(self) -> str:
        return f"ImageFilter(shape={self.shape}, dual={self.dual}, method={self.method!r})"

    def _build_tree(self) -> MergeTree:
        values = -self.signal.ravel() if self.dual else self.signal.ravel()
        if self.dual:  # the point at infinity, born before everything else
            values = np.concatenate((values, [-np.inf]))
        parent, linking_vertex, root = self.backend.reduce_grid(values, self.shape, dual=self.dual)
        return MergeTree(values, parent, linking_vertex, root)

    def _restore(self, values: NDArray[np.float64]) -> np.ndarray:
        out = values[: self.n_cells]
        return (-out if self.dual else out).reshape(self.shape)


class GraphFilter(TopologicalFilter):
    """Low-persistence filtering of a signal on a :class:`~topapprox.graph.GraphWithFaces`."""

    def __init__(
        self, graph: GraphWithFaces, *, dual: bool = False, method: str = "auto"
    ) -> None:
        if not isinstance(graph, GraphWithFaces):
            raise TypeError("graph must be a GraphWithFaces; see GraphFilter.from_faces.")
        super().__init__(dual=dual, method=method)
        self.graph = graph
        self.shape: tuple[int, ...] | None = None

    @classmethod
    def from_faces(
        cls,
        faces: Iterable[Sequence[int]],
        holes: Iterable[Sequence[int]] = (),
        signal: ArrayLike | None = None,
        *,
        edges: ArrayLike | None = None,
        is_triangulated: bool = False,
        dual: bool = False,
        method: str = "auto",
    ) -> "GraphFilter":
        """Build the graph from its cells and filter the signal on it."""
        graph = GraphWithFaces(
            faces, holes, signal, edges=edges, is_triangulated=is_triangulated
        )
        return cls(graph, dual=dual, method=method)

    @classmethod
    def from_array(
        cls, array: ArrayLike, *, dual: bool = False, method: str = "auto"
    ) -> "GraphFilter":
        """Filter a 1-D or 2-D array through its canonical graph-with-faces structure."""
        signal = np.ascontiguousarray(array, dtype=np.float64)
        graph = GraphWithFaces(
            grid_faces(signal.shape), [grid_boundary(signal.shape)], signal.ravel()
        )
        instance = cls(graph, dual=dual, method=method)
        instance.shape = signal.shape
        return instance

    def __repr__(self) -> str:
        return f"GraphFilter({self.graph!r}, dual={self.dual}, method={self.method!r})"

    def _build_tree(self) -> MergeTree:
        if self.dual:
            values, edges = -self.graph.dual_values, self.graph.dual_edges
        else:
            values, edges = self.graph.signal, self.graph.edges
        parent, linking_vertex, root = self.backend.reduce_edges(values, edges)
        return MergeTree(values, parent, linking_vertex, root)

    def _restore(self, values: NDArray[np.float64]) -> np.ndarray:
        out = values[: self.graph.n_vertices]
        out = -out if self.dual else out
        return out.reshape(self.shape) if self.shape is not None else out

    def with_dual(self, dual: bool) -> "GraphFilter":
        """The same graph and backend, filtered in the other degree."""
        if dual == self.dual:
            return self
        twin = GraphFilter(self.graph, dual=dual, method=self.method)
        twin.shape = self.shape
        return twin

    def diagrams(self) -> list[NDArray[np.float64]]:
        """``[H0, H1]`` persistence diagrams as ``(n, 2)`` birth/death arrays.

        The ``H0`` diagram includes the essential class of the global minimum.
        A list, so it can go straight into ``persim.plot_diagrams``.
        """
        primal = self.with_dual(False).diagram(reduced=False)
        dual = self.with_dual(True).diagram(reduced=True)
        return [primal[:, :2], dual[:, :2]]


# ----------------------------------------------------------------------
# One-shot helpers
# ----------------------------------------------------------------------


def _thresholds(epsilon, steps: tuple[str, ...]) -> tuple[float, ...]:
    """Resolve ``epsilon`` into one threshold per filtering step."""
    if isinstance(epsilon, Mapping):
        return tuple(float(epsilon[int(step)] if int(step) in epsilon else epsilon[step])
                     for step in steps)
    if np.isscalar(epsilon):
        return tuple(float(epsilon) for _ in steps)
    values = tuple(float(value) for value in epsilon)
    if len(values) != 2:
        raise ValueError("A sequence epsilon must hold exactly two values, for H0 and H1.")
    return tuple(values[int(step)] for step in steps)


def _make_filter(signal, *, dual: bool, method: str, template: GraphWithFaces | None):
    if template is None:
        return ImageFilter(signal, dual=dual, method=method)
    return GraphFilter(template.with_signal(signal), dual=dual, method=method)


def low_pers_filter(
    signal: ArrayLike | GraphWithFaces,
    epsilon,
    *,
    order: str = "01",
    method: str = "auto",
    size_range: Sequence[float] | None = None,
) -> np.ndarray:
    """Remove low-persistence features from an array or a graph signal.

    Parameters
    ----------
    signal
        An array (1-D, 2-D or 3-D), or a :class:`~topapprox.graph.GraphWithFaces`
        carrying the signal to filter.
    epsilon
        A single threshold, a ``(eps_H0, eps_H1)`` pair, or a mapping keyed by
        ``0``/``1``.
    order
        Which homological degrees to filter and in which order, e.g. ``"01"``
        for components first and cycles second, or ``"0"`` for components only.
    size_range
        Optional ``(min, max)`` basin size; see :meth:`MergeTree.low_pers_filter`.
    """
    steps = tuple(order)
    if any(step not in "01" for step in steps):
        raise ValueError(f"order must consist of '0' and '1' only, got {order!r}.")

    if isinstance(signal, GraphWithFaces):
        template, current = signal, signal.signal
    else:
        template, current = None, np.ascontiguousarray(signal, dtype=np.float64)

    for step, threshold in zip(steps, _thresholds(epsilon, steps)):
        filtered = _make_filter(current, dual=step == "1", method=method, template=template)
        current = filtered.low_pers_filter(threshold, size_range=size_range)
    return current


def persistence_diagram(
    signal: ArrayLike | GraphWithFaces,
    *,
    method: str = "auto",
) -> list[NDArray[np.float64]]:
    """Persistence diagrams of an array or a graph signal, as ``(n, 2)`` arrays.

    The first diagram is always ``H0`` (essential class included).  The second is
    the top-degree one: ``H1`` for graphs and 1-D/2-D arrays, ``H2`` for 3-D
    arrays.
    """
    if isinstance(signal, GraphWithFaces):
        return GraphFilter(signal, method=method).diagrams()
    array = np.ascontiguousarray(signal, dtype=np.float64)
    primal = ImageFilter(array, method=method).diagram(reduced=False)
    dual = ImageFilter(array, dual=True, method=method).diagram(reduced=True)
    return [primal[:, :2], dual[:, :2]]
