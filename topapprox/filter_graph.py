"""Graph-with-faces topological filtering."""

from __future__ import annotations

import warnings

import numpy as np

from ._filter_base import BaseTopologicalFilter
from ._grid import boundary_from_array_shape, faces_from_array_shape
from .gwf import GraphWithFaces


class TopologicalFilterGraph(BaseTopologicalFilter):
    """Low-persistence filtering for graph signals with an embedding."""

    def __init__(
        self,
        input=None,
        *,
        method: str = "cpp",
        bht_method: str = "python",
        dual: bool = False,
        recursive: bool = True,
        is_triangulated: bool = False,
        gwf: GraphWithFaces | None = None,
    ) -> None:
        if bht_method != "python":
            warnings.warn(
                "The dedicated C++ BHT implementation was removed; using the Python "
                "BasinHierarchyTree implementation instead.",
                UserWarning,
                stacklevel=2,
            )

        self.shape: tuple[int, ...] | None = None
        self.signal: np.ndarray | None = None
        self.gwf = gwf
        self.is_triangulated = bool(is_triangulated)
        self.diagram: list[np.ndarray | None] = [None, None]

        super().__init__(method=method, dual=dual, recursive=recursive, dimensions=2)

        if self.gwf is not None:
            self.signal = np.asarray(self.gwf.signal[: self.gwf.n_nodes], dtype=float).copy()

        if isinstance(input, np.ndarray):
            self.from_array(input)
        elif input is not None:
            raise TypeError("input must be a numpy array or None.")

    def compute_gwf(self, F: list, H: list, signal: np.ndarray, E=None) -> None:
        self.signal = np.asarray(signal).ravel().copy()
        self.gwf = GraphWithFaces(
            F=F,
            H=H,
            E=E,
            signal=self.signal,
            compute="dual" if self.dual else "normal",
            is_triangulated=self.is_triangulated,
        )
        self.diagram = [None, None]
        self.bht = None

    def from_array(self, img: np.ndarray) -> None:
        array = np.asarray(img)
        if array.ndim not in {1, 2}:
            raise ValueError("Only 1D and 2D arrays can be converted into a GraphWithFaces.")

        self.shape = array.shape
        self.signal = array.ravel().copy()
        faces = faces_from_array_shape(array.shape)
        holes = [boundary_from_array_shape(array.shape)]
        self.compute_gwf(faces, holes, self.signal)

    def _build_bht(self):
        if self.gwf is None:
            raise RuntimeError("GraphWithFaces is not initialized. Use compute_gwf or from_array first.")

        birth = -self.gwf.signal if self.dual else self.gwf.signal
        edges = self.gwf.dual_edges if self.dual else self.gwf.edges
        raw = self.backend.reduce_edges(birth, edges, 0.0)
        return self._build_tree(birth, raw)

    def _format_output(self, values: np.ndarray) -> np.ndarray:
        if self.signal is None:
            raise RuntimeError("Graph signal is not initialized.")

        n_vertices = self.signal.shape[0]
        output = np.asarray(values)[:n_vertices]
        if self.dual:
            output = -output
        if self.shape is not None:
            return output.reshape(self.shape)
        return output

    @staticmethod
    def _as_birth_death(diagram: np.ndarray) -> np.ndarray:
        diagram = np.asarray(diagram)
        if diagram.ndim == 2 and diagram.shape[0] > 0:
            return diagram[:, :2]
        return np.empty((0, 2), dtype=float)

    def _clone_with_dual(self, *, dual: bool) -> "TopologicalFilterGraph":
        if self.gwf is None or self.signal is None:
            raise RuntimeError("GraphWithFaces is not initialized. Use compute_gwf or from_array first.")

        cloned = TopologicalFilterGraph(
            method=self.method,
            dual=dual,
            recursive=self.recursive,
            is_triangulated=self.is_triangulated,
        )
        cloned.shape = self.shape
        cloned.compute_gwf(self.gwf.F, self.gwf.H, self.signal, E=self.gwf.edges)
        return cloned

    def get_diagram(self) -> list[np.ndarray]:
        if self.gwf is None:
            raise RuntimeError("GraphWithFaces is not initialized. Use compute_gwf or from_array first.")

        if self.diagram[0] is None:
            source = self if not self.dual else self._clone_with_dual(dual=False)
            self.diagram[0] = self._as_birth_death(source.get_persistence(reduced=False))

        if self.diagram[1] is None:
            source = self if self.dual else self._clone_with_dual(dual=True)
            self.diagram[1] = self._as_birth_death(source.get_persistence(reduced=True))

        return [np.asarray(self.diagram[0]), np.asarray(self.diagram[1])]

    def _low_pers_filter(self, epsilon: float = 0.0, dual: bool = False, verbose: bool = False):
        del verbose
        if dual == self.dual:
            filtered = self.low_pers_filter(epsilon)
        else:
            filtered = self._clone_with_dual(dual=dual).low_pers_filter(epsilon)

        values = np.asarray(filtered).ravel()
        return {index: values[index] for index in range(values.shape[0])}
