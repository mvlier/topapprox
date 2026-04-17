"""Image-oriented topological filtering."""

from __future__ import annotations

import warnings

import numpy as np

from ._filter_base import BaseTopologicalFilter
from ._grid import sorted_grid_edges


class TopologicalFilterImage(BaseTopologicalFilter):
    """Low-persistence filtering for 1D/2D/3D arrays."""

    def __init__(
        self,
        img,
        *,
        method: str = "cpp",
        bht_method: str = "python",
        dual: bool = False,
        recursive: bool = True,
        iter_vertex: bool = True,
    ) -> None:
        image = np.asarray(img)
        if image.ndim not in {1, 2, 3}:
            raise ValueError("img must be a 1D, 2D, or 3D numpy-compatible array.")

        self._input_was_1d = image.ndim == 1
        if self._input_was_1d:
            image = image.reshape(1, -1)

        if bht_method != "python":
            warnings.warn(
                "The dedicated C++ BHT implementation was removed; using the Python "
                "BasinHierarchyTree implementation instead.",
                UserWarning,
                stacklevel=2,
            )

        self.image = np.array(image, copy=True)
        self.shape = self.image.shape
        self.iter_vertex = bool(iter_vertex)
        self.birth = self.image.astype(float, copy=False).ravel().copy()
        self._sorted_edges: np.ndarray | None = None

        if dual:
            extra = np.array([-np.inf], dtype=self.birth.dtype)
            self.bht_birth = np.concatenate((-self.birth, extra))
        else:
            self.bht_birth = self.birth.copy()

        super().__init__(method=method, dual=dual, recursive=recursive, dimensions=self.image.ndim)

    def _build_bht(self):
        if self.image.ndim == 3:
            if self.backend.reduce_grid_3d is None:
                raise RuntimeError("The selected backend does not support 3D image filtering.")
            raw = self.backend.reduce_grid_3d(self.bht_birth, self.shape, self.dual)
            return self._build_tree(self.bht_birth, raw)

        if self.iter_vertex:
            raw = self.backend.reduce_grid_2d(self.bht_birth, self.shape, self.dual)
            return self._build_tree(self.bht_birth, raw)

        if self._sorted_edges is None:
            birth_for_sort = self.bht_birth if self.dual else self.birth
            self._sorted_edges = sorted_grid_edges(birth_for_sort, self.shape, dual=self.dual)
        raw = self.backend.reduce_edges(self.bht_birth, self._sorted_edges, 0.0)
        return self._build_tree(self.bht_birth, raw)

    def _format_output(self, values: np.ndarray) -> np.ndarray:
        output = np.asarray(values)
        if self.dual:
            output = -output[:-1]
        output = output.reshape(self.shape)
        if self._input_was_1d:
            return output.reshape(-1)
        return output
