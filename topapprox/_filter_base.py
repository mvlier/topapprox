"""Shared lifecycle for topological filters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ._backends import normalize_link_reduce_result, resolve_backend
from .bht import BasinHierarchyTree


class BaseTopologicalFilter(ABC):
    """Common tree construction and post-processing logic."""

    def __init__(self, *, method: str, dual: bool, recursive: bool, dimensions: int = 2) -> None:
        self.dual = bool(dual)
        self.recursive = bool(recursive)
        self.backend = resolve_backend(method, dimensions=dimensions)
        self.method = self.backend.name
        self.bht: BasinHierarchyTree | None = None

    def _build_tree(
        self,
        birth: NDArray[np.floating],
        raw_result,
    ) -> BasinHierarchyTree:
        result = normalize_link_reduce_result(raw_result)
        tree = BasinHierarchyTree(recursive=self.recursive, dual=self.dual)
        tree.parent = result.parent
        tree.children = result.children
        tree.root = result.root
        tree.linking_vertex = result.linking_vertex
        tree.persistent_children = result.persistent_children
        tree.positive_pers = result.positive_pers
        tree.birth = np.asarray(birth)
        return tree

    def _ensure_bht(self) -> BasinHierarchyTree:
        if self.bht is None or self.bht.children is None:
            self.bht = self._build_bht()
        return self.bht

    @abstractmethod
    def _build_bht(self) -> BasinHierarchyTree:
        """Create the basin hierarchy tree for this filter."""

    @abstractmethod
    def _format_output(self, values: NDArray[np.floating]) -> np.ndarray:
        """Convert flat tree values back into the public return shape."""

    def low_pers_filter(self, epsilon: float, *, size_range=None) -> np.ndarray:
        tree = self._ensure_bht()
        if size_range is None:
            filtered = tree._low_pers_filter(float(epsilon))
        else:
            filtered = tree._lpf_size_filter(float(epsilon), size_range=size_range)
        return self._format_output(np.asarray(filtered))

    def _update_BHT(self):
        """Compatibility wrapper for eagerly building the basin tree."""

        self.bht = self._build_bht()
        return self.bht

    def get_BHT(self, *, with_children: bool = False):
        return self._ensure_bht()._get_BHT(with_children=with_children)

    def get_persistence(self, *, reduced: bool = True) -> np.ndarray:
        return self._ensure_bht().get_persistence(reduced=reduced)

    def basin_map(self, epsilon: float = 0.0) -> np.ndarray:
        labels = self._ensure_bht().basin_map(float(epsilon))
        return self._format_output(np.asarray(labels))
