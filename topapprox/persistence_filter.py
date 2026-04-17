"""Unified persistence-filter orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .filter_graph import TopologicalFilterGraph
from .filter_image import TopologicalFilterImage


@dataclass(frozen=True)
class Filtered:
    """Container for a filtered signal snapshot."""

    signal: np.ndarray
    iteration_order: tuple[str, ...]


@dataclass(frozen=True)
class _RunKey:
    iteration_order: tuple[str, ...]
    epsilon_sequence: tuple[float, ...]
    method: str
    recursive: bool
    iter_vertex: bool
    is_triangulated: bool
    size_range: tuple[float, float] | None


class PersistenceFilter:
    """Apply repeated low-persistence filtering across homology dimensions."""

    def __init__(self) -> None:
        self.type = None
        self.signal = None
        self.dim = None
        self.faces = None
        self.holes = None
        self.edges = None

        self.bht = []
        self.filtered = []
        self.filtered_type = []

        self._filtered_cache: dict[_RunKey, np.ndarray] = {}
        self._bht_cache: dict[_RunKey, object] = {}
        self._latest_by_order: dict[tuple[str, ...], _RunKey] = {}
        self._last_key: _RunKey | None = None

    def _reset_cache(self) -> None:
        self.bht = []
        self.filtered = []
        self.filtered_type = []
        self._filtered_cache = {}
        self._bht_cache = {}
        self._latest_by_order = {}
        self._last_key = None

    def load_signal(self, signal):
        """Load an array or graph-with-faces signal and clear cached runs."""

        if isinstance(signal, np.ndarray):
            self.signal = np.array(signal, copy=True)
            self.type = "array"
            self.dim = self.signal.ndim
            self.faces = None
            self.holes = None
            self.edges = None
            self._reset_cache()
            return self

        if isinstance(signal, (list, tuple)):
            if len(signal) not in (3, 4):
                raise TypeError(
                    "Graph input must be [faces, holes, signal] or [faces, holes, signal, edges]."
                )

            faces, holes, values = signal[:3]
            edges = signal[3] if len(signal) == 4 else None
            if not isinstance(faces, list) or not isinstance(holes, list):
                raise TypeError("faces and holes must be lists.")
            if not isinstance(values, np.ndarray):
                raise TypeError("Graph signal values must be provided as a numpy array.")

            self.faces = faces
            self.holes = holes
            self.edges = edges
            self.signal = np.array(values, copy=True).ravel()
            self.type = "graph"
            self.dim = 1
            self._reset_cache()
            return self

        raise TypeError(
            "signal must be a numpy array, or [faces, holes, signal] "
            "(optionally with edges as a fourth element)."
        )

    @staticmethod
    def _normalize_iteration_order(iteration_order) -> tuple[str, ...]:
        items = list(iteration_order) if isinstance(iteration_order, str) else [str(item) for item in iteration_order]
        if any(item not in {"0", "1"} for item in items):
            raise ValueError("iteration_order should contain only '0' and '1'.")
        return tuple(items)

    @staticmethod
    def _epsilon_for_step(epsilon, step: str) -> float:
        if np.isscalar(epsilon):
            return float(epsilon)
        if isinstance(epsilon, dict):
            if step in epsilon:
                return float(epsilon[step])
            key = int(step)
            if key in epsilon:
                return float(epsilon[key])
            raise KeyError(f"Missing epsilon for homology {step}.")
        if isinstance(epsilon, Sequence):
            if len(epsilon) != 2:
                raise ValueError("epsilon sequences must contain exactly two values.")
            return float(epsilon[int(step)])
        raise TypeError(
            "epsilon should be a scalar, a dict keyed by 0/1 (or '0'/'1'), "
            "or a length-2 sequence."
        )

    @staticmethod
    def _normalize_size_range(size_range) -> tuple[float, float] | None:
        if size_range is None:
            return None
        if len(size_range) != 2:
            raise ValueError("size_range must contain exactly two numbers.")
        return float(size_range[0]), float(size_range[1])

    def _make_run_key(
        self,
        *,
        iteration_order: tuple[str, ...],
        epsilon_sequence: tuple[float, ...],
        method: str,
        recursive: bool,
        iter_vertex: bool,
        is_triangulated: bool,
        size_range: tuple[float, float] | None,
    ) -> _RunKey:
        return _RunKey(
            iteration_order=iteration_order,
            epsilon_sequence=epsilon_sequence,
            method=method,
            recursive=recursive,
            iter_vertex=iter_vertex,
            is_triangulated=is_triangulated,
            size_range=size_range,
        )

    def _build_filter(
        self,
        current_signal: np.ndarray,
        *,
        dual: bool,
        method: str,
        recursive: bool,
        iter_vertex: bool,
        is_triangulated: bool,
    ):
        if self.type == "array":
            return TopologicalFilterImage(
                np.array(current_signal, copy=True),
                method=method,
                dual=dual,
                recursive=recursive,
                iter_vertex=iter_vertex,
            )

        if self.type == "graph":
            graph_filter = TopologicalFilterGraph(
                method=method,
                dual=dual,
                recursive=recursive,
                is_triangulated=is_triangulated,
            )
            graph_filter.compute_gwf(
                self.faces,
                self.holes,
                np.array(current_signal, copy=True).ravel(),
                E=self.edges,
            )
            return graph_filter

        raise RuntimeError("No signal has been loaded. Call load_signal first.")

    def low_pers_filter(
        self,
        epsilon,
        *,
        iteration_order="01",
        method="cpp",
        bht_method="python",
        recursive=True,
        iter_vertex=True,
        is_triangulated=False,
        size_range=None,
        return_sequence=False,
    ):
        del bht_method
        if self.signal is None:
            raise RuntimeError("No signal has been loaded. Use load_signal(...) first.")

        order = self._normalize_iteration_order(iteration_order)
        if not order:
            empty = np.array(self.signal, copy=True)
            return (empty, []) if return_sequence else empty

        normalized_size_range = self._normalize_size_range(size_range)
        epsilon_sequence = tuple(self._epsilon_for_step(epsilon, step) for step in order)
        sequence: list[Filtered] = []
        full_key = self._make_run_key(
            iteration_order=order,
            epsilon_sequence=epsilon_sequence,
            method=method,
            recursive=recursive,
            iter_vertex=iter_vertex,
            is_triangulated=is_triangulated,
            size_range=normalized_size_range,
        )
        if full_key in self._filtered_cache:
            self._last_key = full_key
            self._latest_by_order[order] = full_key
            result = np.array(self._filtered_cache[full_key], copy=True)
            if return_sequence:
                return result, sequence
            return result

        prefix_len = len(order)
        while prefix_len > 0:
            prefix_key = self._make_run_key(
                iteration_order=order[:prefix_len],
                epsilon_sequence=epsilon_sequence[:prefix_len],
                method=method,
                recursive=recursive,
                iter_vertex=iter_vertex,
                is_triangulated=is_triangulated,
                size_range=normalized_size_range,
            )
            if prefix_key in self._filtered_cache:
                break
            prefix_len -= 1

        if prefix_len == 0:
            current = np.array(self.signal, copy=True)
        else:
            current = np.array(self._filtered_cache[prefix_key], copy=True)

        for index in range(prefix_len, len(order)):
            dual = order[index] == "1"
            filter_obj = self._build_filter(
                current,
                dual=dual,
                method=method,
                recursive=recursive,
                iter_vertex=iter_vertex,
                is_triangulated=is_triangulated,
            )
            current = np.array(
                filter_obj.low_pers_filter(epsilon_sequence[index], size_range=normalized_size_range),
                copy=True,
            )
            key = self._make_run_key(
                iteration_order=order[: index + 1],
                epsilon_sequence=epsilon_sequence[: index + 1],
                method=method,
                recursive=recursive,
                iter_vertex=iter_vertex,
                is_triangulated=is_triangulated,
                size_range=normalized_size_range,
            )
            self._filtered_cache[key] = np.array(current, copy=True)
            self._bht_cache[key] = filter_obj.bht
            self._latest_by_order[key.iteration_order] = key
            self._last_key = key

            if key.iteration_order not in self.filtered_type:
                self.filtered_type.append(key.iteration_order)
                self.filtered.append(np.array(current, copy=True))
                self.bht.append(filter_obj.bht)

            sequence.append(Filtered(signal=np.array(current, copy=True), iteration_order=key.iteration_order))

        result_key = self._latest_by_order[order]
        self._last_key = result_key
        result = np.array(self._filtered_cache[result_key], copy=True)
        if return_sequence:
            return result, sequence
        return result

    def get_filtered(self, iteration_order: Optional[Iterable[str]] = None) -> np.ndarray:
        if self.signal is None:
            raise RuntimeError("No signal has been loaded. Use load_signal(...) first.")
        key = self._last_key if iteration_order is None else self._latest_by_order.get(self._normalize_iteration_order(iteration_order))
        if key is None or key not in self._filtered_cache:
            raise KeyError("No cached filtered signal found for the requested iteration order.")
        return np.array(self._filtered_cache[key], copy=True)

    def get_BHT(self, iteration_order: Optional[Iterable[str]] = None):
        key = self._last_key if iteration_order is None else self._latest_by_order.get(self._normalize_iteration_order(iteration_order))
        if key is None or key not in self._bht_cache:
            raise KeyError("No cached BHT found for the requested iteration order.")
        return self._bht_cache[key]
