"""Backend discovery and the uniform kernel interface.

A backend provides one primitive: build a merge tree from a list of edges.  The
C++ backend adds a specialised grid sweep that skips materialising the edge
list; the others get the same result by building it with NumPy.  Everything
else about a merge tree is derived in :mod:`topapprox.merge_tree`.

All backends follow the same total order and the same merge rule, so they build
identical trees -- see :mod:`topapprox._reduce_python` for the reference.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .grids import sweep_edges

__all__ = ["Backend", "BackendName", "available_backends", "resolve_backend"]

BackendName = Literal["python", "numba", "cpp"]
BACKEND_NAMES: tuple[BackendName, ...] = ("python", "numba", "cpp")
PREFERENCE: tuple[BackendName, ...] = ("cpp", "numba", "python")

#: ``(parent, linking_vertex, root)``
MergeTreeArrays = tuple[NDArray[np.int64], NDArray[np.int64], int]


class Backend:
    """A named merge-tree kernel, with argument normalisation around it."""

    def __init__(
        self,
        name: BackendName,
        edge_kernel: Callable[..., MergeTreeArrays],
        grid_kernel: Callable[..., MergeTreeArrays] | None = None,
    ) -> None:
        self.name = name
        self._edge_kernel = edge_kernel
        self._grid_kernel = grid_kernel

    def __repr__(self) -> str:
        return f"Backend({self.name!r})"

    def reduce_edges(self, birth, edges) -> MergeTreeArrays:
        """Merge tree of ``birth`` over ``edges``, rows ``(passive, active)``."""
        return self._edge_kernel(_as_values(birth), _as_edges(edges))

    def reduce_grid(self, birth, shape, *, dual: bool) -> MergeTreeArrays:
        """Merge tree of ``birth`` sampled on a regular grid of the given shape."""
        values = _as_values(birth)
        shape = tuple(int(extent) for extent in shape)
        if self._grid_kernel is None:
            return self._edge_kernel(values, sweep_edges(values, shape, dual=dual))
        order = np.argsort(values[: int(np.prod(shape))], kind="stable")
        return self._grid_kernel(values, np.ascontiguousarray(order, dtype=np.int64), shape, dual)


def _as_values(birth) -> NDArray[np.float64]:
    # The kernels bind typed, C-contiguous arrays; nanobind does not convert
    # dtypes silently, so the conversion happens here, once, for all backends.
    return np.ascontiguousarray(birth, dtype=np.float64)


def _as_edges(edges) -> NDArray[np.int64]:
    array = np.ascontiguousarray(edges, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected edges of shape (m, 2), got {array.shape}.")
    return array


def _load(name: BackendName) -> Backend:
    if name == "cpp":
        module = import_module("._reduce_cpp", __package__)
        return Backend("cpp", module.reduce_edges, module.reduce_grid)
    module = import_module("._reduce_numba" if name == "numba" else "._reduce_python", __package__)
    return Backend(name, module.reduce_edges)


def available_backends() -> tuple[BackendName, ...]:
    """Backend names that can be imported in this environment."""
    usable = []
    for name in BACKEND_NAMES:
        try:
            _load(name)
        except Exception:  # pragma: no cover - depends on the installation
            continue
        usable.append(name)
    return tuple(usable)


def resolve_backend(method: str = "auto") -> Backend:
    """Load a backend by name, or the fastest importable one for ``"auto"``.

    A named backend that cannot be imported falls back to ``"python"`` with a
    warning, so code stays runnable without the compiled extension.
    """
    requested = str(method).lower()
    if requested == "auto":
        for name in PREFERENCE:
            try:
                return _load(name)
            except Exception:  # pragma: no cover - depends on the installation
                continue
        raise RuntimeError("No topapprox backend could be loaded.")

    if requested not in BACKEND_NAMES:
        raise ValueError(f"Unknown method {method!r}; expected one of {BACKEND_NAMES} or 'auto'.")

    try:
        return _load(requested)
    except Exception as error:  # pragma: no cover - depends on the installation
        if requested == "python":
            raise
        warnings.warn(
            f"The {requested!r} backend is unavailable ({error}); falling back to 'python'.",
            UserWarning,
            stacklevel=2,
        )
        return _load("python")
