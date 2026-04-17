"""Backend discovery and link-reduce result normalization."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Literal, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray

BackendName = Literal["python", "numba", "cpp"]
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
RawLinkReduceResult = tuple[
    NDArray[np.integer],
    Sequence[Sequence[int]],
    int,
    NDArray[np.integer],
    Sequence[Sequence[int]],
    NDArray[np.integer],
]


@dataclass(frozen=True, slots=True)
class LinkReduceResult:
    """Normalized output of a link-reduce backend."""

    parent: NDArray[np.int64]
    children: list[list[int]]
    root: int
    linking_vertex: NDArray[np.int64]
    persistent_children: list[list[int]]
    positive_pers: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LinkReduceBackend:
    """Callable bundle for a backend implementation."""

    name: BackendName
    reduce_edges: Callable[[FloatArray, IntArray, float], RawLinkReduceResult]
    reduce_grid_2d: Callable[[FloatArray, tuple[int, int], bool], RawLinkReduceResult]
    reduce_grid_3d: Callable[[FloatArray, tuple[int, int, int], bool], RawLinkReduceResult] | None = None


def _to_int_lists(values: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(item) for item in group] for group in values]


def normalize_link_reduce_result(raw: RawLinkReduceResult) -> LinkReduceResult:
    """Convert backend-specific outputs into plain Python / NumPy objects."""

    parent, children, root, linking_vertex, persistent_children, positive_pers = raw
    return LinkReduceResult(
        parent=np.asarray(parent, dtype=np.int64),
        children=_to_int_lists(children),
        root=int(root),
        linking_vertex=np.asarray(linking_vertex, dtype=np.int64),
        persistent_children=_to_int_lists(persistent_children),
        positive_pers=np.asarray(positive_pers, dtype=np.int64),
    )


def _python_backend() -> LinkReduceBackend:
    module = import_module(".link_reduce", package=__package__)
    return LinkReduceBackend(
        name="python",
        reduce_edges=module._link_reduce,
        reduce_grid_2d=module._link_reduce_vertices,
        reduce_grid_3d=None,
    )


def _numba_backend() -> LinkReduceBackend:
    module = import_module(".link_reduce_numba", package=__package__)
    return LinkReduceBackend(
        name="numba",
        reduce_edges=module.link_reduce_wrapper,
        reduce_grid_2d=module.link_reduce_vertices_wrapper,
        reduce_grid_3d=None,
    )


def _cpp_backend() -> LinkReduceBackend:
    module = import_module(".link_reduce_cpp", package=__package__)
    return LinkReduceBackend(
        name="cpp",
        reduce_edges=module._link_reduce_cpp,
        reduce_grid_2d=module._link_reduce_vertices_cpp,
        reduce_grid_3d=getattr(module, "_link_reduce_vertices_cpp_3D", None),
    )


def resolve_backend(method: str, *, dimensions: int = 2) -> LinkReduceBackend:
    """Resolve a backend name with graceful fallbacks."""

    requested = method.lower()
    if requested not in {"python", "numba", "cpp"}:
        raise ValueError(f"Unknown method: {method!r}")

    if dimensions == 3:
        if requested != "cpp":
            warnings.warn(
                "3D filtering is implemented only for the C++ backend; "
                "falling back to method='cpp'.",
                UserWarning,
                stacklevel=2,
            )
        backend = _cpp_backend()
        if backend.reduce_grid_3d is None:
            raise RuntimeError("The compiled C++ backend does not expose 3D grid filtering.")
        return backend

    if requested == "python":
        return _python_backend()

    if requested == "numba":
        try:
            return _numba_backend()
        except Exception as exc:  # pragma: no cover - exercised when numba is unavailable.
            warnings.warn(
                f"{exc}\nFalling back to method='python' because the numba backend "
                "could not be loaded.",
                UserWarning,
                stacklevel=2,
            )
            return _python_backend()

    try:
        return _cpp_backend()
    except Exception as exc:  # pragma: no cover - exercised when the extension is unavailable.
        warnings.warn(
            f"{exc}\nFalling back to method='python' because the C++ backend "
            "could not be loaded.",
            UserWarning,
            stacklevel=2,
        )
        return _python_backend()


def available_backends(*, dimensions: int = 2) -> tuple[BackendName, ...]:
    """Return importable backend names for the current environment."""

    available: list[BackendName] = ["python"]
    if dimensions == 3:
        try:
            backend = _cpp_backend()
        except Exception:
            return tuple()
        return ("cpp",) if backend.reduce_grid_3d is not None else tuple()

    try:
        _numba_backend()
    except Exception:
        pass
    else:
        available.append("numba")

    try:
        _cpp_backend()
    except Exception:
        pass
    else:
        available.append("cpp")

    return tuple(available)
