"""Persistent-homology-based filtering of scalar signals.

Low-persistence filtering removes the topological features of a signal that are
small relative to a threshold ``epsilon``, leaving the significant ones and the
signal itself within ``epsilon`` in the supremum norm.

Quick start::

    import numpy as np
    import topapprox as ta

    image = np.random.rand(256, 256)

    ta.low_pers_filter(image, 0.1)                 # H0 then H1, in one call
    ta.ImageFilter(image).low_pers_filter(0.1)     # keep the merge tree for reuse
    pd0, pd1 = ta.persistence_diagram(image)

The building blocks are :class:`GraphWithFaces` (a signal on an embedded graph),
:class:`ImageFilter` / :class:`GraphFilter` (filtration front ends) and
:class:`MergeTree` (the basin hierarchy, where the actual work happens).
"""

from __future__ import annotations

from ._version import __version__
from .backends import available_backends, resolve_backend
from .filters import (
    GraphFilter,
    ImageFilter,
    TopologicalFilter,
    low_pers_filter,
    persistence_diagram,
)
from .graph import GraphWithFaces, random_graph_with_faces
from .merge_tree import DIAGRAM_COLUMNS, MergeTree
from .plotting import draw_graph, plot_basin_graph, plot_basin_image

__all__ = [
    "DIAGRAM_COLUMNS",
    "GraphFilter",
    "GraphWithFaces",
    "ImageFilter",
    "MergeTree",
    "TopologicalFilter",
    "__version__",
    "available_backends",
    "draw_graph",
    "low_pers_filter",
    "persistence_diagram",
    "plot_basin_graph",
    "plot_basin_image",
    "random_graph_with_faces",
    "resolve_backend",
]
