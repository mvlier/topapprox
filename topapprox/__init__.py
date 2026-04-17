"""Public package interface for topapprox."""

from __future__ import annotations

from ._backends import available_backends
from ._version import __version__
from .filter_graph import TopologicalFilterGraph
from .filter_image import TopologicalFilterImage
from .gwf import GraphWithFaces
from .persistence_filter import Filtered, PersistenceFilter
from .tools import Tools, tools
from .visualize import plot_basin_graph, plot_basin_image

ImageFilter = TopologicalFilterImage
GraphFilter = TopologicalFilterGraph

__all__ = [
    "__version__",
    "Filtered",
    "GraphFilter",
    "GraphWithFaces",
    "ImageFilter",
    "PersistenceFilter",
    "Tools",
    "TopologicalFilterGraph",
    "TopologicalFilterImage",
    "available_backends",
    "plot_basin_graph",
    "plot_basin_image",
    "tools",
]
