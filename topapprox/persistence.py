"""Convenience helpers for persistence-diagram computation."""

from __future__ import annotations

import numpy as np

from .filter_graph import TopologicalFilterGraph


def get_PD_gwf(F, H, signal, E=None, *, method="cpp", bht_method="python", is_triangulated=False):
    """Return the 0D and 1D persistence diagrams of a graph-with-faces signal."""

    graph_filter = TopologicalFilterGraph(
        method=method,
        bht_method=bht_method,
        dual=False,
        is_triangulated=is_triangulated,
    )
    graph_filter.compute_gwf(F=F, H=H, E=E, signal=np.asarray(signal))
    pd0, pd1 = graph_filter.get_diagram()
    return [np.asarray(pd0), np.asarray(pd1)]
