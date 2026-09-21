"""Every backend and every route must build exactly the same merge tree.

The kernels follow one total order -- by birth value, then by vertex index --
and merge all components meeting at a vertex into the oldest of them.  That
makes the tree a function of the filtration alone, so the C++ grid sweep, the
NumPy edge list fed to any kernel, and the graph-with-faces route all agree
bit for bit, plateaus included.
"""

from __future__ import annotations

import numpy as np
import pytest

import topapprox as ta
from topapprox.backends import BACKEND_NAMES, resolve_backend

SHAPES = [(9,), (1, 8), (5, 6), (6, 7), (4, 4, 4)]


def assert_same_tree(left, right, label=""):
    assert np.array_equal(left.parent, right.parent), f"parent differs {label}"
    assert np.array_equal(left.linking_vertex, right.linking_vertex), f"link differs {label}"
    assert left.root == right.root, f"root differs {label}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dual", [False, True])
def test_backends_agree(shape, dual):
    # Integer values create plenty of plateaus, where tie-breaking shows up.
    array = np.random.default_rng(0).integers(0, 4, size=shape).astype(float)
    reference = ta.ImageFilter(array, dual=dual, method="python").tree
    for name in ta.available_backends():
        assert_same_tree(ta.ImageFilter(array, dual=dual, method=name).tree, reference, name)


@pytest.mark.parametrize("shape", [(5, 6), (6, 7)])
def test_grid_and_graph_routes_agree(shape):
    array = np.random.default_rng(1).integers(0, 4, size=shape).astype(float)
    assert_same_tree(ta.GraphFilter.from_array(array).tree, ta.ImageFilter(array).tree)


def test_available_backends_are_loadable():
    backends = ta.available_backends()
    assert "python" in backends
    assert set(backends) <= set(BACKEND_NAMES)
    for name in backends:
        assert resolve_backend(name).name == name
    assert resolve_backend("auto").name in backends


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        resolve_backend("fortran")
