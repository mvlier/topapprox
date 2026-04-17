import numpy as np
import pytest

from topapprox import GraphFilter, GraphWithFaces, available_backends
from topapprox.persistence import get_PD_gwf


def _simple_graph():
    faces = [[0, 1, 2, 3]]
    holes = [[0, 1, 2, 3]]
    signal = np.array([0.0, 1.0, 0.5, 0.2])
    return faces, holes, signal


def _assert_birth_death_array(pd):
    assert isinstance(pd, np.ndarray)
    assert pd.ndim == 2
    assert pd.shape[1] == 2


@pytest.mark.parametrize("method", available_backends())
def test_get_diagram_shapes_and_cache(method):
    faces, holes, signal = _simple_graph()

    tfg = GraphFilter(method=method)
    tfg.compute_gwf(F=faces, H=holes, signal=signal)

    pd0, pd1 = tfg.get_diagram()
    _assert_birth_death_array(pd0)
    _assert_birth_death_array(pd1)

    # Repeated call should hit cache and remain stable.
    pd0_cached, pd1_cached = tfg.get_diagram()
    assert np.array_equal(pd0, pd0_cached)
    assert np.array_equal(pd1, pd1_cached)


@pytest.mark.parametrize("method", available_backends())
def test_get_diagram_dual_initialized(method):
    faces, holes, signal = _simple_graph()

    tfg = GraphFilter(method=method, dual=True)
    tfg.compute_gwf(F=faces, H=holes, signal=signal)
    pd0, pd1 = tfg.get_diagram()

    _assert_birth_death_array(pd0)
    _assert_birth_death_array(pd1)


@pytest.mark.parametrize("method", available_backends())
def test_get_pd_gwf_matches_filter_graph_api(method):
    faces, holes, signal = _simple_graph()

    tfg = GraphFilter(method=method)
    tfg.compute_gwf(F=faces, H=holes, signal=signal)
    pd_api = tfg.get_diagram()
    pd_fn = get_PD_gwf(faces, holes, signal, method=method)

    assert len(pd_api) == len(pd_fn) == 2
    for x, y in zip(pd_api, pd_fn):
        _assert_birth_death_array(x)
        _assert_birth_death_array(y)
        assert np.array_equal(x, y)


@pytest.mark.parametrize("method", available_backends())
def test_graph_filter_accepts_precomputed_gwf(method):
    faces, holes, signal = _simple_graph()
    gwf = GraphWithFaces(F=faces, H=holes, signal=signal, compute="normal")

    graph_filter = GraphFilter(gwf=gwf, method=method)
    filtered = graph_filter.low_pers_filter(0.5)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == signal.shape
