"""Graph-with-faces construction, filtering and persistence diagrams."""

from __future__ import annotations

import random

import numpy as np
import pytest

import topapprox as ta
from topapprox.grids import grid_boundary, grid_faces


def square():
    return [[0, 1, 2, 3]], [[0, 1, 2, 3]], np.array([0.0, 1.0, 0.5, 0.2])


def assert_birth_death(diagram):
    assert isinstance(diagram, np.ndarray)
    assert diagram.ndim == 2 and diagram.shape[1] == 2
    assert np.all(diagram[:, 1] >= diagram[:, 0])


def test_skeleton_from_cells():
    faces, holes, signal = square()
    graph = ta.GraphWithFaces(faces, holes, signal)
    assert graph.n_vertices == 4
    assert sorted(map(tuple, graph.edges.tolist())) != []
    undirected = {tuple(sorted(edge)) for edge in graph.edges.tolist()}
    assert undirected == {(0, 1), (1, 2), (2, 3), (0, 3)}
    # One apex per face plus one per hole, holes sitting at +inf.
    assert graph.dual_values.shape[0] == 4 + len(faces) + len(holes)
    assert np.isinf(graph.dual_values[-len(holes) :]).all()


def test_explicit_edges_are_deduplicated():
    faces, holes, signal = square()
    graph = ta.GraphWithFaces(faces, holes, signal, edges=[[1, 0], [0, 1], [2, 3], [3, 3]])
    assert {tuple(sorted(e)) for e in graph.edges.tolist()} == {(0, 1), (2, 3)}

    with pytest.raises(ValueError):
        ta.GraphWithFaces(faces, holes, signal, edges=[[0, 99]])


def test_with_signal_keeps_the_complex():
    faces, holes, signal = square()
    graph = ta.GraphWithFaces(faces, holes, signal)
    other = graph.with_signal(signal[::-1])
    assert np.array_equal(
        np.sort(graph.edges, axis=None), np.sort(other.edges, axis=None)
    )
    assert np.array_equal(other.signal, signal[::-1])


def test_grid_boundary_is_a_cycle():
    for shape in [(5,), (1, 5), (5, 1), (3, 3), (4, 6)]:
        cycle = grid_boundary(shape)
        n_cols = shape[-1] if len(shape) == 2 else shape[0]
        coords = [(v // n_cols, v % n_cols) for v in cycle]
        for i, (row, col) in enumerate(coords):
            nrow, ncol = coords[(i + 1) % len(coords)]
            assert abs(row - nrow) + abs(col - ncol) == 1, (shape, cycle)
    assert len(grid_faces((3, 4))) == 2 * 3


def test_diagrams_shapes_and_caching(method):
    faces, holes, signal = square()
    graph_filter = ta.GraphFilter.from_faces(faces, holes, signal, method=method)

    pd0, pd1 = graph_filter.diagrams()
    assert_birth_death(pd0)
    assert_birth_death(pd1)
    assert np.isinf(pd0[:, 1]).sum() == 1  # the essential component

    again = graph_filter.diagrams()
    assert np.array_equal(pd0, again[0]) and np.array_equal(pd1, again[1])


def test_diagrams_independent_of_starting_degree(method):
    faces, holes, signal = square()
    primal = ta.GraphFilter.from_faces(faces, holes, signal, method=method).diagrams()
    dual = ta.GraphFilter.from_faces(
        faces, holes, signal, dual=True, method=method
    ).diagrams()
    for left, right in zip(primal, dual):
        assert np.array_equal(left, right)


def test_module_level_diagram_matches_the_filter(method):
    faces, holes, signal = square()
    graph = ta.GraphWithFaces(faces, holes, signal)
    for left, right in zip(
        ta.GraphFilter(graph, method=method).diagrams(),
        ta.persistence_diagram(graph, method=method),
    ):
        assert np.array_equal(left, right)


def test_filtering_a_random_graph(method):
    faces, holes, n_vertices = ta.random_graph_with_faces(
        10, n_holes=2, rng=random.Random(0)
    )
    signal = np.random.default_rng(0).random(n_vertices)
    graph = ta.GraphWithFaces(faces, holes, signal)

    filtered = ta.GraphFilter(graph, method=method).low_pers_filter(0.2)
    assert filtered.shape == signal.shape
    assert np.abs(filtered - signal).max() < 0.2

    both = ta.low_pers_filter(graph, 0.2, order="01", method=method)
    assert both.shape == signal.shape

    # Filtering removes every H0 feature below the threshold.
    pd0, _ = ta.persistence_diagram(graph.with_signal(filtered), method=method)
    finite = pd0[np.isfinite(pd0[:, 1])]
    assert np.all(finite[:, 1] - finite[:, 0] >= 0.2 - 1e-12)


def test_graph_filter_requires_a_graph():
    with pytest.raises(TypeError):
        ta.GraphFilter([[0, 1, 2]])


def test_from_array_keeps_the_input_shape(method):
    array = np.random.default_rng(2).random((5, 6))
    graph_filter = ta.GraphFilter.from_array(array, method=method)
    assert graph_filter.low_pers_filter(0.1).shape == array.shape
    assert graph_filter.basin_map(0.0).shape == array.shape
