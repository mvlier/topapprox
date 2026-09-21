"""The vectorised merge-tree algorithms against straightforward tree walks.

:class:`~topapprox.merge_tree.MergeTree` answers everything with pointer
doubling and array reductions.  Here the same answers are recomputed with plain
recursion-free traversals, which are obviously correct but slow.
"""

from __future__ import annotations

import numpy as np
import pytest

import topapprox as ta

SHAPES = [(1, 7), (4, 5), (6, 7), (5, 5, 5)]


def children_of(tree):
    children = [[] for _ in range(tree.n_vertices)]
    for vertex in range(tree.n_vertices):
        if vertex != tree.root:
            children[tree.parent[vertex]].append(vertex)
    return children


def descendants_of(children, vertex):
    collected, stack = [], [vertex]
    while stack:
        current = stack.pop()
        collected.append(current)
        stack.extend(children[current])
    return collected


def walk(tree):
    """``(subtree_size, depth, positive_height)`` by explicit traversal."""
    children = children_of(tree)
    size = np.ones(tree.n_vertices, dtype=np.int64)
    depth = np.zeros(tree.n_vertices, dtype=np.int64)
    height = np.zeros(tree.n_vertices, dtype=np.int64)

    order, stack = [], [(tree.root, 0)]
    while stack:
        vertex, level = stack.pop()
        order.append(vertex)
        depth[vertex] = level
        stack.extend((child, level + 1) for child in children[vertex])

    for vertex in reversed(order):
        for child in children[vertex]:
            size[vertex] += size[child]
            if is_basin(tree, child):
                height[vertex] = max(height[vertex], height[child] + 1)
    return size, depth, height


def is_basin(tree, vertex):
    return tree.persistence[vertex] > 0 and tree.linking_vertex[vertex] >= 0


def walk_filter(tree, epsilon, size_range=(0, np.inf)):
    """Top-down basin filling, the definition the vectorised filter implements."""
    low, high = size_range
    children = children_of(tree)
    filtered = tree.birth.copy()
    stack = [tree.root]
    while stack:
        for child in children[stack.pop()]:
            if not is_basin(tree, child):
                continue
            if tree.persistence[child] >= epsilon:
                stack.append(child)
                continue
            basin = descendants_of(children, child)
            if len(basin) > high:
                stack.append(child)
            elif len(basin) >= low:
                filtered[basin] = tree.death[child]
    return filtered


def walk_basins(tree, epsilon):
    children = children_of(tree)
    labels = np.full(tree.n_vertices, tree.birth[tree.root])
    representatives = np.full(tree.n_vertices, tree.root, dtype=np.int64)
    stack = [tree.root]
    while stack:
        for child in children[stack.pop()]:
            if not is_basin(tree, child) or tree.persistence[child] < epsilon:
                continue
            basin = descendants_of(children, child)
            labels[basin] = tree.birth[child]
            representatives[basin] = child
            stack.append(child)
    return labels, representatives


def sample_trees(seed=0):
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        for dual in (False, True):
            array = rng.integers(0, 5, size=shape).astype(float)
            yield ta.ImageFilter(array, dual=dual).tree


@pytest.mark.parametrize("tree", sample_trees(), ids=lambda t: f"n{t.n_vertices}")
def test_derived_quantities(tree):
    size, depth, height = walk(tree)
    assert np.array_equal(tree.subtree_size, size)
    assert np.array_equal(tree.depth, depth)
    assert np.array_equal(tree.positive_height, height)
    assert tree.parent[tree.root] == tree.root
    assert tree.linking_vertex[tree.root] == -1
    # Parents are older than their children, and die later.
    others = np.arange(tree.n_vertices) != tree.root
    assert np.all(tree.birth[tree.parent[others]] <= tree.birth[others])
    assert np.all(tree.death[tree.parent[others]] >= tree.death[others])


@pytest.mark.parametrize("tree", sample_trees(1), ids=lambda t: f"n{t.n_vertices}")
def test_descendants(tree):
    children = children_of(tree)
    for vertex in range(0, tree.n_vertices, max(1, tree.n_vertices // 20)):
        expected = sorted(descendants_of(children, vertex))
        assert sorted(tree.descendants(vertex).tolist()) == expected
        assert len(expected) == tree.subtree_size[vertex]


@pytest.mark.parametrize("tree", sample_trees(2), ids=lambda t: f"n{t.n_vertices}")
@pytest.mark.parametrize("epsilon", [0.0, 0.5, 1.0, 2.0, 10.0])
def test_low_pers_filter(tree, epsilon):
    assert np.allclose(tree.low_pers_filter(epsilon), walk_filter(tree, epsilon))


@pytest.mark.parametrize("tree", sample_trees(3), ids=lambda t: f"n{t.n_vertices}")
@pytest.mark.parametrize("size_range", [(0, np.inf), (2, 6), (3, 3), (5, np.inf)])
def test_size_range_filter(tree, size_range):
    for epsilon in (1.0, 3.0):
        assert np.allclose(
            tree.low_pers_filter(epsilon, size_range=size_range),
            walk_filter(tree, epsilon, size_range),
        )


@pytest.mark.parametrize("tree", sample_trees(4), ids=lambda t: f"n{t.n_vertices}")
@pytest.mark.parametrize("epsilon", [0.0, 1.0, 2.0, 5.0])
def test_basins(tree, epsilon):
    labels, representatives = walk_basins(tree, epsilon)
    assert np.allclose(tree.basin_map(epsilon), labels)
    assert np.array_equal(tree.basin_representatives(epsilon), representatives)


def test_diagram_columns():
    tree = ta.ImageFilter(np.array([[0.0, 5, 3], [5, 6, 4], [2, 5, 1]])).tree
    reduced = tree.diagram()
    full = tree.diagram(reduced=False)

    assert reduced.shape[1] == len(ta.DIAGRAM_COLUMNS) == 7
    assert full.shape[0] == reduced.shape[0] + 1
    assert np.isinf(full[-1, 1]) and full[-1, 3] == -1
    births, deaths = reduced[:, 0], reduced[:, 1]
    assert np.all(deaths > births)
    assert np.array_equal(tree.birth[reduced[:, 2].astype(int)], births)
    assert np.array_equal(tree.birth[reduced[:, 3].astype(int)], deaths)


def test_distributions():
    tree = ta.ImageFilter(np.random.default_rng(0).random((20, 20))).tree
    diagram = tree.diagram()
    assert tree.depth_distribution().sum() == len(diagram)
    assert tree.height_distribution().sum() == len(diagram)
    assert np.isclose(tree.depth_distribution(ratio=True).sum(), 1.0)
    assert np.isclose(
        tree.height_distribution(weight="persistence").sum(),
        (diagram[:, 1] - diagram[:, 0]).sum(),
    )
    with pytest.raises(ValueError):
        tree.depth_distribution(weight="nonsense")
