"""Regression tests for high-level orchestration behaviour."""

from __future__ import annotations

import numpy as np

from topapprox import GraphFilter, ImageFilter, PersistenceFilter
from topapprox.persistence import get_PD_gwf


def test_persistence_filter_cache_is_keyed_by_threshold():
    signal = np.array(
        [
            [0.0, 5.0, 3.0],
            [5.0, 6.0, 4.0],
            [2.0, 5.0, 1.0],
        ]
    )

    persistence_filter = PersistenceFilter().load_signal(signal)
    filtered_low = persistence_filter.low_pers_filter(1.5, iteration_order="0", method="python")
    filtered_high = persistence_filter.low_pers_filter(10.0, iteration_order="0", method="python")

    expected_low = ImageFilter(signal, method="python").low_pers_filter(1.5)
    expected_high = ImageFilter(signal, method="python").low_pers_filter(10.0)

    assert np.array_equal(filtered_low, expected_low)
    assert np.array_equal(filtered_high, expected_high)
    assert not np.array_equal(filtered_low, filtered_high)


def test_readme_quick_start_examples_run():
    img = np.array([[0.0, 5.0, 3.0], [5.0, 6.0, 4.0], [2.0, 5.0, 1.0]])

    image_filter = ImageFilter(img, method="python")
    filtered_h0 = image_filter.low_pers_filter(1.5)

    dual_filter = ImageFilter(img, dual=True, method="python")
    filtered_h1 = dual_filter.low_pers_filter(1.5)

    faces = [[0, 1, 2, 3]]
    holes = [[0, 1, 2, 3]]
    signal = np.array([0.0, 1.0, 0.5, 0.2])

    graph_filter = GraphFilter(method="python")
    graph_filter.compute_gwf(F=faces, H=holes, signal=signal)
    pd0, pd1 = graph_filter.get_diagram()

    persistence_filter = PersistenceFilter().load_signal(img)
    filtered_01 = persistence_filter.low_pers_filter(1.5, iteration_order="01", method="python")

    pd0_fn, pd1_fn = get_PD_gwf(faces, holes, signal, method="python")

    assert filtered_h0.shape == img.shape
    assert filtered_h1.shape == img.shape
    assert filtered_01.shape == img.shape
    assert pd0.shape[1] == 2
    assert pd1.shape[1] == 2
    assert np.array_equal(pd0, pd0_fn)
    assert np.array_equal(pd1, pd1_fn)
