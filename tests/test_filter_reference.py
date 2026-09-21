"""Hand-computed filtering results for small 1-D and 2-D signals.

These pin down the contract of :meth:`ImageFilter.low_pers_filter`: features of
persistence below the threshold disappear, everything else stays put, and the
result never moves further than the threshold from the input.
"""

from __future__ import annotations

import numpy as np
import pytest

import topapprox as ta

CASES_1D = {
    "single_peak": (
        [0, 2, 1],
        [(0.5, [0, 2, 1]), (1.0, [0, 2, 1]), (1.5, [0, 2, 2]), (4.0, [0, 2, 2])],
    ),
    "symmetric_peaks": (
        [0, 1, -2, 0, -1, 2, -1, 0],
        [
            (0.5, [0, 1, -2, 0, -1, 2, -1, 0]),
            (1.5, [1, 1, -2, 0, 0, 2, -1, 0]),
            (4.0, [1, 1, -2, 0, 0, 2, 2, 2]),
        ],
    ),
    "plateaus_and_valleys": (
        [0, 2, 2, 1, 1, 4, 1, 5, 4, 5, -1],
        [
            (0.5, [0, 2, 2, 1, 1, 4, 1, 5, 4, 5, -1]),
            (1.0, [0, 2, 2, 1, 1, 4, 1, 5, 4, 5, -1]),
            (1.5, [0, 2, 2, 2, 2, 4, 1, 5, 5, 5, -1]),
            (3.0, [0, 2, 2, 2, 2, 4, 1, 5, 5, 5, -1]),
            (3.5, [0, 2, 2, 2, 2, 4, 4, 5, 5, 5, -1]),
            (5.0, [0, 2, 2, 2, 2, 4, 4, 5, 5, 5, -1]),
            (5.5, [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1]),
        ],
    ),
}

IMAGE_2D = np.array([[0, 5, 3], [5, 6, 4], [2, 5, 1]], dtype=float)
FILTERED_H0 = np.array([[0, 5, 4], [5, 6, 4], [2, 5, 1]], dtype=float)
FILTERED_H0_H1 = np.array([[0, 5, 4], [5, 5, 4], [2, 5, 1]], dtype=float)


@pytest.mark.parametrize("name", sorted(CASES_1D))
@pytest.mark.parametrize("as_row", [False, True], ids=["1d", "row"])
def test_1d_cases(method, name, as_row):
    signal, expectations = CASES_1D[name]
    array = np.array([signal], dtype=float) if as_row else np.array(signal, dtype=float)
    image_filter = ta.ImageFilter(array, method=method)

    for epsilon, expected in expectations:
        result = image_filter.low_pers_filter(epsilon)
        assert result.shape == array.shape
        assert np.array_equal(result.ravel(), expected), f"{name} at epsilon={epsilon}"


def test_2d_case(method):
    h0 = ta.ImageFilter(IMAGE_2D, method=method).low_pers_filter(1.5)
    assert np.array_equal(h0, FILTERED_H0)

    h1 = ta.ImageFilter(h0, dual=True, method=method).low_pers_filter(1.5)
    assert np.array_equal(h1, FILTERED_H0_H1)


def test_one_shot_matches_step_by_step(method):
    assert np.array_equal(
        ta.low_pers_filter(IMAGE_2D, 1.5, order="01", method=method), FILTERED_H0_H1
    )
    assert np.array_equal(ta.low_pers_filter(IMAGE_2D, 1.5, order="0", method=method), FILTERED_H0)
    assert np.array_equal(ta.low_pers_filter(IMAGE_2D, 1.5, order="", method=method), IMAGE_2D)


def test_per_degree_thresholds(method):
    paired = ta.low_pers_filter(IMAGE_2D, (1.5, 0.0), order="01", method=method)
    assert np.array_equal(paired, FILTERED_H0)
    assert np.array_equal(
        ta.low_pers_filter(IMAGE_2D, {0: 1.5, 1: 1.5}, order="01", method=method),
        FILTERED_H0_H1,
    )


def test_rejects_unsupported_shapes():
    with pytest.raises(ValueError):
        ta.ImageFilter(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError):
        ta.low_pers_filter(IMAGE_2D, 1.0, order="02")
