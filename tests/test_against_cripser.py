"""Cross-check against an independent persistent-homology implementation.

Requires ``cripser`` (CubicalRipser).  Two properties are checked on real and
random data: the filtered signal stays within ``epsilon`` of the input in the
supremum norm, and its diagram is exactly the input's diagram with every pair of
persistence below ``epsilon`` deleted.
"""

from __future__ import annotations

import numpy as np
import pytest

import topapprox as ta

cripser = pytest.importorskip("cripser")

PAGODA = "tests/data_for_testing/pagoda.npy"


def diagram_by_degree(array, maxdim):
    raw = cripser.computePH(np.ascontiguousarray(array), maxdim=maxdim)
    return [raw[raw[:, 0] == degree][:, 1:3] for degree in range(maxdim + 1)]


def lexsorted(pairs):
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


@pytest.mark.parametrize("epsilon", [0.2])
def test_pagoda_filtering(epsilon, method):
    image = np.load(PAGODA)
    h0 = ta.ImageFilter(image, method=method).low_pers_filter(epsilon)
    h1 = ta.ImageFilter(image, dual=True, method=method).low_pers_filter(epsilon)

    assert np.abs(h0 - image).max() < epsilon
    assert np.abs(h1 - image).max() < epsilon

    original = diagram_by_degree(image, maxdim=1)
    expected = [pairs[pairs[:, 1] - pairs[:, 0] >= epsilon] for pairs in original]
    assert np.array_equal(diagram_by_degree(h0, maxdim=0)[0], expected[0])
    assert np.array_equal(diagram_by_degree(h1, maxdim=1)[1], expected[1])


def test_3d_diagrams_match():
    array = np.random.default_rng(0).random((30, 30, 30))
    ours = [
        ta.ImageFilter(array).diagram()[:, :2],
        ta.ImageFilter(array, dual=True).diagram()[:, :2],
    ]
    theirs = diagram_by_degree(array, maxdim=2)
    # CubicalRipser reports the essential H0 class, our reduced diagram does not.
    assert np.allclose(lexsorted(ours[0]), lexsorted(theirs[0])[1:])
    assert np.allclose(lexsorted(ours[1]), lexsorted(theirs[2]))


def test_2d_diagrams_match():
    array = np.random.default_rng(1).random((40, 50))
    ours = [
        ta.ImageFilter(array).diagram()[:, :2],
        ta.ImageFilter(array, dual=True).diagram()[:, :2],
    ]
    theirs = diagram_by_degree(array, maxdim=1)
    assert np.allclose(lexsorted(ours[0]), lexsorted(theirs[0])[1:])
    assert np.allclose(lexsorted(ours[1]), lexsorted(theirs[1]))
