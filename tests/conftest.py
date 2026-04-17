"""Pytest configuration for topapprox."""

from __future__ import annotations

import importlib.util

import pytest

from topapprox import available_backends


AVAILABLE_BACKENDS = available_backends()
HAS_CRIPSER = importlib.util.find_spec("cripser") is not None


def pytest_report_header(config):
    del config
    return f"topapprox backends: {', '.join(AVAILABLE_BACKENDS) or 'none'}"


@pytest.fixture(scope="session")
def available_methods():
    return AVAILABLE_BACKENDS
