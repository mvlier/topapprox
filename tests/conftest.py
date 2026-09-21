"""Pytest configuration for topapprox."""

from __future__ import annotations

import pytest

from topapprox import available_backends

BACKENDS = available_backends()


def pytest_report_header(config):
    del config
    return f"topapprox backends: {', '.join(BACKENDS) or 'none'}"


@pytest.fixture(params=BACKENDS)
def method(request):
    """Run a test once per importable backend."""
    return request.param
