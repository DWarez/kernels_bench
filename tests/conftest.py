"""Shared fixtures and markers for kernels-bench tests."""

import pytest
import torch

from kernels_bench.runtime import Runtime, detect_runtime


def _detect_gpu() -> Runtime | None:
    """Return the current runtime, or None if no GPU is available."""
    try:
        return detect_runtime()
    except RuntimeError:
        return None


_runtime = _detect_gpu()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: test requires a GPU runtime")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _runtime is not None:
        return
    skip = pytest.mark.skip(reason="no supported GPU runtime available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def runtime() -> Runtime:
    """Return the detected GPU runtime. Tests using this are auto-marked as gpu."""
    assert _runtime is not None
    return _runtime


@pytest.fixture
def device(runtime: Runtime) -> str:
    """Return the device string for the current runtime (e.g. 'cuda', 'mps')."""
    return runtime.device
