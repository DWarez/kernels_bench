"""Tests for runtime detection and backend implementations."""

import pytest

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime import Runtime, detect_runtime


def test_detect_runtime_returns_runtime():
    rt = detect_runtime()
    assert isinstance(rt, Runtime)


def test_detect_runtime_is_available():
    rt = detect_runtime()
    assert rt.is_available()


def test_detect_runtime_has_name():
    rt = detect_runtime()
    assert isinstance(rt.name, str)
    assert len(rt.name) > 0


def test_detect_runtime_has_device():
    rt = detect_runtime()
    assert isinstance(rt.device, str)
    assert len(rt.device) > 0


@pytest.mark.gpu
def test_runtime_synchronize(runtime):
    """synchronize() should not raise."""
    runtime.synchronize()


@pytest.mark.gpu
def test_runtime_create_timer(runtime):
    timer = runtime.create_timer()
    # Should have the Timer interface
    assert hasattr(timer, "record_start")
    assert hasattr(timer, "record_end")
    assert hasattr(timer, "elapsed_ms")


@pytest.mark.gpu
def test_runtime_timer_measures_time(runtime, device):
    """Timer should measure non-negative time for a real GPU operation."""
    import torch

    timer = runtime.create_timer()
    x = torch.randn(512, 512, device=device)

    timer.record_start()
    _ = x @ x
    timer.record_end()
    runtime.synchronize()

    elapsed = timer.elapsed_ms()
    assert elapsed >= 0.0


@pytest.mark.gpu
def test_runtime_get_device_info(runtime):
    info = runtime.get_device_info()
    assert isinstance(info, DeviceInfo)
    assert info.runtime_name == runtime.name
    assert info.torch_version != ""
    assert info.python_version != ""
    assert info.gpu_name != ""
    assert info.gpu_memory_gb >= 0.0
