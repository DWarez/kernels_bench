"""Tests for device info collection."""

import pytest

from kernels_bench.device import DeviceInfo


@pytest.mark.gpu
def test_get_device_info_returns_device_info(runtime):
    info = runtime.get_device_info()
    assert isinstance(info, DeviceInfo)
    assert info.torch_version != ""
    assert info.python_version != ""


@pytest.mark.gpu
def test_device_info_to_dict(runtime):
    info = runtime.get_device_info()
    d = info.to_dict()
    assert "gpu_name" in d
    assert "runtime_name" in d
    assert "runtime_version" in d
    assert "torch_version" in d
    assert "gpu_memory_gb" in d


@pytest.mark.gpu
def test_device_info_has_gpu(runtime):
    info = runtime.get_device_info()
    assert info.gpu_name != "N/A (no GPU)"
    assert info.gpu_memory_gb > 0
