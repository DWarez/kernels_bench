"""Tests for device info collection."""

from kernels_bench.device import DeviceInfo, get_device_info


def test_get_device_info_returns_device_info():
    info = get_device_info()
    assert isinstance(info, DeviceInfo)
    assert info.torch_version != ""
    assert info.python_version != ""


def test_device_info_to_dict():
    info = get_device_info()
    d = info.to_dict()
    assert "gpu_name" in d
    assert "cuda_version" in d
    assert "torch_version" in d
    assert "gpu_memory_gb" in d


def test_device_info_has_gpu():
    info = get_device_info()
    # This test assumes CUDA is available in the test environment
    assert info.gpu_name != "N/A (no CUDA)"
    assert info.gpu_memory_gb > 0
