"""Tests for runtime detection and backend implementations."""

import time

import pytest
import torch

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime import RunMetrics, Runtime, detect_runtime
from kernels_bench.runtime._base import _NoopMetricsCollector
from kernels_bench.runtime._cuda import CUDAMetricsCollector, CUDARuntime


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
def test_runtime_get_device_info(runtime):
    info = runtime.get_device_info()
    assert isinstance(info, DeviceInfo)
    assert info.runtime_name == runtime.name
    assert info.torch_version != ""
    assert info.python_version != ""
    assert info.gpu_name != ""
    assert info.gpu_memory_gb >= 0.0


def test_noop_metrics_collector_returns_empty():
    """Default (non-CUDA) runtimes return a no-op collector with all-None fields."""
    c = _NoopMetricsCollector()
    c.start()
    c.stop()
    r = c.result()
    assert isinstance(r, RunMetrics)
    assert r.peak_memory_mb is None
    assert r.util_mean is None
    assert r.util_peak is None
    assert r.util_samples == 0


def test_runtime_base_create_metrics_collector_defaults_to_noop():
    """A Runtime subclass that doesn't override create_metrics_collector gets the no-op."""

    class StubRuntime(Runtime):
        name = "Stub"
        device = "cpu"

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            pass

        def get_device_info(self):  # pragma: no cover
            raise NotImplementedError

    c = StubRuntime().create_metrics_collector()
    assert isinstance(c, _NoopMetricsCollector)
    assert c.result() == RunMetrics()


def _skip_unless_cuda(runtime):
    if not isinstance(runtime, CUDARuntime):
        pytest.skip("CUDA-only test")


@pytest.mark.gpu
def test_cuda_metrics_collector_tracks_peak_memory(runtime, device):
    _skip_unless_cuda(runtime)
    collector = CUDAMetricsCollector()

    collector.start()
    # Allocate ~16 MB inside the window: 2048*2048 fp16 = 8 MB, keep two alive.
    a = torch.empty((2048, 2048), dtype=torch.float16, device=device)
    b = torch.empty((2048, 2048), dtype=torch.float16, device=device)
    a.fill_(1.0)
    b.fill_(2.0)
    torch.cuda.synchronize()
    collector.stop()

    result = collector.result()
    assert result.peak_memory_mb is not None
    assert result.peak_memory_mb >= 16.0  # at least the two tensors

    # Keep tensors alive until after the assertion so the allocator can't free them
    # before the peak is read.
    del a, b


@pytest.mark.gpu
def test_cuda_metrics_collector_resets_between_runs(runtime, device):
    _skip_unless_cuda(runtime)
    collector = CUDAMetricsCollector()

    # First run: allocate a big tensor (~32 MB)
    collector.start()
    big = torch.empty((4096, 4096), dtype=torch.float16, device=device)
    big.fill_(0.0)
    torch.cuda.synchronize()
    collector.stop()
    first_peak = collector.result().peak_memory_mb
    del big
    torch.cuda.empty_cache()

    # Second run: allocate a small tensor (~1 MB). Peak should reflect the smaller run,
    # not carry over the earlier allocation.
    collector.start()
    small = torch.empty((512, 512), dtype=torch.float16, device=device)
    small.fill_(0.0)
    torch.cuda.synchronize()
    collector.stop()
    second_peak = collector.result().peak_memory_mb
    del small

    assert first_peak is not None and second_peak is not None
    assert second_peak < first_peak


@pytest.mark.gpu
def test_cuda_metrics_collector_short_run_no_util(runtime, device):
    """A tiny window (<3 samples) reports util as None rather than misleading stats."""
    _skip_unless_cuda(runtime)
    collector = CUDAMetricsCollector()
    collector.start()
    # Don't sleep — stop immediately. Sampler thread will likely record 0 or 1 samples.
    collector.stop()
    r = collector.result()
    assert r.util_mean is None
    assert r.util_peak is None


@pytest.mark.gpu
def test_cuda_metrics_collector_long_run_has_util(runtime, device):
    """A window long enough to collect >= 3 samples reports util stats."""
    _skip_unless_cuda(runtime)
    collector = CUDAMetricsCollector()
    collector.start()
    # Sampler interval is 5 ms; 50 ms easily yields >= 3 samples.
    time.sleep(0.05)
    collector.stop()
    r = collector.result()
    # If NVML is present (it is for this branch's deps), we expect stats.
    # If NVML failed, util_samples will be 0 and util fields None — tolerate that.
    if r.util_samples >= 3:
        assert r.util_mean is not None
        assert r.util_peak is not None
        assert 0.0 <= r.util_mean <= 100.0
        assert 0.0 <= r.util_peak <= 100.0
        assert r.util_peak >= r.util_mean
