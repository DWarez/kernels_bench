"""CUDA runtime (also covers AMD ROCm via HIP, which exposes torch.cuda)."""

from __future__ import annotations

import platform
import statistics
import threading
import time

import torch

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime._base import MetricsCollector, RunMetrics, Runtime, Timer


class CUDATimer(Timer):
    """GPU timer using CUDA events."""

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def record_start(self) -> None:
        self._start.record()

    def record_end(self) -> None:
        self._end.record()

    def elapsed_ms(self) -> float:
        return self._start.elapsed_time(self._end)


class CUDAMetricsCollector(MetricsCollector):
    """Samples GPU utilization in a background thread and tracks peak memory.

    Peak memory uses torch.cuda.max_memory_allocated(), reset at start().
    Utilization is sampled via torch.cuda.utilization() (NVML under the hood).
    If NVML is unavailable (e.g. some containers), util fields stay None.

    Note: on very short benchmarks (<3 samples collected), util stats are
    reported as None — a handful of samples are not representative.
    """

    # Sample every 5 ms. NVML query is cheap (~0.1 ms) relative to the
    # interval, and 5 ms keeps per-iteration overhead negligible while still
    # giving usable coverage for sub-second benchmarks.
    _SAMPLE_INTERVAL_S = 0.005
    _MIN_SAMPLES = 3

    def __init__(self) -> None:
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_broken = False

    def start(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(torch.cuda.utilization())
            except Exception:
                # NVML not available or failed — give up silently for this run.
                self._nvml_broken = True
                return
            time.sleep(self._SAMPLE_INTERVAL_S)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def result(self) -> RunMetrics:
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024**2) if peak_bytes > 0 else None

        have_samples = len(self._samples) >= self._MIN_SAMPLES and not self._nvml_broken
        util_mean = statistics.fmean(self._samples) if have_samples else None
        util_peak = float(max(self._samples)) if have_samples else None

        return RunMetrics(
            peak_memory_mb=peak_mb,
            util_mean=util_mean,
            util_peak=util_peak,
            util_samples=len(self._samples) if not self._nvml_broken else 0,
        )


class CUDARuntime(Runtime):
    """Runtime for NVIDIA CUDA and AMD ROCm (via HIP)."""

    @property
    def name(self) -> str:
        if torch.version.hip is not None:
            return "ROCm"
        return "CUDA"

    @property
    def device(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def create_timer(self) -> CUDATimer:
        return CUDATimer()

    def create_metrics_collector(self) -> CUDAMetricsCollector:
        return CUDAMetricsCollector()

    def get_device_info(self) -> DeviceInfo:
        if not self.is_available():
            return DeviceInfo(
                gpu_name="N/A (no GPU)",
                runtime_name=self.name,
                runtime_version="N/A",
                driver_version="N/A",
                torch_version=torch.__version__,
                gpu_memory_gb=0.0,
                python_version=platform.python_version(),
            )

        props = torch.cuda.get_device_properties(0)
        if torch.version.hip is not None:
            runtime_version = torch.version.hip or "unknown"
        else:
            runtime_version = torch.version.cuda or "unknown"

        return DeviceInfo(
            gpu_name=props.name,
            runtime_name=self.name,
            runtime_version=runtime_version,
            driver_version=str(torch.cuda.get_device_capability(0)),
            torch_version=torch.__version__,
            gpu_memory_gb=round(props.total_memory / (1024**3), 2),
            python_version=platform.python_version(),
        )
