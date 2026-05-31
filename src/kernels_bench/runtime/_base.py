"""Abstract base classes for GPU runtimes."""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from kernels_bench.device import DeviceInfo


@dataclasses.dataclass(frozen=True)
class RunMetrics:
    """Device-side metrics collected during a benchmark run.

    All fields are optional — a runtime that cannot (or chose not to) collect
    a given metric leaves it as None.
    """

    peak_memory_mb: float | None = None
    util_mean: float | None = None
    util_peak: float | None = None
    util_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "util_mean": self.util_mean,
            "util_peak": self.util_peak,
            "util_samples": self.util_samples,
        }


class MetricsCollector(ABC):
    """Collects device metrics over a timed region.

    Usage:
        collector.start()      # before the timed iterations
        # ... run the kernel repeatedly ...
        collector.stop()       # after the last iteration
        metrics = collector.result()
    """

    @abstractmethod
    def start(self) -> None:
        """Begin collecting metrics."""

    @abstractmethod
    def stop(self) -> None:
        """Stop collecting metrics."""

    @abstractmethod
    def result(self) -> RunMetrics:
        """Return the collected metrics. Safe to call only after stop()."""


class _NoopMetricsCollector(MetricsCollector):
    """Default collector that records nothing — used by runtimes without support."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def result(self) -> RunMetrics:
        return RunMetrics()


class Runtime(ABC):
    """Abstract GPU runtime. Timing, synchronization, and device info."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable runtime name, e.g. 'CUDA', 'ROCm', 'MPS'."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Torch device string for tensor allocation, e.g. 'cuda', 'mps'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the runtime's hardware is present and usable."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all pending device operations complete."""

    def time_calls(self, fn: Callable[..., Any], args: list[Any], n: int) -> float:
        """Return elapsed seconds for `n` back-to-back ``fn(*args)`` calls.

        The default implementation uses a synchronized host clock: it
        synchronizes, enqueues all `n` calls, synchronizes again, and returns
        the wall time. This includes Python launch overhead in the measurement.

        Subclasses backed by a device timer (CUDA/MPS events) override this to
        measure pure *device* time, which excludes the host-side cost of
        enqueuing each launch — essential for fast kernels whose runtime is
        smaller than the launch overhead.

        Batching `n` calls under a single event pair (in the device-timer
        overrides) keeps launches pipelined and amortizes the fixed cost of
        starting/stopping the timer, so the per-call figure isn't inflated by
        timer overhead — important for kernels only a few µs long.
        """
        self.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn(*args)
        self.synchronize()
        return time.perf_counter() - start

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """Collect information about the current device."""

    def create_metrics_collector(self) -> MetricsCollector:
        """Create a collector for device metrics. Defaults to a no-op."""
        return _NoopMetricsCollector()
