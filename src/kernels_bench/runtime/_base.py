"""Abstract base classes for GPU runtimes."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

from kernels_bench.device import DeviceInfo


class Timer(ABC):
    """Opaque handle for recording GPU-side elapsed time."""

    @abstractmethod
    def record_start(self) -> None:
        """Mark the start of a timed region on the device timeline."""

    @abstractmethod
    def record_end(self) -> None:
        """Mark the end of a timed region on the device timeline."""

    @abstractmethod
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds between start and end."""


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

    @abstractmethod
    def create_timer(self) -> Timer:
        """Create a new timer for measuring device-side execution time."""

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """Collect information about the current device."""

    def create_metrics_collector(self) -> MetricsCollector:
        """Create a collector for device metrics. Defaults to a no-op."""
        return _NoopMetricsCollector()
