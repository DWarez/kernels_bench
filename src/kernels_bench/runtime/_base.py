"""Abstract base classes for GPU runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
