"""Runtime abstraction for GPU timing, synchronization, and device info.

This module decouples the benchmark engine from any specific GPU backend.
Currently ships with a CUDA runtime (which also covers AMD ROCm via HIP).
To add a new backend, subclass Runtime and Timer.
"""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod

import torch

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
        """Human-readable runtime name, e.g. 'CUDA', 'ROCm', 'XPU'."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Torch device string for tensor allocation, e.g. 'cuda', 'xpu'."""

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


# ---------------------------------------------------------------------------
# CUDA runtime (also covers AMD ROCm via HIP, which exposes torch.cuda)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

_RUNTIMES: list[type[Runtime]] = [CUDARuntime]


def detect_runtime() -> Runtime:
    """Return the first available runtime, or raise if none found."""
    for cls in _RUNTIMES:
        rt = cls()
        if rt.is_available():
            return rt
    available = ", ".join(cls.__name__ for cls in _RUNTIMES)
    raise RuntimeError(
        f"no supported GPU runtime detected (checked: {available}). kernels-bench requires a GPU."
    )
