"""Runtime abstraction for GPU timing, synchronization, and device info.

This package decouples the benchmark engine from any specific GPU backend.
To add a new backend, create a new module with Runtime and Timer subclasses,
then register the runtime class in _RUNTIMES below.
"""

from kernels_bench.runtime._base import MetricsCollector, RunMetrics, Runtime, Timer
from kernels_bench.runtime._cuda import CUDAMetricsCollector, CUDARuntime, CUDATimer
from kernels_bench.runtime._mps import MPSRuntime, MPSTimer

# Order matters: first match wins in detect_runtime().
_RUNTIMES: list[type[Runtime]] = [CUDARuntime, MPSRuntime]


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


__all__ = [
    "CUDAMetricsCollector",
    "CUDARuntime",
    "CUDATimer",
    "MPSRuntime",
    "MPSTimer",
    "MetricsCollector",
    "RunMetrics",
    "Runtime",
    "Timer",
    "detect_runtime",
]
