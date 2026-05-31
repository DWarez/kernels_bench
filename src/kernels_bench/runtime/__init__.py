"""Runtime abstraction for GPU synchronization, metrics, and device info.

This package decouples the benchmark engine from any specific GPU backend.
To add a new backend, create a new Runtime subclass, then register it in
_RUNTIMES below. Measurement timing is handled by the runtime's `time_calls`
(device events on CUDA/MPS, a synchronized host clock otherwise), driven by
runner._timed_loop.
"""

from kernels_bench.runtime._base import MetricsCollector, RunMetrics, Runtime
from kernels_bench.runtime._cuda import CUDAMetricsCollector, CUDARuntime
from kernels_bench.runtime._mps import MPSRuntime

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
    "MPSRuntime",
    "MetricsCollector",
    "RunMetrics",
    "Runtime",
    "detect_runtime",
]
