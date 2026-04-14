"""CUDA runtime (also covers AMD ROCm via HIP, which exposes torch.cuda)."""

from __future__ import annotations

import platform

import torch

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime._base import Runtime, Timer


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
