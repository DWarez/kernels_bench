"""Apple MPS (Metal Performance Shaders) runtime."""

from __future__ import annotations

import contextlib
import platform

import torch

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime._base import Runtime


class MPSRuntime(Runtime):
    """Runtime for Apple Silicon GPUs via Metal Performance Shaders."""

    @property
    def name(self) -> str:
        return "MPS"

    @property
    def device(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        return torch.backends.mps.is_available()

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def get_device_info(self) -> DeviceInfo:
        if not self.is_available():
            return DeviceInfo(
                gpu_name="N/A (no MPS)",
                runtime_name=self.name,
                runtime_version="N/A",
                driver_version="N/A",
                torch_version=torch.__version__,
                gpu_memory_gb=0.0,
                python_version=platform.python_version(),
            )

        # Apple Silicon doesn't expose per-GPU properties via torch the way
        # CUDA does.  We pull what we can from the platform.
        chip = platform.processor() or "Apple Silicon"
        mac_ver = platform.mac_ver()[0]
        runtime_version = f"macOS {mac_ver}" if mac_ver else "macOS"

        # torch.mps has no API for total GPU memory; use recommended max
        # working set size when available, otherwise report 0.
        gpu_memory_gb = 0.0
        rec_max = getattr(torch.mps, "recommended_max_memory", None)
        if rec_max is not None:
            with contextlib.suppress(Exception):
                gpu_memory_gb = round(rec_max() / (1024**3), 2)

        return DeviceInfo(
            gpu_name=chip,
            runtime_name=self.name,
            runtime_version=runtime_version,
            driver_version="Metal",
            torch_version=torch.__version__,
            gpu_memory_gb=gpu_memory_gb,
            python_version=platform.python_version(),
        )
