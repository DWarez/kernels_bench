"""Collect GPU and system information for benchmark reproducibility."""

from __future__ import annotations

import dataclasses
import platform

import torch


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    gpu_name: str
    cuda_version: str
    driver_version: str
    torch_version: str
    gpu_memory_gb: float
    python_version: str

    def to_dict(self) -> dict[str, str | float]:
        return dataclasses.asdict(self)


def get_device_info() -> DeviceInfo:
    """Collect information about the current CUDA device."""
    if not torch.cuda.is_available():
        return DeviceInfo(
            gpu_name="N/A (no CUDA)",
            cuda_version="N/A",
            driver_version="N/A",
            torch_version=torch.__version__,
            gpu_memory_gb=0.0,
            python_version=platform.python_version(),
        )

    props = torch.cuda.get_device_properties(0)
    return DeviceInfo(
        gpu_name=props.name,
        cuda_version=torch.version.cuda or "unknown",
        driver_version=str(torch.cuda.get_device_capability(0)),
        torch_version=torch.__version__,
        gpu_memory_gb=round(props.total_memory / (1024**3), 2),
        python_version=platform.python_version(),
    )
