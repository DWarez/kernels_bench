"""Device information for benchmark reproducibility."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    gpu_name: str
    runtime_name: str
    runtime_version: str
    driver_version: str
    torch_version: str
    gpu_memory_gb: float
    python_version: str

    def to_dict(self) -> dict[str, str | float]:
        return dataclasses.asdict(self)
