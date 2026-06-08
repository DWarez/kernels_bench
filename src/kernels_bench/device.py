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

    @classmethod
    def from_dict(cls, d: dict[str, str | float]) -> DeviceInfo:
        """Rebuild from a ``to_dict`` payload (used for remote-result transport)."""
        return cls(
            gpu_name=str(d["gpu_name"]),
            runtime_name=str(d["runtime_name"]),
            runtime_version=str(d["runtime_version"]),
            driver_version=str(d["driver_version"]),
            torch_version=str(d["torch_version"]),
            gpu_memory_gb=float(d["gpu_memory_gb"]),
            python_version=str(d["python_version"]),
        )
