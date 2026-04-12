"""Tensor specifications for benchmark inputs and outputs."""

from __future__ import annotations

import torch
from pydantic import BaseModel, model_validator

# A dimension is either a concrete int or a symbolic name to be resolved from params
Dim = int | str

# Role determines how a tensor is allocated
Role = str  # "input" or "output"


class TensorSpec(BaseModel):
    """Describes a tensor's name, shape, dtype, device, and role.

    Shape dimensions can be concrete ints or symbolic strings that get resolved
    from the benchmark's `params` dict at runtime.

    The `role` field controls allocation: "input" tensors are filled with random
    data, "output" tensors are allocated empty. Defaults to "input".

    Examples:
        TensorSpec("x", shape=(1024, 1024), dtype=torch.float16)
        TensorSpec("y", shape=(1024, 1024), dtype=torch.float16, role="output")
        TensorSpec("x", shape=("M", "N"), dtype=torch.float16)  # symbolic dims
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    shape: tuple[Dim, ...]
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    role: Role = "input"

    def __init__(
        self,
        name: str,
        *,
        shape: tuple[Dim, ...],
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        role: Role = "input",
    ) -> None:
        super().__init__(name=name, shape=shape, dtype=dtype, device=device, role=role)

    @model_validator(mode="after")
    def _validate(self) -> TensorSpec:
        if not self.shape:
            raise ValueError("shape must have at least one dimension")
        for dim in self.shape:
            if isinstance(dim, int) and dim <= 0:
                raise ValueError(f"concrete dimensions must be positive, got {dim}")
            elif isinstance(dim, str) and not dim.isidentifier():
                raise ValueError(f"symbolic dimension must be a valid identifier, got {dim!r}")
        if self.role not in ("input", "output"):
            raise ValueError(f"role must be 'input' or 'output', got {self.role!r}")
        return self

    @property
    def symbolic_dims(self) -> set[str]:
        """Return the set of symbolic dimension names in this spec."""
        return {d for d in self.shape if isinstance(d, str)}

    def resolve(self, params: dict[str, int]) -> TensorSpec:
        """Return a new TensorSpec with all symbolic dims replaced by concrete values."""
        resolved_shape: list[int] = []
        for dim in self.shape:
            if isinstance(dim, str):
                if dim not in params:
                    raise ValueError(f"symbolic dimension {dim!r} not found in params")
                resolved_shape.append(params[dim])
            else:
                resolved_shape.append(dim)
        return TensorSpec(
            self.name,
            shape=tuple(resolved_shape),
            dtype=self.dtype,
            device=self.device,
            role=self.role,
        )

    def allocate(self) -> torch.Tensor:
        """Allocate a tensor based on role: random data for inputs, empty for outputs."""
        if self.role == "output":
            return self.allocate_output()
        return self.allocate_input()

    def allocate_input(self) -> torch.Tensor:
        """Allocate a tensor filled with random data (for inputs)."""
        concrete = [d for d in self.shape if isinstance(d, int)]
        if len(concrete) != len(self.shape):
            raise RuntimeError("cannot allocate tensor with unresolved symbolic dims")
        return torch.randn(concrete, dtype=self.dtype, device=self.device)

    def allocate_output(self) -> torch.Tensor:
        """Allocate an empty tensor (for outputs)."""
        concrete = [d for d in self.shape if isinstance(d, int)]
        if len(concrete) != len(self.shape):
            raise RuntimeError("cannot allocate tensor with unresolved symbolic dims")
        return torch.empty(concrete, dtype=self.dtype, device=self.device)
