"""Correctness validation — compare output tensors across kernels."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from kernels_bench.spec import TensorSpec

if TYPE_CHECKING:
    from kernels_bench.runtime import Runtime


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Result of comparing two kernels' outputs."""

    kernel_a: str
    kernel_b: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mismatched_elements: int
    total_elements: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Full validation report across all kernel pairs."""

    comparisons: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def _collect_outputs_quick(
    kernel: Any,
    fn_name: str,
    specs: list[TensorSpec],
    input_tensors: list[torch.Tensor],
    runtime: Runtime,
) -> list[torch.Tensor]:
    """Run a kernel function once and return the output tensors.

    Uses shared input tensors so all kernels get the same inputs.
    """
    fn = getattr(kernel, fn_name)

    # Allocate fresh outputs, reuse the shared inputs
    args: list[torch.Tensor] = []
    output_tensors: list[torch.Tensor] = []
    device = input_tensors[0].device.type if input_tensors else "cuda"
    for i, spec in enumerate(specs):
        if spec.role == "output":
            t = spec.allocate_output(device)
            output_tensors.append(t)
            args.append(t)
        else:
            args.append(input_tensors[i])

    fn(*args)
    runtime.synchronize()
    return output_tensors


def _collect_outputs_bench(
    bench_fn: Any,
    kernel: Any,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    input_tensors: dict[str, torch.Tensor],
    runtime: Runtime,
) -> dict[str, torch.Tensor]:
    """Run a bench function once and return the output tensors by name.

    Uses shared input tensors so all kernels get the same inputs.
    """
    # Allocate fresh outputs on the same device as the inputs
    device = next(iter(input_tensors.values())).device.type
    output_tensors: dict[str, torch.Tensor] = {}
    for spec in output_specs:
        output_tensors[spec.name] = spec.allocate_output(device)

    args = [kernel]
    args.extend(input_tensors[s.name] for s in input_specs)
    args.extend(output_tensors[s.name] for s in output_specs)

    bench_fn(*args)
    runtime.synchronize()
    return output_tensors


def _compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float, int]:
    """Compare two tensors and return (passed, max_abs_diff, max_rel_diff, mismatched_count)."""
    # Cast to float32 for accurate comparison
    a_f = a.float()
    b_f = b.float()

    abs_diff = (a_f - b_f).abs()
    max_abs = abs_diff.max().item()

    # Relative diff: |a - b| / max(|a|, |b|, 1e-8)
    denom = torch.maximum(a_f.abs(), b_f.abs()).clamp(min=1e-8)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()

    passed = bool(torch.allclose(a_f, b_f, atol=atol, rtol=rtol))
    mismatched = int((abs_diff > atol + rtol * b_f.abs()).sum().item())

    return passed, max_abs, max_rel, mismatched


def validate_quick(
    kernels: dict[str, Any],
    fn_name: str,
    specs: list[TensorSpec],
    runtime: Runtime,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> ValidationReport:
    """Validate that all kernels produce the same outputs for the quick command."""
    # Allocate shared input tensors once
    input_tensors: list[torch.Tensor] = []
    for spec in specs:
        if spec.role == "input":
            input_tensors.append(spec.allocate_input(runtime.device))
        else:
            input_tensors.append(torch.empty(0))  # placeholder, won't be used

    # Collect outputs for each kernel
    kernel_outputs: dict[str, list[torch.Tensor]] = {}
    for kernel_id, kernel in kernels.items():
        kernel_outputs[kernel_id] = _collect_outputs_quick(
            kernel, fn_name, specs, input_tensors, runtime
        )

    # Pairwise comparison
    kernel_ids = list(kernels.keys())
    comparisons: list[ValidationResult] = []

    for i in range(len(kernel_ids)):
        for j in range(i + 1, len(kernel_ids)):
            id_a, id_b = kernel_ids[i], kernel_ids[j]
            outputs_a = kernel_outputs[id_a]
            outputs_b = kernel_outputs[id_b]

            # Compare each output tensor
            all_passed = True
            total_max_abs = 0.0
            total_max_rel = 0.0
            total_mismatched = 0
            total_elements = 0

            for out_a, out_b in zip(outputs_a, outputs_b, strict=True):
                passed, max_abs, max_rel, mismatched = _compare_tensors(out_a, out_b, atol, rtol)
                all_passed = all_passed and passed
                total_max_abs = max(total_max_abs, max_abs)
                total_max_rel = max(total_max_rel, max_rel)
                total_mismatched += mismatched
                total_elements += out_a.numel()

            comparisons.append(
                ValidationResult(
                    kernel_a=id_a,
                    kernel_b=id_b,
                    passed=all_passed,
                    max_abs_diff=total_max_abs,
                    max_rel_diff=total_max_rel,
                    mismatched_elements=total_mismatched,
                    total_elements=total_elements,
                )
            )

    return ValidationReport(comparisons=comparisons)


def validate_bench(
    bench_fn: Any,
    kernels: dict[str, Any],
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    runtime: Runtime,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> ValidationReport:
    """Validate that all kernels produce the same outputs for the run command."""
    # Allocate shared input tensors once
    input_tensors: dict[str, torch.Tensor] = {}
    for spec in input_specs:
        input_tensors[spec.name] = spec.allocate_input(runtime.device)

    # Collect outputs for each kernel
    kernel_outputs: dict[str, dict[str, torch.Tensor]] = {}
    for kernel_id, kernel in kernels.items():
        kernel_outputs[kernel_id] = _collect_outputs_bench(
            bench_fn, kernel, input_specs, output_specs, input_tensors, runtime
        )

    # Pairwise comparison
    kernel_ids = list(kernels.keys())
    comparisons: list[ValidationResult] = []

    for i in range(len(kernel_ids)):
        for j in range(i + 1, len(kernel_ids)):
            id_a, id_b = kernel_ids[i], kernel_ids[j]

            all_passed = True
            total_max_abs = 0.0
            total_max_rel = 0.0
            total_mismatched = 0
            total_elements = 0

            for name in kernel_outputs[id_a]:
                out_a = kernel_outputs[id_a][name]
                out_b = kernel_outputs[id_b][name]
                passed, max_abs, max_rel, mismatched = _compare_tensors(out_a, out_b, atol, rtol)
                all_passed = all_passed and passed
                total_max_abs = max(total_max_abs, max_abs)
                total_max_rel = max(total_max_rel, max_rel)
                total_mismatched += mismatched
                total_elements += out_a.numel()

            comparisons.append(
                ValidationResult(
                    kernel_a=id_a,
                    kernel_b=id_b,
                    passed=all_passed,
                    max_abs_diff=total_max_abs,
                    max_rel_diff=total_max_rel,
                    mismatched_elements=total_mismatched,
                    total_elements=total_elements,
                )
            )

    return ValidationReport(comparisons=comparisons)
