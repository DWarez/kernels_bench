"""Correctness validation — compare output tensors across kernels."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from kernels_bench.spec import TensorSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from kernels_bench.runtime import Runtime


REFERENCE_LABEL = "reference"
"""Result label / dict key for the reference baseline (e.g. plain PyTorch)."""


class ValidationError(ValueError):
    """Raised when --validate is requested but the outputs cannot be compared.

    e.g. a kernel produced no capturable output, or two kernels returned
    structurally different outputs. Distinct from a *mismatch* (which is a
    normal FAIL result), this signals the comparison itself can't be done.
    """


def _tensors_from_return(ret: Any) -> list[torch.Tensor]:
    """Extract output tensors from a kernel's return value.

    Functional kernels return their result instead of writing into a
    preallocated buffer. A single tensor, or a tuple/list of tensors (e.g.
    flash attention's ``(out, lse, ...)``), are all captured; non-tensor
    members (metadata, None) are ignored.
    """
    if isinstance(ret, torch.Tensor):
        return [ret]
    if isinstance(ret, (tuple, list)):
        return [t for t in ret if isinstance(t, torch.Tensor)]
    return []


def _named_return_tensors(ret: Any) -> dict[str, torch.Tensor]:
    """Name a return value's tensors for the dict-keyed bench comparison."""
    tensors = _tensors_from_return(ret)
    if len(tensors) == 1:
        return {"return": tensors[0]}
    return {f"return_{i}": t for i, t in enumerate(tensors)}


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

    Uses shared input tensors so all kernels get the same inputs. Outputs come
    from preallocated ``role=="output"`` buffers; if the kernel declares none,
    its return value is captured instead, so functional kernels (which return
    their result rather than writing into a buffer) are actually compared
    rather than silently passing over zero elements.
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

    ret = fn(*args)
    runtime.synchronize()
    if not output_tensors:
        output_tensors = _tensors_from_return(ret)
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

    Uses shared input tensors so all kernels get the same inputs. As in the
    quick path, a bench function that returns its result instead of writing
    into a declared output spec has its return value captured under synthetic
    ``return``/``return_N`` keys.
    """
    # Allocate fresh outputs on the same device as the inputs
    device = next(iter(input_tensors.values())).device.type
    output_tensors: dict[str, torch.Tensor] = {}
    for spec in output_specs:
        output_tensors[spec.name] = spec.allocate_output(device)

    args = [kernel]
    args.extend(input_tensors[s.name] for s in input_specs)
    args.extend(output_tensors[s.name] for s in output_specs)

    ret = bench_fn(*args)
    runtime.synchronize()
    if not output_tensors:
        output_tensors = _named_return_tensors(ret)
    return output_tensors


def _collect_outputs_ref(
    ref: Callable[..., Any],
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    input_tensors: dict[str, torch.Tensor],
    runtime: Runtime,
) -> dict[str, torch.Tensor]:
    """Run the functional reference once and key its outputs to match kernels.

    The reference takes the inputs in spec order and returns its result. When
    the bench declares output specs, the returned tensors are keyed by those
    names so they line up with kernels that write into named output buffers;
    otherwise they fall back to ``return``/``return_N`` keys (matching a
    functional bench fn).
    """
    ret = ref(*(input_tensors[s.name] for s in input_specs))
    runtime.synchronize()
    tensors = _tensors_from_return(ret)
    if output_specs and len(tensors) == len(output_specs):
        return {spec.name: t for spec, t in zip(output_specs, tensors, strict=True)}
    return _named_return_tensors(ret)


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
        outputs = _collect_outputs_quick(kernel, fn_name, specs, input_tensors, runtime)
        if not outputs:
            raise ValidationError(
                f"cannot validate {kernel_id!r}: {fn_name!r} produced no output tensors "
                "to compare. Declare an :output arg (e.g. y:1024,1024:float16:output), "
                "or have the function return its result."
            )
        kernel_outputs[kernel_id] = outputs

    # Pairwise comparison
    kernel_ids = list(kernels.keys())
    comparisons: list[ValidationResult] = []

    for i in range(len(kernel_ids)):
        for j in range(i + 1, len(kernel_ids)):
            id_a, id_b = kernel_ids[i], kernel_ids[j]
            outputs_a = kernel_outputs[id_a]
            outputs_b = kernel_outputs[id_b]
            if len(outputs_a) != len(outputs_b):
                raise ValidationError(
                    f"{id_a!r} and {id_b!r} returned different numbers of output tensors "
                    f"({len(outputs_a)} vs {len(outputs_b)}); cannot compare."
                )

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
    ref: Callable[..., Any] | None = None,
    ref_label: str = REFERENCE_LABEL,
) -> ValidationReport:
    """Validate that all kernels produce the same outputs for the run command.

    When ``ref`` (a functional ``(*inputs) -> output`` callable) is given, it is
    included as the first participant, so every kernel is compared against it —
    a ground-truth oracle, not just pairwise agreement. This also makes a single
    kernel validatable (kernel vs reference), which pairwise alone can't do.
    """
    # Allocate shared input tensors once
    input_tensors: dict[str, torch.Tensor] = {}
    for spec in input_specs:
        input_tensors[spec.name] = spec.allocate_input(runtime.device)

    # Collect outputs, reference first so comparisons read as reference-vs-kernel.
    kernel_outputs: dict[str, dict[str, torch.Tensor]] = {}
    if ref is not None:
        ref_outputs = _collect_outputs_ref(ref, input_specs, output_specs, input_tensors, runtime)
        if not ref_outputs:
            raise ValidationError(
                f"cannot validate against {ref_label!r}: the reference produced no output "
                "tensors. Make it return its result."
            )
        kernel_outputs[ref_label] = ref_outputs
    for kernel_id, kernel in kernels.items():
        outputs = _collect_outputs_bench(
            bench_fn, kernel, input_specs, output_specs, input_tensors, runtime
        )
        if not outputs:
            raise ValidationError(
                f"cannot validate {kernel_id!r}: the bench function produced no output "
                "tensors to compare. Declare an output TensorSpec (role='output'), or "
                "return the result from the bench function."
            )
        kernel_outputs[kernel_id] = outputs

    # Pairwise comparison (includes the reference when present)
    kernel_ids = list(kernel_outputs.keys())
    comparisons: list[ValidationResult] = []

    for i in range(len(kernel_ids)):
        for j in range(i + 1, len(kernel_ids)):
            id_a, id_b = kernel_ids[i], kernel_ids[j]
            if kernel_outputs[id_a].keys() != kernel_outputs[id_b].keys():
                raise ValidationError(
                    f"{id_a!r} and {id_b!r} produced different output tensors "
                    f"({sorted(kernel_outputs[id_a])} vs {sorted(kernel_outputs[id_b])}); "
                    "cannot compare."
                )

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
