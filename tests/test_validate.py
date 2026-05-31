"""Test that validation correctly detects matching and mismatching outputs."""

import pytest
import torch

from kernels_bench.spec import TensorSpec
from kernels_bench.validate import (
    REFERENCE_LABEL,
    ValidationError,
    _compare_tensors,
    validate_bench,
    validate_quick,
)


@pytest.mark.gpu
def test_compare_tensors_identical(device):
    a = torch.randn(100, 100, device=device, dtype=torch.float16)
    passed, max_abs, _max_rel, mismatched = _compare_tensors(a, a.clone(), atol=1e-3, rtol=1e-3)
    assert passed
    assert max_abs == 0.0
    assert mismatched == 0


@pytest.mark.gpu
def test_compare_tensors_different(device):
    a = torch.ones(100, 100, device=device, dtype=torch.float16)
    b = torch.ones(100, 100, device=device, dtype=torch.float16) + 1.0  # off by 1.0
    passed, max_abs, _max_rel, mismatched = _compare_tensors(a, b, atol=1e-3, rtol=1e-3)
    assert not passed
    assert max_abs >= 1.0
    assert mismatched == 10000


@pytest.mark.gpu
def test_compare_tensors_within_tolerance(device):
    a = torch.ones(100, 100, device=device, dtype=torch.float16)
    b = a + 1e-4  # tiny diff, within default tolerance
    passed, _max_abs, _max_rel, _mismatched = _compare_tensors(a, b, atol=1e-3, rtol=1e-3)
    assert passed


class FakeKernelCorrect:
    """Fake kernel that computes y = x * 2."""

    def my_fn(self, y: torch.Tensor, x: torch.Tensor) -> None:
        y.copy_(x * 2)


class FakeKernelWrong:
    """Fake kernel that computes y = x * 3 (intentionally wrong)."""

    def my_fn(self, y: torch.Tensor, x: torch.Tensor) -> None:
        y.copy_(x * 3)


class FakeKernelAlsoCorrect:
    """Another kernel that computes y = x * 2 (same as Correct)."""

    def my_fn(self, y: torch.Tensor, x: torch.Tensor) -> None:
        y.copy_(x * 2)


@pytest.mark.gpu
def test_validate_quick_matching(runtime, device):
    specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, device=device, role="output"),
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input"),
    ]
    kernels = {
        "correct-a": FakeKernelCorrect(),
        "correct-b": FakeKernelAlsoCorrect(),
    }
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)
    assert report.all_passed
    assert len(report.comparisons) == 1
    assert report.comparisons[0].passed


@pytest.mark.gpu
def test_validate_quick_mismatching(runtime, device):
    specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, device=device, role="output"),
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input"),
    ]
    kernels = {
        "correct": FakeKernelCorrect(),
        "wrong": FakeKernelWrong(),
    }
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)
    assert not report.all_passed
    assert len(report.comparisons) == 1
    assert not report.comparisons[0].passed
    assert report.comparisons[0].mismatched_elements > 0


class FakeKernelFunctional:
    """Functional kernel: returns y = x * 2 instead of writing into a buffer."""

    def my_fn(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class FakeKernelFunctionalWrong:
    """Functional kernel that returns x * 3 (intentionally wrong)."""

    def my_fn(self, x: torch.Tensor) -> torch.Tensor:
        return x * 3


class FakeKernelTupleReturn:
    """Returns a tuple (out, aux) like flash attention's (out, lse)."""

    def my_fn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 2, x.sum(dim=-1)


class FakeKernelInPlace:
    """Mutates an input in place and returns None — nothing capturable."""

    def my_fn(self, x: torch.Tensor) -> None:
        x.mul_(2)


@pytest.mark.gpu
def test_validate_quick_functional_return_matching(runtime, device):
    """A kernel that *returns* its output is captured and actually compared."""
    specs = [TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")]
    kernels = {"a": FakeKernelFunctional(), "b": FakeKernelFunctional()}
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)
    assert report.all_passed
    # The whole point: it compared real elements, not 0.
    assert report.comparisons[0].total_elements == 64 * 64


@pytest.mark.gpu
def test_validate_quick_functional_return_mismatching(runtime, device):
    specs = [TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")]
    kernels = {"right": FakeKernelFunctional(), "wrong": FakeKernelFunctionalWrong()}
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)
    assert not report.all_passed
    assert report.comparisons[0].mismatched_elements > 0


@pytest.mark.gpu
def test_validate_quick_tuple_return(runtime, device):
    """Every tensor in a tuple return is compared."""
    specs = [TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")]
    kernels = {"a": FakeKernelTupleReturn(), "b": FakeKernelTupleReturn()}
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)
    assert report.all_passed
    assert report.comparisons[0].total_elements == 64 * 64 + 64  # out + aux


@pytest.mark.gpu
def test_validate_quick_no_output_raises(runtime, device):
    """In-place kernel returning None yields nothing to compare → loud error, not PASS."""
    specs = [TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")]
    kernels = {"a": FakeKernelInPlace(), "b": FakeKernelInPlace()}
    with pytest.raises(ValidationError, match="no output tensors"):
        validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)


@pytest.mark.gpu
def test_validate_quick_mismatched_output_count_raises(runtime, device):
    specs = [TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")]
    kernels = {"single": FakeKernelFunctional(), "tuple": FakeKernelTupleReturn()}
    with pytest.raises(ValidationError, match="different numbers of output"):
        validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)


@pytest.mark.gpu
def test_validate_bench_functional_return(runtime, device):
    """The bench path also captures a returned result when no output spec is declared."""

    def bench_fn(kernel, x):
        return kernel.my_fn(x)

    input_specs = [
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")
    ]
    kernels = {"a": FakeKernelFunctional(), "b": FakeKernelFunctional()}
    report = validate_bench(bench_fn, kernels, input_specs, [], runtime)
    assert report.all_passed
    assert report.comparisons[0].total_elements == 64 * 64


@pytest.mark.gpu
def test_validate_bench_no_output_raises(runtime, device):
    def bench_fn(kernel, x):
        kernel.my_fn(x)  # in-place, returns None

    input_specs = [
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")
    ]
    kernels = {"a": FakeKernelInPlace(), "b": FakeKernelInPlace()}
    with pytest.raises(ValidationError, match="no output tensors"):
        validate_bench(bench_fn, kernels, input_specs, [], runtime)


@pytest.mark.gpu
def test_validate_quick_three_kernels(runtime, device):
    """With 3 kernels, we get 3 pairwise comparisons."""
    specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, device=device, role="output"),
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input"),
    ]
    kernels = {
        "correct-a": FakeKernelCorrect(),
        "correct-b": FakeKernelAlsoCorrect(),
        "wrong": FakeKernelWrong(),
    }
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs, runtime=runtime)
    assert not report.all_passed
    assert len(report.comparisons) == 3

    # correct-a vs correct-b should pass
    assert report.comparisons[0].passed
    # correct-a vs wrong should fail
    assert not report.comparisons[1].passed
    # correct-b vs wrong should fail
    assert not report.comparisons[2].passed


def _bench_fn(kernel, x, y):
    """Buffer-style bench fn: delegates to kernel.my_fn(y, x)."""
    kernel.my_fn(y, x)


@pytest.mark.gpu
def test_validate_bench_reference_is_oracle(runtime, device):
    """Two kernels that agree but are both wrong are caught by the reference."""

    def ref(x):
        return x * 3  # ground truth; both kernels compute x * 2

    input_specs = [
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")
    ]
    output_specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, device=device, role="output")
    ]
    kernels = {"a": FakeKernelCorrect(), "b": FakeKernelAlsoCorrect()}
    report = validate_bench(_bench_fn, kernels, input_specs, output_specs, runtime, ref=ref)

    # 3 comparisons: reference-vs-a, reference-vs-b, a-vs-b.
    assert len(report.comparisons) == 3
    # a and b agree with each other...
    ab = next(c for c in report.comparisons if {c.kernel_a, c.kernel_b} == {"a", "b"})
    assert ab.passed
    # ...but both disagree with the reference, so validation does not pass overall.
    ref_comps = [c for c in report.comparisons if REFERENCE_LABEL in (c.kernel_a, c.kernel_b)]
    assert len(ref_comps) == 2
    assert all(not c.passed for c in ref_comps)
    assert not report.all_passed


@pytest.mark.gpu
def test_validate_bench_reference_validates_single_kernel(runtime, device):
    """A single kernel can be validated against the reference (pairwise can't)."""

    def ref(x):
        return x * 2  # matches the kernel

    input_specs = [
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, device=device, role="input")
    ]
    output_specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, device=device, role="output")
    ]
    report = validate_bench(
        _bench_fn, {"a": FakeKernelCorrect()}, input_specs, output_specs, runtime, ref=ref
    )
    assert report.all_passed
    assert len(report.comparisons) == 1
    assert report.comparisons[0].kernel_a == REFERENCE_LABEL
    assert report.comparisons[0].total_elements == 64 * 64
