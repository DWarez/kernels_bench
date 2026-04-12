"""Test that validation correctly detects matching and mismatching outputs."""

import torch

from kernels_bench.validate import ValidationReport, _compare_tensors, validate_quick
from kernels_bench.spec import TensorSpec


def test_compare_tensors_identical():
    a = torch.randn(100, 100, device="cuda", dtype=torch.float16)
    passed, max_abs, max_rel, mismatched = _compare_tensors(a, a.clone(), atol=1e-3, rtol=1e-3)
    assert passed
    assert max_abs == 0.0
    assert mismatched == 0


def test_compare_tensors_different():
    a = torch.ones(100, 100, device="cuda", dtype=torch.float16)
    b = torch.ones(100, 100, device="cuda", dtype=torch.float16) + 1.0  # off by 1.0
    passed, max_abs, max_rel, mismatched = _compare_tensors(a, b, atol=1e-3, rtol=1e-3)
    assert not passed
    assert max_abs >= 1.0
    assert mismatched == 10000


def test_compare_tensors_within_tolerance():
    a = torch.ones(100, 100, device="cuda", dtype=torch.float16)
    b = a + 1e-4  # tiny diff, within default tolerance
    passed, max_abs, max_rel, mismatched = _compare_tensors(a, b, atol=1e-3, rtol=1e-3)
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


def test_validate_quick_matching():
    specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, role="output"),
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, role="input"),
    ]
    kernels = {
        "correct-a": FakeKernelCorrect(),
        "correct-b": FakeKernelAlsoCorrect(),
    }
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs)
    assert report.all_passed
    assert len(report.comparisons) == 1
    assert report.comparisons[0].passed


def test_validate_quick_mismatching():
    specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, role="output"),
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, role="input"),
    ]
    kernels = {
        "correct": FakeKernelCorrect(),
        "wrong": FakeKernelWrong(),
    }
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs)
    assert not report.all_passed
    assert len(report.comparisons) == 1
    assert not report.comparisons[0].passed
    assert report.comparisons[0].mismatched_elements > 0


def test_validate_quick_three_kernels():
    """With 3 kernels, we get 3 pairwise comparisons."""
    specs = [
        TensorSpec("y", shape=(64, 64), dtype=torch.float16, role="output"),
        TensorSpec("x", shape=(64, 64), dtype=torch.float16, role="input"),
    ]
    kernels = {
        "correct-a": FakeKernelCorrect(),
        "correct-b": FakeKernelAlsoCorrect(),
        "wrong": FakeKernelWrong(),
    }
    report = validate_quick(kernels=kernels, fn_name="my_fn", specs=specs)
    assert not report.all_passed
    assert len(report.comparisons) == 3

    # correct-a vs correct-b should pass
    assert report.comparisons[0].passed
    # correct-a vs wrong should fail
    assert not report.comparisons[1].passed
    # correct-b vs wrong should fail
    assert not report.comparisons[2].passed
