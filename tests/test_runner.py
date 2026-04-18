"""Tests for the benchmark runner."""

import pytest
import torch

from kernels_bench.runner import KernelResult, _timed_loop, run_benchmark, run_benchmark_quick
from kernels_bench.spec import TensorSpec


def _noop(*args):
    """A do-nothing function for timing tests."""
    pass


def _copy_kernel_fn(kernel, x, y):
    """Bench function that copies input to output."""
    y.copy_(x)


class FakeKernel:
    def double(self, y: torch.Tensor, x: torch.Tensor) -> None:
        y.copy_(x * 2)


@pytest.mark.gpu
def test_timed_loop_returns_correct_count(runtime, device):
    x = torch.randn(16, 16, device=device)
    times, _metrics = _timed_loop(_noop, [x], warmup=2, iterations=10, runtime=runtime)
    assert len(times) == 10
    assert all(t >= 0 for t in times)


@pytest.mark.gpu
def test_timed_loop_with_callback(runtime, device):
    steps: list[tuple[str, int, int]] = []

    def on_step(phase, current, total):
        steps.append((phase, current, total))

    x = torch.randn(16, 16, device=device)
    _timed_loop(_noop, [x], warmup=3, iterations=5, runtime=runtime, on_step=on_step)

    warmup_steps = [s for s in steps if s[0] == "warmup"]
    bench_steps = [s for s in steps if s[0] == "bench"]
    assert len(warmup_steps) == 3
    assert len(bench_steps) == 5
    assert warmup_steps[-1] == ("warmup", 3, 3)
    assert bench_steps[-1] == ("bench", 5, 5)


@pytest.mark.gpu
def test_run_benchmark(runtime, device):
    input_specs = [TensorSpec("x", shape=(32, 32), dtype=torch.float16, device=device)]
    output_specs = [
        TensorSpec("y", shape=(32, 32), dtype=torch.float16, device=device, role="output")
    ]

    times, _metrics = run_benchmark(
        bench_fn=_copy_kernel_fn,
        kernel=None,  # not used by _copy_kernel_fn
        input_specs=input_specs,
        output_specs=output_specs,
        warmup=2,
        iterations=5,
        runtime=runtime,
    )
    assert len(times) == 5


@pytest.mark.gpu
def test_run_benchmark_quick(runtime, device):
    specs = [
        TensorSpec("y", shape=(32, 32), dtype=torch.float16, device=device, role="output"),
        TensorSpec("x", shape=(32, 32), dtype=torch.float16, device=device, role="input"),
    ]
    times, _metrics = run_benchmark_quick(
        kernel=FakeKernel(),
        fn_name="double",
        specs=specs,
        warmup=2,
        iterations=5,
        runtime=runtime,
    )
    assert len(times) == 5


def test_kernel_result_stats():
    kr = KernelResult(
        kernel_id="test/kernel",
        params={"M": 1024},
        times_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert kr.mean_ms == 3.0
    assert kr.median_ms == 3.0
    assert kr.min_ms == 1.0
    assert kr.max_ms == 5.0
    assert kr.std_ms > 0


def test_kernel_result_single_iteration():
    kr = KernelResult(kernel_id="test/kernel", params={}, times_ms=[1.5])
    assert kr.mean_ms == 1.5
    assert kr.median_ms == 1.5
    assert kr.std_ms == 0.0
