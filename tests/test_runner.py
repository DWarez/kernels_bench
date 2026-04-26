"""Tests for the benchmark runner."""

import pytest
import torch

from kernels_bench.runner import (
    BenchResult,
    KernelResult,
    _timed_loop,
    profile_call,
    run_benchmark,
    run_benchmark_quick,
)
from kernels_bench.runtime import RunMetrics
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
    times, _metrics, compile_ms = _timed_loop(
        _noop, [x], warmup=2, iterations=10, runtime=runtime
    )
    assert len(times) == 10
    assert all(t >= 0 for t in times)
    assert compile_ms >= 0


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

    times, _metrics, _compile = run_benchmark(
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
    times, _metrics, _compile = run_benchmark_quick(
        kernel=FakeKernel(),
        fn_name="double",
        specs=specs,
        warmup=2,
        iterations=5,
        runtime=runtime,
    )
    assert len(times) == 5


@pytest.mark.gpu
def test_run_benchmark_quick_collect_metrics_false_returns_empty(runtime, device):
    """collect_metrics=False swaps in the noop collector — all fields are empty."""
    specs = [
        TensorSpec("y", shape=(32, 32), dtype=torch.float16, device=device, role="output"),
        TensorSpec("x", shape=(32, 32), dtype=torch.float16, device=device, role="input"),
    ]
    times, metrics, _compile = run_benchmark_quick(
        kernel=FakeKernel(),
        fn_name="double",
        specs=specs,
        warmup=2,
        iterations=5,
        runtime=runtime,
        collect_metrics=False,
    )
    assert len(times) == 5
    # Every field stays at its default — we didn't sample or read memory stats.
    from kernels_bench.runtime import RunMetrics as _RM

    assert metrics == _RM()


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


def test_kernel_result_quantiles():
    # Sorted: 1..11 → p10≈1, p50≈6, p90≈10. Nearest-rank quantile.
    kr = KernelResult(
        kernel_id="k", params={}, times_ms=[float(i) for i in range(1, 12)]
    )
    assert kr.p10_ms == 2.0
    assert kr.median_ms == 6.0
    assert kr.p90_ms == 10.0
    # Nearest-rank with n=10: p25 → s[2]=3, p75 → s[7]=8 → IQR = 5.
    assert kr.iqr_ms == 5.0


def test_kernel_result_has_warnings_noisy():
    # Wide spread: median 1.0, p25=0.5, p75=10 → IQR/median = 9.5 → noisy.
    kr = KernelResult(kernel_id="k", params={}, times_ms=[0.5, 0.5, 1.0, 10.0, 10.0])
    assert kr.has_warnings is True


def test_kernel_result_has_warnings_quiet():
    kr = KernelResult(
        kernel_id="k", params={}, times_ms=[1.00, 1.01, 1.00, 1.01, 1.00, 1.01]
    )
    assert kr.has_warnings is False


def test_kernel_result_has_warnings_zero_median():
    # Defensive: never divide by zero.
    kr = KernelResult(kernel_id="k", params={}, times_ms=[0.0, 0.0, 0.0])
    assert kr.has_warnings is False


def test_kernel_result_default_metrics_empty():
    """Omitting metrics gives an empty RunMetrics rather than None."""
    kr = KernelResult(kernel_id="test/kernel", params={}, times_ms=[1.0])
    assert isinstance(kr.metrics, RunMetrics)
    assert kr.metrics == RunMetrics()


def test_bench_result_to_dict_includes_metrics():
    metrics = RunMetrics(peak_memory_mb=128.0, util_mean=72.5, util_peak=95.0, util_samples=20)
    kr = KernelResult(
        kernel_id="test/kernel",
        params={"M": 1024},
        times_ms=[1.0, 2.0, 3.0],
        metrics=metrics,
    )
    result = BenchResult(bench_name="b", kernel_results=[kr])
    d = result.to_dict()

    assert d["results"][0]["metrics"] == {
        "peak_memory_mb": 128.0,
        "util_mean": 72.5,
        "util_peak": 95.0,
        "util_samples": 20,
    }


def test_bench_result_to_dict_metrics_all_none_when_missing():
    """KernelResult without explicit metrics still serializes a metrics block."""
    kr = KernelResult(kernel_id="k", params={}, times_ms=[1.0])
    result = BenchResult(bench_name="b", kernel_results=[kr])
    d = result.to_dict()
    m = d["results"][0]["metrics"]
    assert m == {
        "peak_memory_mb": None,
        "util_mean": None,
        "util_peak": None,
        "util_samples": 0,
    }


def test_bench_result_to_dict_includes_compile_ms():
    kr = KernelResult(kernel_id="k", params={}, times_ms=[1.0], compile_ms=4.2)
    d = BenchResult(bench_name="b", kernel_results=[kr]).to_dict()
    assert d["results"][0]["compile_ms"] == 4.2


def test_profile_call_returns_table_string():
    """profile_call should produce a non-empty op breakdown for a trivial fn."""
    x = torch.randn(8, 8)

    def add_one(t):
        return t + 1

    table = profile_call(add_one, [x], runs=2, label="add_one")
    assert isinstance(table, str)
    # key_averages tables include a "Self" header — cheap presence check.
    assert "Self" in table


def test_bench_result_to_dict_compile_ms_none_when_missing():
    kr = KernelResult(kernel_id="k", params={}, times_ms=[1.0])
    d = BenchResult(bench_name="b", kernel_results=[kr]).to_dict()
    assert d["results"][0]["compile_ms"] is None
