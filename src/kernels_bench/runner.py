"""Benchmark execution engine: tensor allocation, warmup, timing, stats."""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.benchmark import Timer

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime import RunMetrics
from kernels_bench.runtime._base import _NoopMetricsCollector
from kernels_bench.spec import TensorSpec
from kernels_bench.validate import ValidationReport

if TYPE_CHECKING:
    from kernels_bench.runtime import Runtime


def _quantile(xs: list[float], q: float) -> float:
    """Quantile via nearest-rank on sorted values. Matches HF kernels-benchmarks."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s) - 1
    return s[max(0, min(n, int(q * n)))]


@dataclasses.dataclass(frozen=True)
class KernelResult:
    """Timing results for a single kernel on a single param combination."""

    kernel_id: str
    params: dict[str, int]
    times_ms: list[float]
    metrics: RunMetrics = dataclasses.field(default_factory=RunMetrics)
    compile_ms: float | None = None
    flops: int | None = None
    bytes_per_iter: int | None = None

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def p10_ms(self) -> float:
        return _quantile(self.times_ms, 0.10)

    @property
    def p90_ms(self) -> float:
        return _quantile(self.times_ms, 0.90)

    @property
    def iqr_ms(self) -> float:
        """Interquartile range (p75 - p25) — a robust measure of variance."""
        return _quantile(self.times_ms, 0.75) - _quantile(self.times_ms, 0.25)

    @property
    def gflops_per_s(self) -> float | None:
        """Compute throughput in GFLOP/s from flops and the median time."""
        if self.flops is None or self.median_ms <= 0:
            return None
        return self.flops / (self.median_ms * 1e-3) / 1e9

    @property
    def gb_per_s(self) -> float | None:
        """Memory bandwidth in GB/s from bytes_per_iter and the median time."""
        if self.bytes_per_iter is None or self.median_ms <= 0:
            return None
        return self.bytes_per_iter / (self.median_ms * 1e-3) / 1e9

    @property
    def has_warnings(self) -> bool:
        """True when IQR > 10% of the median — flags a noisy measurement."""
        med = self.median_ms
        if med <= 0:
            return False
        return (self.iqr_ms / med) > 0.10


@dataclasses.dataclass(frozen=True)
class BenchResult:
    """Aggregated results for an entire benchmark run."""

    bench_name: str
    kernel_results: list[KernelResult]
    device: DeviceInfo | None = None
    validation: ValidationReport | None = None

    def fastest(self, params: dict[str, int] | None = None) -> KernelResult:
        """Return the fastest kernel result, optionally filtered by params."""
        candidates = self.kernel_results
        if params is not None:
            candidates = [r for r in candidates if r.params == params]
        return min(candidates, key=lambda r: r.median_ms)

    def to_csv_rows(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Flatten results into (header, rows) for CSV export.

        Each row is one (kernel_id, params) pair. Param keys vary by run, so we
        union them across all results to keep the column set stable.
        """
        param_keys = sorted({k for kr in self.kernel_results for k in kr.params})
        header = [
            "kernel_id",
            *param_keys,
            "median_ms",
            "p10_ms",
            "p90_ms",
            "iqr_ms",
            "has_warnings",
            "compile_ms",
            "gflops_per_s",
            "gb_per_s",
            "peak_memory_mb",
            "util_mean",
            "util_peak",
        ]
        rows: list[dict[str, Any]] = []
        for kr in self.kernel_results:
            row: dict[str, Any] = {"kernel_id": kr.kernel_id}
            for k in param_keys:
                row[k] = kr.params.get(k, "")
            row["median_ms"] = kr.median_ms
            row["p10_ms"] = kr.p10_ms
            row["p90_ms"] = kr.p90_ms
            row["iqr_ms"] = kr.iqr_ms
            row["has_warnings"] = kr.has_warnings
            row["compile_ms"] = kr.compile_ms
            row["gflops_per_s"] = kr.gflops_per_s
            row["gb_per_s"] = kr.gb_per_s
            row["peak_memory_mb"] = kr.metrics.peak_memory_mb
            row["util_mean"] = kr.metrics.util_mean
            row["util_peak"] = kr.metrics.util_peak
            rows.append(row)
        return header, rows

    def to_dict(self) -> dict[str, Any]:
        """Serialize results to a dict suitable for JSON export."""
        return {
            "bench_name": self.bench_name,
            "device": self.device.to_dict() if self.device else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "results": [
                {
                    "kernel_id": kr.kernel_id,
                    "params": kr.params,
                    "median_ms": kr.median_ms,
                    "mean_ms": kr.mean_ms,
                    "std_ms": kr.std_ms,
                    "min_ms": kr.min_ms,
                    "max_ms": kr.max_ms,
                    "p10_ms": kr.p10_ms,
                    "p90_ms": kr.p90_ms,
                    "iqr_ms": kr.iqr_ms,
                    "has_warnings": kr.has_warnings,
                    "times_ms": kr.times_ms,
                    "compile_ms": kr.compile_ms,
                    "flops": kr.flops,
                    "bytes_per_iter": kr.bytes_per_iter,
                    "gflops_per_s": kr.gflops_per_s,
                    "gb_per_s": kr.gb_per_s,
                    "metrics": kr.metrics.to_dict(),
                }
                for kr in self.kernel_results
            ],
        }


def _resolve_specs(specs: list[TensorSpec], params: dict[str, int]) -> list[TensorSpec]:
    """Resolve symbolic dims in specs using params."""
    return [s.resolve(params) if s.symbolic_dims else s for s in specs]


def _allocate_tensors(
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    device: str,
) -> dict[str, torch.Tensor]:
    """Allocate tensors for all specs, keyed by name."""
    tensors: dict[str, torch.Tensor] = {}
    for spec in input_specs:
        tensors[spec.name] = spec.allocate_input(device)
    for spec in output_specs:
        tensors[spec.name] = spec.allocate_output(device)
    return tensors


# Callback signature: (phase, current, total) where phase is "warmup" or "bench"
ProgressCallback = Callable[[str, int, int], None] | None


def profile_call(
    fn: Callable[..., Any],
    args: list[Any],
    runs: int = 3,
    label: str = "",
    row_limit: int = 20,
) -> str:
    """Run torch.profiler over a few invocations and return the key_averages table.

    Captures CPU/CUDA op time, kernel launches and memory ops — answers
    "where is the time going?" in a way the SM-util sampler can't.
    """
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with (
        profile(activities=activities, record_shapes=True, with_stack=False) as prof,
        record_function(label or "kernel"),
    ):
        for _ in range(runs):
            fn(*args)

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    return prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)


def _timed_loop(
    fn: Callable[..., Any],
    args: list[Any],
    warmup: int,
    iterations: int,
    runtime: Runtime,
    on_step: ProgressCallback = None,
    collect_metrics: bool = True,
) -> tuple[list[float], RunMetrics, float]:
    """Run compile + warmup + timed iterations.

    Returns (per-iter times in ms, metrics, compile_ms). The first call is
    timed separately as compile_ms so JIT/autotune cost is visible instead of
    being absorbed (or not) by the warmup window.

    Uses torch.utils.benchmark.Timer for every measurement: it handles CUDA
    synchronization, autograd state and stream context for us.

    When collect_metrics is False, the runtime's real collector is replaced with
    a no-op — peak_memory / util fields come back as None.
    """
    # We inject runtime.synchronize() into the stmt so each measurement bounds
    # the device queue. torch.utils.benchmark.Timer already synchronizes for
    # CUDA, but not for MPS — adding our own sync makes the timing path uniform.
    timer = Timer(
        stmt="fn(*args); sync()",
        globals={"fn": fn, "args": args, "sync": runtime.synchronize},
    )

    compile_m = timer.timeit(1)
    compile_ms = compile_m.mean * 1000.0

    for i in range(warmup):
        timer.timeit(1)
        if on_step:
            on_step("warmup", i + 1, warmup)
    runtime.synchronize()

    collector = runtime.create_metrics_collector() if collect_metrics else _NoopMetricsCollector()
    collector.start()

    times: list[float] = []
    try:
        for i in range(iterations):
            m = timer.timeit(1)
            times.append(m.mean * 1000.0)
            if on_step:
                on_step("bench", i + 1, iterations)
    finally:
        collector.stop()

    return times, collector.result(), compile_ms


def run_benchmark(
    bench_fn: Callable[..., Any],
    kernel: Any,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    warmup: int,
    iterations: int,
    runtime: Runtime,
    on_step: ProgressCallback = None,
    collect_metrics: bool = True,
    profile: bool = False,
    profile_label: str = "",
) -> tuple[list[float], RunMetrics, float]:
    """Run a user-defined benchmark function (kernel, *inputs, *outputs)."""
    tensors = _allocate_tensors(input_specs, output_specs, runtime.device)

    args: list[Any] = [kernel]
    args.extend(tensors[s.name] for s in input_specs)
    args.extend(tensors[s.name] for s in output_specs)

    result = _timed_loop(bench_fn, args, warmup, iterations, runtime, on_step, collect_metrics)
    if profile:
        print(f"\nPROFILE TRACE: {profile_label or 'kernel'}")
        print(profile_call(bench_fn, args, label=profile_label))
    return result


def run_benchmark_quick(
    kernel: Any,
    fn_name: str,
    specs: list[TensorSpec],
    warmup: int,
    iterations: int,
    runtime: Runtime,
    on_step: ProgressCallback = None,
    collect_metrics: bool = True,
    profile: bool = False,
    profile_label: str = "",
) -> tuple[list[float], RunMetrics, float]:
    """Run a kernel function directly with args in spec order.

    This is used by the `quick` command — no user-defined bench function needed.
    Tensors are allocated according to each spec's role and passed in order.
    """
    fn = getattr(kernel, fn_name)
    tensors = [spec.allocate(runtime.device) for spec in specs]
    result = _timed_loop(fn, tensors, warmup, iterations, runtime, on_step, collect_metrics)
    if profile:
        print(f"\nPROFILE TRACE: {profile_label or fn_name}")
        print(profile_call(fn, tensors, label=profile_label or fn_name))
    return result
