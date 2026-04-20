"""Benchmark execution engine: tensor allocation, warmup, timing, stats."""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from kernels_bench.device import DeviceInfo
from kernels_bench.runtime import RunMetrics
from kernels_bench.spec import TensorSpec
from kernels_bench.validate import ValidationReport

if TYPE_CHECKING:
    from kernels_bench.runtime import Runtime


@dataclasses.dataclass(frozen=True)
class KernelResult:
    """Timing results for a single kernel on a single param combination."""

    kernel_id: str
    params: dict[str, int]
    times_ms: list[float]
    metrics: RunMetrics = dataclasses.field(default_factory=RunMetrics)

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
                    "times_ms": kr.times_ms,
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


def _timed_loop(
    fn: Callable[..., Any],
    args: list[Any],
    warmup: int,
    iterations: int,
    runtime: Runtime,
    on_step: ProgressCallback = None,
) -> tuple[list[float], RunMetrics]:
    """Run warmup + timed iterations; return per-iter times (ms) and metrics."""
    for i in range(warmup):
        fn(*args)
        if on_step:
            on_step("warmup", i + 1, warmup)
    runtime.synchronize()

    collector = runtime.create_metrics_collector()
    collector.start()

    times: list[float] = []
    try:
        for i in range(iterations):
            timer = runtime.create_timer()
            timer.record_start()
            fn(*args)
            timer.record_end()

            runtime.synchronize()
            times.append(timer.elapsed_ms())
            if on_step:
                on_step("bench", i + 1, iterations)
    finally:
        collector.stop()

    return times, collector.result()


def run_benchmark(
    bench_fn: Callable[..., Any],
    kernel: Any,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    warmup: int,
    iterations: int,
    runtime: Runtime,
    on_step: ProgressCallback = None,
) -> tuple[list[float], RunMetrics]:
    """Run a user-defined benchmark function (kernel, *inputs, *outputs)."""
    tensors = _allocate_tensors(input_specs, output_specs, runtime.device)

    args: list[Any] = [kernel]
    args.extend(tensors[s.name] for s in input_specs)
    args.extend(tensors[s.name] for s in output_specs)

    return _timed_loop(bench_fn, args, warmup, iterations, runtime, on_step)


def run_benchmark_quick(
    kernel: Any,
    fn_name: str,
    specs: list[TensorSpec],
    warmup: int,
    iterations: int,
    runtime: Runtime,
    on_step: ProgressCallback = None,
) -> tuple[list[float], RunMetrics]:
    """Run a kernel function directly with args in spec order.

    This is used by the `quick` command — no user-defined bench function needed.
    Tensors are allocated according to each spec's role and passed in order.
    """
    fn = getattr(kernel, fn_name)
    tensors = [spec.allocate(runtime.device) for spec in specs]
    return _timed_loop(fn, tensors, warmup, iterations, runtime, on_step)
