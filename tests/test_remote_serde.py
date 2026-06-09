"""Round-trip tests for result deserialization (the remote-result linchpin).

A remote job emits ``BenchResult.to_dict()`` as JSON; the launcher must rebuild
an equivalent object locally. These tests assert ``from_dict(to_dict(x))``
reproduces the same serialized form for every result dataclass.
"""

from kernels_bench.device import DeviceInfo
from kernels_bench.runner import BenchResult, KernelResult
from kernels_bench.runtime import RunMetrics
from kernels_bench.validate import ValidationReport, ValidationResult


def _device() -> DeviceInfo:
    return DeviceInfo(
        gpu_name="NVIDIA H200",
        runtime_name="CUDA",
        runtime_version="12.8",
        driver_version="(9, 0)",
        torch_version="2.11.0",
        gpu_memory_gb=141.0,
        python_version="3.12.13",
    )


def _kernel_result() -> KernelResult:
    return KernelResult(
        kernel_id="org/a@dev",
        params={"M": 1024, "N": 1024},
        times_ms=[0.10, 0.12, 0.11, 0.13],
        metrics=RunMetrics(peak_memory_mb=64.0, util_mean=87.0, util_peak=99.0, util_samples=42),
        compile_ms=5.0,
        flops=2_000_000,
        bytes_per_iter=4096,
    )


def test_run_metrics_round_trip():
    m = RunMetrics(peak_memory_mb=12.5, util_mean=80.0, util_peak=95.0, util_samples=7)
    assert RunMetrics.from_dict(m.to_dict()) == m


def test_run_metrics_round_trip_with_nones():
    m = RunMetrics()  # all-None (e.g. metrics disabled / MPS)
    assert RunMetrics.from_dict(m.to_dict()) == m


def test_device_info_round_trip():
    d = _device()
    assert DeviceInfo.from_dict(d.to_dict()) == d


def test_validation_result_round_trip():
    v = ValidationResult("org/a", "org/b", True, 1e-4, 2e-3, 0, 1_048_576)
    assert ValidationResult.from_dict(v.to_dict()) == v


def test_validation_report_round_trip():
    report = ValidationReport(
        comparisons=[
            ValidationResult("org/a", "org/b", True, 1e-4, 2e-3, 0, 1024),
            ValidationResult("org/a", "org/c", False, 0.5, 0.9, 17, 1024),
        ]
    )
    rebuilt = ValidationReport.from_dict(report.to_dict())
    assert rebuilt == report
    assert rebuilt.all_passed is False


def test_kernel_result_round_trip_via_bench_result():
    # KernelResult is serialized inline by BenchResult.to_dict, so round-trip it
    # through that payload shape.
    kr = _kernel_result()
    entry = BenchResult("f", [kr]).to_dict()["results"][0]
    rebuilt = KernelResult.from_dict(entry)
    # Derived stats recompute from times_ms — compare them explicitly.
    assert rebuilt.kernel_id == kr.kernel_id
    assert rebuilt.params == kr.params
    assert rebuilt.times_ms == kr.times_ms
    assert rebuilt.metrics == kr.metrics
    assert rebuilt.compile_ms == kr.compile_ms
    assert rebuilt.flops == kr.flops
    assert rebuilt.bytes_per_iter == kr.bytes_per_iter
    assert abs(rebuilt.median_ms - kr.median_ms) < 1e-12


def test_bench_result_full_round_trip():
    br = BenchResult(
        bench_name="gelu",
        kernel_results=[_kernel_result(), _kernel_result()],
        device=_device(),
        validation=ValidationReport(
            comparisons=[ValidationResult("org/a", "org/b", True, 1e-4, 2e-3, 0, 1024)]
        ),
    )
    # to_dict is the wire format; it must be identical after a round-trip.
    assert BenchResult.from_dict(br.to_dict()).to_dict() == br.to_dict()


def test_bench_result_round_trip_minimal():
    br = BenchResult(bench_name="f", kernel_results=[_kernel_result()])
    rebuilt = BenchResult.from_dict(br.to_dict())
    assert rebuilt.device is None
    assert rebuilt.validation is None
    assert rebuilt.to_dict() == br.to_dict()
