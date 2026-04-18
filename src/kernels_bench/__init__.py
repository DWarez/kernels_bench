"""Benchmark tool for HuggingFace Kernels."""

__version__ = "0.1.0"

from kernels_bench.bench import Bench
from kernels_bench.device import DeviceInfo
from kernels_bench.display import print_results
from kernels_bench.runner import BenchResult, KernelResult
from kernels_bench.runtime import CUDARuntime, MPSRuntime, RunMetrics, Runtime, detect_runtime
from kernels_bench.spec import TensorSpec
from kernels_bench.validate import ValidationReport, ValidationResult

__all__ = [
    "Bench",
    "BenchResult",
    "CUDARuntime",
    "DeviceInfo",
    "KernelResult",
    "MPSRuntime",
    "RunMetrics",
    "Runtime",
    "TensorSpec",
    "ValidationReport",
    "ValidationResult",
    "detect_runtime",
    "print_results",
]
