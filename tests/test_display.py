"""Tests for the display module."""

from kernels_bench.display import (
    _format_comparison,
    _format_metrics,
    _format_params,
    _format_throughput,
    _make_bar,
    _pad_right,
    _truncate,
    _visible_len,
)
from kernels_bench.runner import KernelResult
from kernels_bench.runtime import RunMetrics


def test_visible_len_plain():
    assert _visible_len("hello") == 5


def test_visible_len_with_ansi():
    assert _visible_len("\x1b[1mhello\x1b[0m") == 5
    assert _visible_len("\x1b[38;2;244;183;63mtext\x1b[0m") == 4


def test_pad_right_plain():
    assert _pad_right("hi", 5) == "hi   "


def test_pad_right_with_ansi():
    s = "\x1b[1mhi\x1b[0m"
    padded = _pad_right(s, 5)
    assert _visible_len(padded) == 5
    assert padded.startswith("\x1b[1mhi\x1b[0m")


def test_make_bar_full():
    bar = _make_bar(1.0, 1.0, 10)
    assert bar == "\u2588" * 10


def test_make_bar_empty():
    bar = _make_bar(0.0, 1.0, 10)
    assert bar == "\u2591" * 10


def test_make_bar_half():
    bar = _make_bar(0.5, 1.0, 10)
    assert len(bar) == 10
    assert "\u2588" in bar
    assert "\u2591" in bar


def test_make_bar_zero_max():
    bar = _make_bar(5.0, 0.0, 10)
    assert bar == "\u2591" * 10


def test_format_params_empty():
    assert _format_params({}) == ""


def test_format_params():
    assert _format_params({"M": 1024, "N": 512}) == "M=1024, N=512"


def test_truncate_short():
    assert _truncate("hello", 10) == "hello"


def test_truncate_long():
    result = _truncate("a very long string", 10)
    assert len(result) == 10
    assert result.endswith("...")


def test_format_metrics_empty():
    assert _format_metrics(RunMetrics()) is None


def test_format_metrics_peak_mem_mb():
    m = RunMetrics(peak_memory_mb=128.0)
    assert _format_metrics(m) == "peak_mem=128.0 MB"


def test_format_metrics_peak_mem_gb_threshold():
    # >= 1024 MB switches to GB rendering
    m = RunMetrics(peak_memory_mb=2048.0)
    assert _format_metrics(m) == "peak_mem=2.00 GB"


def test_format_metrics_peak_mem_just_below_gb():
    m = RunMetrics(peak_memory_mb=1023.9)
    assert _format_metrics(m) == "peak_mem=1023.9 MB"


def test_format_metrics_util_only():
    m = RunMetrics(util_mean=66.2, util_peak=98.0, util_samples=30)
    assert _format_metrics(m) == "util=66% (peak 98%)"


def test_format_metrics_util_requires_both_mean_and_peak():
    # If only one of util_mean/util_peak is set, util is omitted
    assert _format_metrics(RunMetrics(util_mean=50.0)) is None
    assert _format_metrics(RunMetrics(util_peak=50.0)) is None


def test_format_metrics_combined():
    m = RunMetrics(peak_memory_mb=512.0, util_mean=75.4, util_peak=99.1, util_samples=40)
    assert _format_metrics(m) == "peak_mem=512.0 MB  util=75% (peak 99%)"


def _mk_kr(kernel_id: str, median: float, util_mean: float | None = None) -> KernelResult:
    metrics = RunMetrics(util_mean=util_mean) if util_mean is not None else RunMetrics()
    # Construct times_ms so median_ms equals `median` exactly (list of one).
    return KernelResult(kernel_id=kernel_id, params={}, times_ms=[median], metrics=metrics)


def test_format_comparison_no_util():
    fastest = _mk_kr("a", 1.0)
    slower = _mk_kr("b", 1.28)
    assert _format_comparison(slower, fastest) == "1.28x slower"


def test_format_comparison_with_util_both_set():
    fastest = _mk_kr("a", 1.0, util_mean=67.0)
    slower = _mk_kr("b", 1.28, util_mean=94.4)
    # Slower but much better util — useful signal for the user.
    assert _format_comparison(slower, fastest) == "1.28x slower  \u00b7  util 94% (fastest: 67%)"


def test_format_comparison_util_only_on_fastest_is_omitted():
    fastest = _mk_kr("a", 1.0, util_mean=80.0)
    slower = _mk_kr("b", 2.0)  # no util
    assert _format_comparison(slower, fastest) == "2.00x slower"


def test_format_comparison_util_only_on_slower_is_omitted():
    fastest = _mk_kr("a", 1.0)  # no util
    slower = _mk_kr("b", 2.0, util_mean=90.0)
    assert _format_comparison(slower, fastest) == "2.00x slower"


def test_format_comparison_identical_util():
    fastest = _mk_kr("a", 1.0, util_mean=85.0)
    slower = _mk_kr("b", 1.5, util_mean=85.0)
    assert _format_comparison(slower, fastest) == "1.50x slower  \u00b7  util 85% (fastest: 85%)"


def test_format_throughput_none_when_unset():
    kr = KernelResult(kernel_id="k", params={}, times_ms=[1.0])
    assert _format_throughput(kr) is None


def test_format_throughput_gflops_below_1tflop():
    # 500 GFLOPs in 1 ms = 500 GFLOP/s
    kr = KernelResult(kernel_id="k", params={}, times_ms=[1.0], flops=5 * 10**8)
    out = _format_throughput(kr)
    assert out == "500.0 GFLOP/s"


def test_format_throughput_tflops_scale():
    # 5e9 FLOPs in 1 ms = 5e12 FLOP/s = 5 TFLOP/s
    kr = KernelResult(kernel_id="k", params={}, times_ms=[1.0], flops=5 * 10**9)
    assert _format_throughput(kr) == "5.00 TFLOP/s"


def test_format_throughput_combined():
    # 2e9 flops/ms = 2 TFLOP/s; 8e8 bytes/ms = 800 GB/s.
    kr = KernelResult(
        kernel_id="k",
        params={},
        times_ms=[1.0],
        flops=2 * 10**9,
        bytes_per_iter=8 * 10**8,
    )
    assert _format_throughput(kr) == "2.00 TFLOP/s  800.0 GB/s"
