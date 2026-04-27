"""Box-drawing display for benchmark results, inspired by hf-mem."""

from __future__ import annotations

import re
import sys

from kernels_bench import __version__
from kernels_bench.runner import BenchResult, KernelResult
from kernels_bench.runtime import RunMetrics

# Box-drawing characters
BOX = {
    "tl": "\u250c",
    "tr": "\u2510",
    "bl": "\u2514",
    "br": "\u2518",
    "ht": "\u2500",
    "vt": "\u2502",
    "tsep": "\u252c",
    "bsep": "\u2534",
    "lm": "\u251c",
    "rm": "\u2524",
    "mm": "\u253c",
}

# ANSI escape codes
COLOR = "\x1b[38;2;244;183;63m"  # Amber (matching hf-mem)
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[38;2;130;224;170m"
RED = "\x1b[38;2;255;100;100m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    """Length of a string ignoring ANSI escape codes."""
    return len(_ANSI_RE.sub("", s))


def _pad_right(s: str, width: int) -> str:
    """Pad a string (which may contain ANSI codes) to a visible width."""
    pad = width - _visible_len(s)
    return s + " " * max(0, pad)


def _print_color(text: str) -> None:
    print(f"{COLOR}{text}{RESET}")


def _make_bar(value: float, max_value: float, width: int) -> str:
    """Create a progress bar using block characters."""
    if max_value <= 0:
        return "\u2591" * width
    frac = min(max(value / max_value, 0.0), 1.0)
    filled = round(frac * width)
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _print_header(width: int, badge: str | None = None) -> None:
    inner = width - 2
    if badge:
        badge_text = f" {badge} "
        fill_len = max(0, inner - len(badge_text))
        top = BOX["tl"] + (BOX["tsep"] * fill_len) + badge_text + BOX["tr"]
    else:
        top = BOX["tl"] + (BOX["tsep"] * inner) + BOX["tr"]
    _print_color(top)


def _print_centered(text: str, width: int) -> None:
    inner = width - 2
    vis_len = _visible_len(text)
    pad_left = (inner - vis_len) // 2
    pad_right = inner - vis_len - pad_left
    _print_color(f"{BOX['vt']}{' ' * pad_left}{text}{' ' * pad_right}{BOX['vt']}")


def _print_divider(width: int, style: str = "mid") -> None:
    inner = width - 2
    match style:
        case "top":
            left, fill, right = BOX["lm"], BOX["tsep"], BOX["rm"]
        case "mid":
            left, fill, right = BOX["lm"], BOX["ht"], BOX["rm"]
        case "bottom":
            left, fill, right = BOX["bl"], BOX["bsep"], BOX["br"]
        case "section":
            left, fill, right = BOX["lm"], BOX["bsep"], BOX["rm"]
        case _:
            left, fill, right = BOX["lm"], BOX["ht"], BOX["rm"]
    _print_color(left + fill * inner + right)


def _print_row(label: str, value: str, width: int, label_width: int) -> None:
    """Print a two-column row: | LABEL | VALUE |"""
    inner = width - 2
    data_width = inner - label_width - 3  # 3 accounts for " ", "|", " " between columns
    label_fmt = _pad_right(label, label_width)
    value_fmt = _pad_right(value, data_width)
    _print_color(f"{BOX['vt']} {label_fmt}{BOX['vt']} {value_fmt}{BOX['vt']}")


def _print_row_divider(width: int, label_width: int, style: str = "mid") -> None:
    inner = width - 2
    data_width = inner - label_width - 3
    match style:
        case "top":
            left, mid, right = BOX["lm"], BOX["tsep"], BOX["rm"]
        case "bottom":
            left, mid, right = BOX["bl"], BOX["bsep"], BOX["br"]
        case _:
            left, mid, right = BOX["lm"], BOX["mm"], BOX["rm"]
    name_section = BOX["ht"] * (label_width + 1)
    data_section = BOX["ht"] * (data_width + 1)
    _print_color(f"{left}{name_section}{mid}{data_section}{right}")


def _format_params(params: dict[str, int]) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _truncate(text: str, max_len: int) -> str:
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _format_comparison(kr: KernelResult, fastest: KernelResult) -> str:
    """Format the comparison line for a non-fastest kernel result.

    Always shows the slowdown factor. When utilization data is available for
    both this kernel and the fastest, appends a util comparison so users can
    tell whether a slower kernel is less efficient or just doing more work.
    """
    slowdown = kr.median_ms / fastest.median_ms
    base = f"{slowdown:.2f}x slower"

    u_kr = kr.metrics.util_mean
    u_fast = fastest.metrics.util_mean
    if u_kr is not None and u_fast is not None:
        base += f"  \u00b7  util {u_kr:.0f}% (fastest: {u_fast:.0f}%)"
    return base


def _format_throughput(kr: KernelResult) -> str | None:
    """Format compute and bandwidth throughput on one line, or None if neither."""
    parts: list[str] = []
    if kr.gflops_per_s is not None:
        gflops = kr.gflops_per_s
        if gflops >= 1000:
            parts.append(f"{gflops / 1000:.2f} TFLOP/s")
        else:
            parts.append(f"{gflops:.1f} GFLOP/s")
    if kr.gb_per_s is not None:
        parts.append(f"{kr.gb_per_s:.1f} GB/s")
    if not parts:
        return None
    return "  ".join(parts)


def _format_metrics(m: RunMetrics) -> str | None:
    """Format RunMetrics as a single short line, or None if nothing to show."""
    parts: list[str] = []
    if m.peak_memory_mb is not None:
        if m.peak_memory_mb >= 1024:
            parts.append(f"peak_mem={m.peak_memory_mb / 1024:.2f} GB")
        else:
            parts.append(f"peak_mem={m.peak_memory_mb:.1f} MB")
    if m.util_mean is not None and m.util_peak is not None:
        parts.append(f"util={m.util_mean:.0f}% (peak {m.util_peak:.0f}%)")
    if not parts:
        return None
    return "  ".join(parts)


SWEEP_SUMMARY_THRESHOLD = 4
"""Switch to compact per-kernel summary when there are this many param combos or more."""


def _print_sweep_summary(
    result: BenchResult,
    total_width: int,
) -> None:
    """One row per param combo per kernel — readable at a glance for sweeps.

    Grouped visually by kernel so scaling across the swept dim is easy to spot.
    Bar width is normalized within each kernel to highlight relative cost.
    """
    by_kernel: dict[str, list[KernelResult]] = {}
    for kr in result.kernel_results:
        by_kernel.setdefault(kr.kernel_id, []).append(kr)

    # Label column width sized to the longest params string so the bars line up.
    max_params_len = max(len(_format_params(kr.params)) for kr in result.kernel_results)
    label_width = max(max_params_len + 1, 12)
    inner = total_width - 2
    data_col_width = inner - label_width - 3
    bar_width = max(data_col_width - 24, 16)  # leave room for "0.000 ms  " + GB/s suffix

    kernels = list(by_kernel.items())
    for ki, (kernel_id, rows) in enumerate(kernels):
        _print_divider(total_width, "section")
        _print_centered(kernel_id, total_width)
        _print_row_divider(total_width, label_width, "top")

        slowest = max(r.median_ms for r in rows)
        for i, kr in enumerate(rows):
            params_str = _format_params(kr.params) or "(no params)"
            bar = _make_bar(kr.median_ms, slowest, bar_width)
            median_text = f"{kr.median_ms:.3f} ms"
            value = f"{median_text}  {bar}"
            if kr.gb_per_s is not None:
                value += f"  {DIM}{kr.gb_per_s:.1f} GB/s{RESET}{COLOR}"
            _print_row(params_str, value, total_width, label_width)
            if i < len(rows) - 1:
                _print_row_divider(total_width, label_width)

        # Last kernel uses "bottom" to close the outer box; intermediate
        # kernels use a section-style divider so the next block flows cleanly.
        is_last = ki == len(kernels) - 1
        _print_row_divider(total_width, label_width, "bottom" if is_last else "mid")


def print_results(result: BenchResult) -> None:
    """Print benchmark results in hf-mem box-drawing style."""
    kernel_results = result.kernel_results
    if not kernel_results:
        print("No results to display.")
        return

    # Group results by param combination
    param_groups: dict[str, list[KernelResult]] = {}
    for kr in kernel_results:
        key = _format_params(kr.params)
        param_groups.setdefault(key, []).append(kr)

    # Compute layout widths
    kernel_ids = list({kr.kernel_id for kr in kernel_results})
    max_kernel_len = max(len(k) for k in kernel_ids)
    label_width = max(max_kernel_len + 1, 16)  # +1 for breathing room
    bar_width = 40
    total_width = label_width + bar_width + 20 + 5  # 20 for timing text, 5 for borders
    total_width = max(total_width, 60)

    # Derive actual bar_width from total
    inner = total_width - 2
    data_col_width = inner - label_width - 3
    bar_width = max(data_col_width - 14, 20)  # 14 chars for "999.999 ms  "

    # Clear any leftover progress bar output from kernel downloads
    sys.stderr.flush()
    sys.stdout.flush()
    print()

    # Header
    _print_header(total_width, badge=f"kernels-bench v{__version__}")
    _print_centered("KERNEL BENCHMARK RESULTS", total_width)
    _print_centered(f'"{result.bench_name}"', total_width)

    # Device info
    if result.device:
        d = result.device
        _print_centered(
            f"{d.gpu_name} | {d.runtime_name} {d.runtime_version} | {d.gpu_memory_gb} GB",
            total_width,
        )
        _print_centered(f"torch {d.torch_version} | python {d.python_version}", total_width)

    n_kernels = len(kernel_ids)
    n_params = len(param_groups)
    summary = f"{n_kernels} KERNEL{'S' if n_kernels > 1 else ''}"
    if n_params > 1:
        summary += f" x {n_params} PARAM SETS"
    _print_centered(summary, total_width)

    # Validation section
    if result.validation:
        _print_divider(total_width, "section")
        if result.validation.all_passed:
            _print_centered(f"{GREEN}VALIDATION: ALL PASSED{RESET}{COLOR}", total_width)
        else:
            _print_centered(f"{RED}{BOLD}VALIDATION: MISMATCH DETECTED{RESET}{COLOR}", total_width)
        _print_row_divider(total_width, label_width, "top")
        for comp in result.validation.comparisons:
            status = f"{GREEN}PASS{RESET}{COLOR}" if comp.passed else f"{RED}FAIL{RESET}{COLOR}"
            pair_label = _truncate(f"{comp.kernel_a} vs", label_width)
            _print_row(pair_label, f"{status}", total_width, label_width)
            _print_row(
                _truncate(comp.kernel_b, label_width),
                f"{DIM}max_abs={comp.max_abs_diff:.2e}  "
                f"max_rel={comp.max_rel_diff:.2e}  "
                f"mismatched={comp.mismatched_elements}/{comp.total_elements}"
                f"{RESET}{COLOR}",
                total_width,
                label_width,
            )
        _print_row_divider(total_width, label_width, "bottom")

    # Compact view for large sweeps — full per-combo blocks become unreadable.
    if len(param_groups) >= SWEEP_SUMMARY_THRESHOLD:
        _print_sweep_summary(result, total_width)
        return

    for group_key, group_results in param_groups.items():
        _print_divider(total_width, "section")
        if group_key:
            _print_centered(f"PARAMS: {group_key}", total_width)
        _print_row_divider(total_width, label_width, "top")

        fastest = min(group_results, key=lambda r: r.median_ms)
        slowest_median = max(r.median_ms for r in group_results)

        for i, kr in enumerate(group_results):
            is_fastest = kr is fastest and len(group_results) > 1
            kernel_label = _truncate(kr.kernel_id, label_width)
            bar = _make_bar(kr.median_ms, slowest_median, bar_width)

            # Timing line with bar
            median_text = f"{kr.median_ms:.3f} ms"
            if is_fastest:
                timing = f"{GREEN}{BOLD}{median_text}{RESET}{COLOR}  {bar}"
            else:
                timing = f"{median_text}  {bar}"
            _print_row(kernel_label, timing, total_width, label_width)

            # Stats line (dimmed): quantiles + IQR are more robust to GPU
            # tail latency than mean/std/min/max. A noisy run gets a warning
            # marker so the reader knows the median is suspect.
            stats = (
                f"p10={kr.p10_ms:.3f}  p50={kr.median_ms:.3f}  p90={kr.p90_ms:.3f}  "
                f"iqr={kr.iqr_ms:.3f}"
            )
            if kr.has_warnings:
                stats += f"{RESET}{COLOR}  {RED}⚠ noisy{RESET}{DIM}"
            _print_row("", f"{DIM}{stats}{RESET}{COLOR}", total_width, label_width)

            # Throughput line (dimmed) — derived from flops/bytes when set.
            throughput_str = _format_throughput(kr)
            if throughput_str:
                _print_row("", f"{DIM}{throughput_str}{RESET}{COLOR}", total_width, label_width)

            # Metrics line (dimmed, only if any metric was collected)
            metrics_str = _format_metrics(kr.metrics)
            if metrics_str:
                _print_row("", f"{DIM}{metrics_str}{RESET}{COLOR}", total_width, label_width)

            # Compile time (dimmed) — first-call cost separated from steady-state.
            if kr.compile_ms is not None:
                _print_row(
                    "",
                    f"{DIM}compile={kr.compile_ms:.3f} ms{RESET}{COLOR}",
                    total_width,
                    label_width,
                )

            # Comparison line
            if is_fastest:
                _print_row("", f"{GREEN}FASTEST{RESET}{COLOR}", total_width, label_width)
            elif len(group_results) > 1:
                _print_row("", _format_comparison(kr, fastest), total_width, label_width)

            if i < len(group_results) - 1:
                _print_row_divider(total_width, label_width)

    _print_row_divider(total_width, label_width, "bottom")
