"""CLI entrypoint for kernels-bench."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import click
import torch

from kernels_bench.bench import Bench
from kernels_bench.display import print_results
from kernels_bench.runner import BenchResult
from kernels_bench.spec import TensorSpec

DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _parse_arg(arg: str) -> TensorSpec:
    """Parse a CLI arg spec string into a TensorSpec.

    Format: name:dim1,dim2,...:dtype:role
    Examples:
        x:1024,1024:float16:input
        y:1024,1024:float16:output
        x:1024,1024:float16        (role defaults to input)
        x:1024,1024                 (dtype defaults to float16, role to input)
    """
    parts = arg.split(":")
    if len(parts) < 2:
        raise click.ClickException(
            f"invalid arg format {arg!r}, expected name:shape[:dtype[:role]]"
        )

    name = parts[0]
    shape = tuple(int(d) for d in parts[1].split(","))
    dtype = DTYPE_MAP.get(parts[2], torch.float16) if len(parts) > 2 else torch.float16
    role = parts[3] if len(parts) > 3 else "input"

    if len(parts) > 2 and parts[2] not in DTYPE_MAP:
        raise click.ClickException(
            f"unknown dtype {parts[2]!r}, valid options: {', '.join(DTYPE_MAP)}"
        )
    if role not in ("input", "output"):
        raise click.ClickException(f"role must be 'input' or 'output', got {role!r}")

    return TensorSpec(name, shape=shape, dtype=dtype, role=role)


def _load_bench_from_file(path: str) -> Bench:
    """Import a Python file and find the Bench instance in it."""
    filepath = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_bench_module", filepath)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"cannot load {path} as a Python module")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_module"] = module
    spec.loader.exec_module(module)

    benches = [v for v in vars(module).values() if isinstance(v, Bench)]
    if not benches:
        raise click.ClickException(f"no Bench instance found in {path}")
    if len(benches) > 1:
        raise click.ClickException(
            f"multiple Bench instances found in {path}, expected exactly one"
        )
    return benches[0]


def _handle_output(result: BenchResult, output: str | None) -> None:
    """Print results to terminal and optionally write JSON to file."""
    print_results(result)
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"\nResults saved to {output}")


@click.group()
@click.version_option()
def main() -> None:
    """Benchmark tool for HuggingFace Kernels."""


@main.command()
@click.argument("bench_file", type=click.Path(exists=True))
@click.option(
    "--kernels",
    "-k",
    required=True,
    help="Comma-separated list of HuggingFace kernel repo IDs.",
)
@click.option("--warmup", "-w", default=10, show_default=True, help="Number of warmup iterations.")
@click.option(
    "--iterations", "-n", default=100, show_default=True, help="Number of timed iterations."
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Write results to a JSON file.",
)
@click.option("--validate", is_flag=True, help="Validate output correctness across kernels.")
@click.option("--atol", default=1e-3, show_default=True, help="Absolute tolerance for validation.")
@click.option("--rtol", default=1e-3, show_default=True, help="Relative tolerance for validation.")
@click.option(
    "--no-metrics",
    "no_metrics",
    is_flag=True,
    help="Skip collecting peak memory and GPU utilization metrics.",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Run torch.profiler after timing and print the per-op breakdown.",
)
def run(
    bench_file: str,
    kernels: str,
    warmup: int,
    iterations: int,
    output: str | None,
    validate: bool,
    atol: float,
    rtol: float,
    no_metrics: bool,
    profile: bool,
) -> None:
    """Run a benchmark defined in BENCH_FILE against the specified kernels."""
    bench = _load_bench_from_file(bench_file)
    kernel_list = [k.strip() for k in kernels.split(",")]

    result = bench.run(
        kernels=kernel_list,
        warmup=warmup,
        iterations=iterations,
        validate=validate,
        atol=atol,
        rtol=rtol,
        collect_metrics=not no_metrics,
        profile=profile,
    )
    _handle_output(result, output)


@main.command()
@click.option(
    "--kernels",
    "-k",
    required=True,
    help="Comma-separated list of HuggingFace kernel repo IDs.",
)
@click.option(
    "--fn",
    "-f",
    required=True,
    help="Name of the kernel function to benchmark.",
)
@click.option(
    "--arg",
    "-a",
    "args",
    multiple=True,
    required=True,
    help=(
        "Tensor arg in order: name:shape:dtype:role. "
        "e.g. --arg y:1024,1024:float16:output --arg x:1024,1024:float16:input. "
        "dtype defaults to float16, role defaults to input."
    ),
)
@click.option("--warmup", "-w", default=10, show_default=True, help="Number of warmup iterations.")
@click.option(
    "--iterations", "-n", default=100, show_default=True, help="Number of timed iterations."
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Write results to a JSON file.",
)
@click.option("--validate", is_flag=True, help="Validate output correctness across kernels.")
@click.option("--atol", default=1e-3, show_default=True, help="Absolute tolerance for validation.")
@click.option("--rtol", default=1e-3, show_default=True, help="Relative tolerance for validation.")
@click.option(
    "--no-metrics",
    "no_metrics",
    is_flag=True,
    help="Skip collecting peak memory and GPU utilization metrics.",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Run torch.profiler after timing and print the per-op breakdown.",
)
@click.option(
    "--flops",
    type=int,
    default=None,
    help="Number of FLOPs per kernel call, used to compute TFLOP/s.",
)
@click.option(
    "--bytes",
    "bytes_per_iter",
    type=int,
    default=None,
    help="Bytes moved per kernel call. Defaults to the sum of input+output tensor sizes.",
)
def quick(
    kernels: str,
    fn: str,
    args: tuple[str, ...],
    warmup: int,
    iterations: int,
    output: str | None,
    validate: bool,
    atol: float,
    rtol: float,
    no_metrics: bool,
    profile: bool,
    flops: int | None,
    bytes_per_iter: int | None,
) -> None:
    """Benchmark a kernel function directly — no bench file needed.

    Pass tensor arguments in the order the kernel function expects them.

    Example:

        kernels-bench quick -k kernels-community/activation
        --fn gelu_fast --arg y:1024,1024:float16:output --arg x:1024,1024:float16:input
    """
    from kernels import get_kernel

    from kernels_bench.bench import auto_bytes
    from kernels_bench.progress import benchmark_progress, make_on_step
    from kernels_bench.runner import KernelResult, run_benchmark_quick
    from kernels_bench.runtime import detect_runtime
    from kernels_bench.validate import validate_quick

    specs = [_parse_arg(a) for a in args]
    if bytes_per_iter is None:
        bytes_per_iter = auto_bytes(specs)
    kernel_list = [k.strip() for k in kernels.split(",")]
    runtime = detect_runtime()

    # Load all kernels upfront
    loaded_kernels: dict[str, object] = {}
    for kernel_id in kernel_list:
        try:
            loaded_kernels[kernel_id] = get_kernel(kernel_id)
        except Exception as e:
            raise click.ClickException(f"failed to load kernel {kernel_id!r}: {e}") from e

    # Validation
    validation = None
    if validate and len(loaded_kernels) > 1:
        validation = validate_quick(
            kernels=loaded_kernels,
            fn_name=fn,
            specs=specs,
            runtime=runtime,
            atol=atol,
            rtol=rtol,
        )

    all_results: list[KernelResult] = []
    with benchmark_progress() as progress:
        for kernel_id, kernel in loaded_kernels.items():
            warmup_tid = progress.add_task(f"{kernel_id} warmup", total=warmup)
            bench_tid = progress.add_task(f"{kernel_id} bench", total=iterations)
            on_step = make_on_step(progress, warmup_tid, bench_tid)
            times, metrics, compile_ms = run_benchmark_quick(
                kernel=kernel,
                fn_name=fn,
                specs=specs,
                warmup=warmup,
                iterations=iterations,
                runtime=runtime,
                on_step=on_step,
                collect_metrics=not no_metrics,
                profile=profile,
                profile_label=kernel_id,
            )
            all_results.append(
                KernelResult(
                    kernel_id=kernel_id,
                    params={},
                    times_ms=times,
                    metrics=metrics,
                    compile_ms=compile_ms,
                    flops=flops,
                    bytes_per_iter=bytes_per_iter,
                )
            )

    result = BenchResult(
        bench_name=fn,
        kernel_results=all_results,
        device=runtime.get_device_info(),
        validation=validation,
    )
    _handle_output(result, output)


@main.command("list")
@click.argument("kernel_id")
def list_functions(kernel_id: str) -> None:
    """List available functions in a HuggingFace kernel.

    Example:

        kernels-bench list kernels-community/activation
    """
    import logging
    import warnings

    from kernels import get_kernel

    # Suppress HF download progress bar noise
    logging.disable(logging.INFO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Redirect stderr to suppress tqdm progress bars from HF hub
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        try:
            kernel = get_kernel(kernel_id)
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
            logging.disable(logging.NOTSET)

    from rich.console import Console
    from rich.table import Table

    console = Console()

    functions = sorted(
        name
        for name in dir(kernel)
        if not name.startswith("_") and callable(getattr(kernel, name, None))
    )

    if not functions:
        console.print("[dim]No public functions found.[/dim]")
        return

    min_width = max(len(kernel_id) + 10, 40)
    table = Table(title=f"Functions in {kernel_id}", min_width=min_width)
    table.add_column("Function", style="cyan", no_wrap=True)
    for fn_name in functions:
        table.add_row(fn_name)
    console.print(table)
    console.print(f"\n[dim]Use with:[/dim] kernels-bench quick -k {kernel_id} --fn <function>")
