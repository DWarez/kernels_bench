"""Core Bench class — the main user-facing API."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from kernels import get_kernel

from kernels_bench.progress import benchmark_progress, make_on_step
from kernels_bench.runner import BenchResult, KernelResult, _resolve_specs, run_benchmark
from kernels_bench.runtime import Runtime, detect_runtime
from kernels_bench.spec import TensorSpec
from kernels_bench.validate import validate_bench


class Bench:
    """Define and run a benchmark comparing HuggingFace Kernels.

    Example:
        bench = Bench(
            name="gelu",
            inputs=[TensorSpec("x", shape=(1024, 1024), dtype=torch.float16)],
            outputs=[TensorSpec("y", shape=(1024, 1024), dtype=torch.float16)],
        )

        @bench.fn
        def forward(kernel, x, y):
            kernel.gelu_fast(y, x)

        results = bench.run(kernels=["kernels-community/activation"])
    """

    def __init__(
        self,
        name: str,
        inputs: list[TensorSpec],
        outputs: list[TensorSpec],
        params: dict[str, list[int]] | None = None,
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = params or {}
        self._fn: Callable[..., Any] | None = None

        self._validate()

    def _validate(self) -> None:
        """Check that all symbolic dims are covered by params."""
        all_specs = self.inputs + self.outputs
        required_symbols: set[str] = set()
        for spec in all_specs:
            required_symbols |= spec.symbolic_dims

        provided = set(self.params.keys())
        missing = required_symbols - provided
        if missing:
            raise ValueError(
                f"symbolic dimensions {missing} used in specs but not provided in params"
            )

        unused = provided - required_symbols
        if unused:
            raise ValueError(f"params {unused} provided but not used in any spec")

        # Check for duplicate names
        names = [s.name for s in all_specs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tensor names in inputs/outputs")

    def fn(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register the benchmark function.

        The function signature must be: (kernel, *inputs, *outputs)
        where input/output args match the TensorSpec names in order.
        """
        self._fn = func
        return func

    def _param_combinations(self) -> list[dict[str, int]]:
        """Generate all combinations of param values."""
        if not self.params:
            return [{}]
        keys = sorted(self.params.keys())
        values = [self.params[k] for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    def run(
        self,
        kernels: list[str],
        warmup: int = 10,
        iterations: int = 100,
        validate: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        runtime: Runtime | None = None,
    ) -> BenchResult:
        """Run the benchmark for all kernels and param combinations.

        Args:
            kernels: list of HuggingFace kernel repo IDs (e.g. "kernels-community/activation")
            warmup: number of warmup iterations before timing
            iterations: number of timed iterations
            validate: if True, compare outputs across kernels before benchmarking
            atol: absolute tolerance for validation
            rtol: relative tolerance for validation
            runtime: GPU runtime to use (auto-detected if not provided)
        """
        if self._fn is None:
            raise RuntimeError("no benchmark function registered — use @bench.fn")

        if runtime is None:
            runtime = detect_runtime()

        # Load all kernels upfront (needed for validation)
        loaded_kernels: dict[str, Any] = {}
        for kernel_id in kernels:
            try:
                loaded_kernels[kernel_id] = get_kernel(kernel_id)
            except Exception as e:
                raise RuntimeError(f"failed to load kernel {kernel_id!r}: {e}") from e

        # Validation (uses first param combo if there are symbolic dims)
        validation = None
        if validate and len(loaded_kernels) > 1:
            param_combos = self._param_combinations()
            first_params = param_combos[0] if param_combos else {}
            resolved_inputs = _resolve_specs(self.inputs, first_params)
            resolved_outputs = _resolve_specs(self.outputs, first_params)
            validation = validate_bench(
                bench_fn=self._fn,
                kernels=loaded_kernels,
                input_specs=resolved_inputs,
                output_specs=resolved_outputs,
                runtime=runtime,
                atol=atol,
                rtol=rtol,
            )

        param_combos = self._param_combinations()
        all_results: list[KernelResult] = []

        with benchmark_progress() as progress:
            for kernel_id, kernel in loaded_kernels.items():
                for param_set in param_combos:
                    params_str = ", ".join(f"{k}={v}" for k, v in sorted(param_set.items()))
                    label = f"{kernel_id}"
                    if params_str:
                        label += f" ({params_str})"

                    warmup_tid = progress.add_task(f"{label} warmup", total=warmup)
                    bench_tid = progress.add_task(f"{label} bench", total=iterations)
                    on_step = make_on_step(progress, warmup_tid, bench_tid)

                    resolved_inputs = _resolve_specs(self.inputs, param_set)
                    resolved_outputs = _resolve_specs(self.outputs, param_set)

                    times = run_benchmark(
                        bench_fn=self._fn,
                        kernel=kernel,
                        input_specs=resolved_inputs,
                        output_specs=resolved_outputs,
                        warmup=warmup,
                        iterations=iterations,
                        runtime=runtime,
                        on_step=on_step,
                    )

                    all_results.append(
                        KernelResult(
                            kernel_id=kernel_id,
                            params=param_set,
                            times_ms=times,
                        )
                    )

        return BenchResult(
            bench_name=self.name,
            kernel_results=all_results,
            device=runtime.get_device_info(),
            validation=validation,
        )
