"""Serializable description of a remote benchmark request.

A ``RemoteRequest`` travels from the local CLI to the HF Jobs worker as a JSON
env var. ``build_bench`` reconstructs a :class:`Bench` for both ``quick`` and
``run`` modes so the worker has one execution path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from kernels_bench.bench import Bench

# Sentinels delimiting the result JSON on the worker's stdout. The launcher
# scans the streamed logs for this block; everything else on the stream is
# progress/diagnostic noise routed to stderr.
RESULT_BEGIN = "<<<KB_RESULT_BEGIN>>>"
RESULT_END = "<<<KB_RESULT_END>>>"


class RemoteRequest(BaseModel):
    """Everything the remote worker needs to run a benchmark.

    Fields are mode-specific: ``quick`` carries the raw ``--arg``/``--sweep``
    strings and the function name; ``run`` carries the uploaded bench file's
    remote name. Shared timing/validation knobs apply to both.
    """

    mode: Literal["quick", "run"]
    kernels: list[str]
    warmup: int = 10
    iterations: int = 100
    # Named ``validate_outputs`` rather than ``validate`` to avoid shadowing
    # pydantic ``BaseModel.validate`` (which emits a warning at class creation).
    validate_outputs: bool = False
    atol: float = 1e-3
    rtol: float = 1e-3
    collect_metrics: bool = True

    # quick-only
    fn: str | None = None
    args: list[str] = []
    flops: int | None = None
    bytes_per_iter: int | None = None
    sweeps: list[str] = []

    # run-only: the uploaded bench file's name on the remote side
    bench_filename: str | None = None

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> RemoteRequest:
        return cls.model_validate_json(raw)

    def build_bench(self) -> Bench:
        """Reconstruct a runnable :class:`Bench` from this request.

        ``run`` mode loads the uploaded bench file with the same loader the CLI
        uses. ``quick`` mode synthesizes a bench whose function calls the named
        kernel method with the tensor args in the *original CLI order* (which
        matters — e.g. ``gelu_fast(y, x)`` takes the output first).
        """
        # Deferred: these only execute remotely, inside the worker.
        from kernels_bench.bench import Bench
        from kernels_bench.cli import _load_bench_from_file, _parse_arg, _parse_sweep

        if self.mode == "run":
            if not self.bench_filename:
                raise ValueError("run mode requires bench_filename")
            return _load_bench_from_file(self.bench_filename)

        if not self.fn:
            raise ValueError("quick mode requires fn")

        specs = [_parse_arg(a) for a in self.args]
        input_specs = [s for s in specs if s.role == "input"]
        output_specs = [s for s in specs if s.role == "output"]
        params = dict(_parse_sweep(s) for s in self.sweeps)

        bench = Bench(
            name=self.fn,
            inputs=input_specs,
            outputs=output_specs,
            params=params or None,
            flops=self.flops,
            bytes_per_iter=self.bytes_per_iter,
        )

        # Bench passes (kernel, *inputs, *outputs); rebuild the original arg
        # order from the tensor names so the kernel call matches the CLI.
        grouped_names = [s.name for s in input_specs] + [s.name for s in output_specs]
        original_order = [s.name for s in specs]
        fn_name = self.fn

        @bench.fn
        def forward(kernel, *tensors):
            named = dict(zip(grouped_names, tensors, strict=True))
            return getattr(kernel, fn_name)(*(named[n] for n in original_order))

        return bench
