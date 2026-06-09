# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
#
# # Pin torch to a CUDA-12.6 wheel. HF Jobs instances have heterogeneous NVIDIA
# # drivers; the default PyPI torch is now a cu130 build that fails on the common
# # CUDA-12.9 driver ("driver too old"). cu126 runs on every HF driver (12.6+ and
# # 13.x via backward compat). `uv run` ignores --torch-backend, so we pin the
# # index here instead — uv reads this script metadata even with --with deps.
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# ///
"""Remote worker — runs inside an HF Job on the target GPU.

This script is uploaded and executed by ``launch.run_remote`` via ``uv run``,
with ``kernels-bench`` (and its torch/kernels deps) injected as dependencies.
It rebuilds the benchmark from the ``KB_REQUEST`` env var, runs the *unchanged*
engine on the job's GPU, and writes the result as JSON between sentinels on
**stdout**. Everything else — progress bars, kernel-download chatter — is
routed to **stderr** so the launcher can parse stdout cleanly.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

from rich.console import Console

from kernels_bench.remote.request import RESULT_BEGIN, RESULT_END, RemoteRequest


def main() -> int:
    raw = os.environ.get("KB_REQUEST")
    if not raw:
        print("KB_REQUEST env var is missing", file=sys.stderr)
        return 1

    request = RemoteRequest.from_json(raw)
    bench = request.build_bench()

    # Send all engine output (progress + any stray prints) to stderr so stdout
    # carries only the sentinel-wrapped result.
    err_console = Console(file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        result = bench.run(
            kernels=request.kernels,
            warmup=request.warmup,
            iterations=request.iterations,
            validate=request.validate_outputs,
            atol=request.atol,
            rtol=request.rtol,
            collect_metrics=request.collect_metrics,
            console=err_console,
        )

    sys.stdout.write(RESULT_BEGIN + "\n")
    sys.stdout.write(json.dumps(result.to_dict()))
    sys.stdout.write("\n" + RESULT_END + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
