"""Launch a benchmark on HuggingFace Jobs and bring the results home.

``run_remote`` packages a :class:`RemoteRequest`, starts a uv Job on the chosen
GPU flavor, streams its logs (echoed dimmed under a spinner), and parses the
sentinel-wrapped result JSON back into a :class:`BenchResult` — which the CLI
then renders exactly as a local run.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import click
from huggingface_hub import (
    JobStage,
    fetch_job_logs,
    get_token,
    inspect_job,
    run_uv_job,
)
from rich.console import Console

from kernels_bench.remote import torch_select
from kernels_bench.remote.request import RESULT_BEGIN, RESULT_END, RemoteRequest
from kernels_bench.runner import BenchResult

# Public git source the remote job installs kernels-bench from. The revision is
# overridable via KB_REMOTE_REF so an unmerged dev branch can be benchmarked
# before it lands (remote runs the *pushed* code, never local working changes).
_GIT_URL = "https://github.com/dwarez/kernels_bench.git"
_WORKER = Path(__file__).parent / "_worker.py"


# Fallback torch CUDA build, used only when no kernel in the run publishes CUDA
# builds for the resolver to key off (see torch_select). cu128 runs on every
# current HF Jobs GPU driver.
_FALLBACK_TORCH_CUDA = "cu128"


def _dependency_spec() -> str:
    ref = os.environ.get("KB_REMOTE_REF")
    base = f"kernels-bench @ git+{_GIT_URL}"
    return f"{base}@{ref}" if ref else base


def _resolve_torch(kernels: list[str], console: Console) -> tuple[str, str]:
    """Decide ``(torch_spec, cuda_tag)`` for the worker.

    The build is *derived* from the kernels' published variants
    (:func:`torch_select.resolve`). ``KB_REMOTE_TORCH`` / ``KB_REMOTE_TORCH_CUDA``
    are overrides only: each, when set, wins over the resolved value (and a forced
    CUDA constrains the resolution). If no kernel publishes CUDA builds, fall back
    to the latest torch at cu128.
    """
    env_torch = os.environ.get("KB_REMOTE_TORCH")
    env_cuda = os.environ.get("KB_REMOTE_TORCH_CUDA")

    if env_torch and env_cuda:
        torch_spec, cuda = env_torch, env_cuda  # fully manual — no Hub lookup
    else:
        forced = torch_select.cuda_from_tag(env_cuda) if env_cuda else None
        try:
            resolved = torch_select.resolve(kernels, forced_cuda=forced)
        except torch_select.NoCompatibleTorch as e:
            raise click.ClickException(str(e)) from e
        base_torch, base_cuda = resolved or ("torch", _FALLBACK_TORCH_CUDA)
        torch_spec = env_torch or base_torch
        cuda = env_cuda or base_cuda

    console.print(f"[dim]torch build: {torch_spec} ({cuda})[/dim]")
    return torch_spec, cuda


def _build_worker_script(torch_spec: str, cuda: str) -> tuple[str, str]:
    """Write the worker with a PEP-723 header pinning all deps; return (path, tmpdir).

    torch and kernels-bench must resolve in one uv pass. Passing kernels-bench via
    ``uv run --with`` triggers a second resolution that ignores ``[tool.uv.sources]``
    and re-pulls the default-index torch; inlining both (no ``--with``) keeps torch
    on the chosen CUDA index.
    """
    index = f"pytorch-{cuda}"
    header = f'''\
# /// script
# requires-python = ">=3.12"
# dependencies = ["{torch_spec}", "{_dependency_spec()}"]
#
# [[tool.uv.index]]
# name = "{index}"
# url = "https://download.pytorch.org/whl/{cuda}"
# explicit = true
#
# [tool.uv.sources]
# torch = {{ index = "{index}" }}
# ///
'''
    tmpdir = tempfile.mkdtemp(prefix="kb_remote_")
    path = Path(tmpdir) / "_worker.py"
    path.write_text(header + _WORKER.read_text())
    return str(path), tmpdir


def _extract_result(logs: str) -> dict | None:
    """Pull the JSON payload between the result sentinels, if present."""
    start = logs.find(RESULT_BEGIN)
    end = logs.find(RESULT_END, start + 1)
    if start == -1 or end == -1:
        return None
    payload = logs[start + len(RESULT_BEGIN) : end].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def run_remote(
    request: RemoteRequest,
    *,
    flavor: str,
    bench_file: str | None = None,
    timeout: str | int = "30m",
    token: str | None = None,
    image: str | None = None,
    namespace: str | None = None,
    console: Console | None = None,
) -> BenchResult:
    """Run ``request`` on an HF Jobs GPU of ``flavor`` and return its results.

    Args:
        request: the serialized benchmark to run remotely.
        flavor: a validated HF GPU flavor (see :mod:`.flavors`).
        bench_file: local path to the bench file for ``run`` mode; it is
            uploaded alongside the worker and its basename is recorded in the
            request so the worker can find it.
        timeout: max job duration (HF format, e.g. ``"30m"``) — caps cost.
        token: HF token; falls back to the locally stored login token.
        image: optional custom Docker image (e.g. torch-preinstalled to cut
            cold start); defaults to HF's uv base image.
        namespace: account the job runs under (defaults to your personal
            namespace). Set to an org you have write access to — e.g. when the
            org, not your personal account, carries the Jobs quota/credits.
        console: console for status output (defaults to stdout).
    """
    console = console or Console()
    token = token or get_token()

    script_args: list[str] = []
    if request.mode == "run":
        if not bench_file:
            raise click.ClickException("run mode requires a bench file to upload")
        request = request.model_copy(update={"bench_filename": Path(bench_file).name})
        script_args = [bench_file]

    secrets = {"HF_TOKEN": token} if token else None
    job_env = {"KB_REQUEST": request.to_json()}

    # Deps live in the worker's PEP-723 header (single resolution); no --with.
    # torch is derived from the kernels' published build matrix (see _resolve_torch).
    torch_spec, cuda = _resolve_torch(request.kernels, console)
    worker_path, worker_tmpdir = _build_worker_script(torch_spec, cuda)

    run_kwargs = dict(
        script=worker_path,
        script_args=script_args,
        env=job_env,
        secrets=secrets,
        flavor=flavor,
        timeout=timeout,
        token=token,
    )
    if image:
        run_kwargs["image"] = image
    if namespace:
        run_kwargs["namespace"] = namespace

    console.print(f"[dim]Launching job on [bold]{flavor}[/bold]…[/dim]")
    try:
        job = run_uv_job(**run_kwargs)
    except Exception as e:
        raise click.ClickException(f"failed to launch HF job: {e}") from e
    finally:
        # run_uv_job has already read+uploaded the file content by now.
        shutil.rmtree(worker_tmpdir, ignore_errors=True)

    if getattr(job, "url", None):
        console.print(f"[dim]Job {job.id} — {job.url}[/dim]")

    collected: list[str] = []
    with console.status(f"[bold]Running on {flavor}…", spinner="dots"):
        for line in fetch_job_logs(job_id=job.id, follow=True, namespace=namespace, token=token):
            collected.append(line)
            console.print(f"[dim]{line.rstrip()}[/dim]")

    logs = "\n".join(collected)
    payload = _extract_result(logs)
    if payload is None:
        stage = _final_stage(job.id, token, namespace)
        tail = "\n".join(collected[-25:])
        raise click.ClickException(
            f"remote job produced no result (final status: {stage}).\n"
            f"--- last log lines ---\n{tail}"
        )

    return BenchResult.from_dict(payload)


def _final_stage(job_id: str, token: str | None, namespace: str | None = None) -> str:
    """Best-effort terminal-stage lookup for error reporting."""
    try:
        status = inspect_job(job_id=job_id, namespace=namespace, token=token).status
        stage = status.stage
        if isinstance(stage, JobStage):
            stage = stage.value
        msg = getattr(status, "message", None)
        return f"{stage} ({msg})" if msg else str(stage)
    except Exception:
        return "unknown"
