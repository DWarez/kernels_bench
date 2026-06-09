"""Tests for run_remote — the HF Jobs launcher — with HF calls mocked."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from rich.console import Console

from kernels_bench.remote import launch
from kernels_bench.remote.request import RESULT_BEGIN, RESULT_END, RemoteRequest
from kernels_bench.runner import BenchResult, KernelResult


def _silent_console() -> Console:
    return Console(file=io.StringIO())


def _expected_result() -> BenchResult:
    return BenchResult(
        bench_name="gelu_fast",
        kernel_results=[KernelResult(kernel_id="org/a", params={}, times_ms=[0.1, 0.11, 0.12])],
    )


def _quick_request() -> RemoteRequest:
    return RemoteRequest(
        mode="quick", kernels=["org/a"], fn="gelu_fast", args=["x:8,8:float16:input"]
    )


def test_run_remote_parses_sentinel_result(monkeypatch):
    monkeypatch.delenv("KB_REMOTE_REF", raising=False)
    expected = _expected_result()
    logs = ["starting", "warming up", RESULT_BEGIN, json.dumps(expected.to_dict()), RESULT_END]

    captured = {}

    def fake_run_uv_job(**kwargs):
        captured.update(kwargs)
        # The generated worker exists only during the call; read it now.
        captured["script_text"] = Path(kwargs["script"]).read_text()
        return SimpleNamespace(id="job-123", url="https://hf.co/jobs/job-123")

    monkeypatch.setattr(launch, "run_uv_job", fake_run_uv_job)
    monkeypatch.setattr(launch, "fetch_job_logs", lambda **kw: logs)
    monkeypatch.setattr(launch, "get_token", lambda: "hf_tok")

    result = launch.run_remote(_quick_request(), flavor="h200", console=_silent_console())

    assert result.to_dict() == expected.to_dict()
    # Job was wired correctly.
    assert captured["flavor"] == "h200"
    assert "KB_REQUEST" in captured["env"]
    assert captured["secrets"]["HF_TOKEN"] == "hf_tok"
    # All deps live in the generated worker's PEP-723 header (no --with).
    assert "dependencies" not in captured
    assert "kernels-bench @ git+" in captured["script_text"]
    assert "pytorch-cu126" in captured["script_text"]


def test_run_remote_respects_remote_ref_env(monkeypatch):
    monkeypatch.setenv("KB_REMOTE_REF", "feat/remote-hf-jobs")
    captured = {}

    def fake_run_uv_job(**kwargs):
        captured.update(kwargs)
        captured["script_text"] = Path(kwargs["script"]).read_text()
        return SimpleNamespace(id="j", url=None)

    expected = _expected_result()
    monkeypatch.setattr(launch, "run_uv_job", fake_run_uv_job)
    monkeypatch.setattr(
        launch,
        "fetch_job_logs",
        lambda **kw: [RESULT_BEGIN, json.dumps(expected.to_dict()), RESULT_END],
    )
    monkeypatch.setattr(launch, "get_token", lambda: None)

    launch.run_remote(_quick_request(), flavor="t4-small", console=_silent_console())
    assert "@feat/remote-hf-jobs" in captured["script_text"]
    # No token -> no secrets injected.
    assert captured["secrets"] is None


def test_run_remote_run_mode_uploads_bench_file(monkeypatch, tmp_path):
    monkeypatch.delenv("KB_REMOTE_REF", raising=False)
    bench_file = tmp_path / "mybench.py"
    bench_file.write_text("# bench")
    captured = {}

    def fake_run_uv_job(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="j", url=None)

    expected = _expected_result()
    monkeypatch.setattr(launch, "run_uv_job", fake_run_uv_job)
    monkeypatch.setattr(
        launch,
        "fetch_job_logs",
        lambda **kw: [RESULT_BEGIN, json.dumps(expected.to_dict()), RESULT_END],
    )
    monkeypatch.setattr(launch, "get_token", lambda: "tok")

    req = RemoteRequest(mode="run", kernels=["org/a"])
    launch.run_remote(req, flavor="h200", bench_file=str(bench_file), console=_silent_console())

    # The bench file is passed as a script arg (HF uploads it) and its basename
    # recorded in the request env.
    assert str(bench_file) in captured["script_args"]
    sent = RemoteRequest.from_json(captured["env"]["KB_REQUEST"])
    assert sent.bench_filename == "mybench.py"


def test_run_remote_run_mode_requires_bench_file(monkeypatch):
    monkeypatch.setattr(launch, "get_token", lambda: "tok")
    req = RemoteRequest(mode="run", kernels=["org/a"])
    with pytest.raises(click.ClickException, match="bench file"):
        launch.run_remote(req, flavor="h200", console=_silent_console())


def test_run_remote_no_result_raises_with_status(monkeypatch):
    monkeypatch.setattr(launch, "run_uv_job", lambda **k: SimpleNamespace(id="job-x", url=None))
    monkeypatch.setattr(launch, "fetch_job_logs", lambda **k: ["boom", "no sentinel here"])
    monkeypatch.setattr(launch, "get_token", lambda: None)
    monkeypatch.setattr(
        launch,
        "inspect_job",
        lambda **k: SimpleNamespace(status=SimpleNamespace(stage="ERROR", message="OOM")),
    )

    with pytest.raises(click.ClickException, match="no result"):
        launch.run_remote(_quick_request(), flavor="h200", console=_silent_console())
