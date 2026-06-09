"""Tests for CLI argument parsers and the --remote / hardware surface."""

import click
import pytest
import torch
from click.testing import CliRunner

from kernels_bench import cli
from kernels_bench.cli import _parse_arg, _parse_sweep, main
from kernels_bench.runner import BenchResult, KernelResult


def test_parse_arg_concrete_dims():
    spec = _parse_arg("x:1024,512:float16:input")
    assert spec.name == "x"
    assert spec.shape == (1024, 512)
    assert spec.dtype is torch.float16
    assert spec.role == "input"


def test_parse_arg_symbolic_dims():
    spec = _parse_arg("x:M,N:float16:input")
    assert spec.shape == ("M", "N")
    assert spec.symbolic_dims == {"M", "N"}


def test_parse_arg_mixed_dims():
    spec = _parse_arg("y:M,128:float16:output")
    assert spec.shape == ("M", 128)


def test_parse_sweep_basic():
    key, values = _parse_sweep("M=512,1024,2048")
    assert key == "M"
    assert values == [512, 1024, 2048]


def test_parse_sweep_strips_key():
    key, _ = _parse_sweep(" N =64,128")
    assert key == "N"


def test_parse_sweep_missing_equals():
    with pytest.raises(click.ClickException, match="expected KEY="):
        _parse_sweep("M:512,1024")


def test_parse_sweep_non_int_values():
    with pytest.raises(click.ClickException, match="must be ints"):
        _parse_sweep("M=512,foo")


def test_parse_sweep_invalid_key():
    with pytest.raises(click.ClickException, match="valid identifier"):
        _parse_sweep("1bad=1,2")


# --- remote surface -------------------------------------------------------


def _fake_result() -> BenchResult:
    return BenchResult(
        bench_name="gelu_fast",
        kernel_results=[KernelResult(kernel_id="org/a", params={}, times_ms=[0.1, 0.11])],
    )


def test_hardware_command_lists_flavors():
    result = CliRunner().invoke(main, ["hardware"])
    assert result.exit_code == 0
    assert "h200" in result.output
    assert "a100-large" in result.output


def test_quick_remote_bad_flavor_fails_fast():
    # Must error on the flavor without needing a GPU or launching anything.
    result = CliRunner().invoke(
        main,
        ["quick", "-k", "org/a", "--fn", "f", "--arg", "x:8,8:float16:input", "--remote", "zzz"],
    )
    assert result.exit_code != 0
    assert "unknown GPU flavor" in result.output


def test_quick_remote_routes_to_run_remote(monkeypatch):
    captured = {}

    def fake_run_remote(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return _fake_result()

    # _run_remote_bench does `from kernels_bench.remote import run_remote`, which
    # binds the module attribute at call time — patch it there.
    import kernels_bench.remote as remote_pkg

    monkeypatch.setattr(remote_pkg, "run_remote", fake_run_remote)

    result = CliRunner().invoke(
        main,
        [
            "quick",
            "-k",
            "org/a,org/b",
            "--fn",
            "gelu_fast",
            "--arg",
            "y:8,8:float16:output",
            "--arg",
            "x:8,8:float16:input",
            "--remote",
            "h200",
            "-n",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    req = captured["request"]
    assert req.mode == "quick"
    assert req.kernels == ["org/a", "org/b"]
    assert req.fn == "gelu_fast"
    assert req.iterations == 5
    assert captured["kwargs"]["flavor"] == "h200"


def test_remote_rejects_profile(monkeypatch):
    import kernels_bench.remote as remote_pkg

    monkeypatch.setattr(remote_pkg, "run_remote", lambda *a, **k: _fake_result())
    result = CliRunner().invoke(
        main,
        [
            "quick",
            "-k",
            "org/a",
            "--fn",
            "f",
            "--arg",
            "x:8,8:float16:input",
            "--remote",
            "h200",
            "--profile",
        ],
    )
    assert result.exit_code != 0
    assert "profile" in result.output.lower()


def test_local_quick_unaffected(monkeypatch):
    # Without --remote, the quick path still tries the local runtime (which is
    # absent on CI), proving the remote branch doesn't shadow local behaviour.
    sentinel = RuntimeError("detect_runtime called")

    def boom():
        raise sentinel

    monkeypatch.setattr(cli, "_run_remote_bench", lambda **k: pytest.fail("went remote"))
    # detect_runtime is imported inside quick(); patch the source module.
    import kernels_bench.runtime as rt

    monkeypatch.setattr(rt, "detect_runtime", boom)
    result = CliRunner().invoke(
        main, ["quick", "-k", "org/a", "--fn", "f", "--arg", "x:8,8:float16:input"]
    )
    assert result.exit_code != 0
    assert "detect_runtime called" in str(result.exception) or "detect_runtime" in result.output
