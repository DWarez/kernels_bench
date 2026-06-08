"""Tests for RemoteRequest serialization and bench reconstruction."""

from pathlib import Path

import pytest

from kernels_bench.remote.request import RemoteRequest

EXAMPLE_BENCH = Path(__file__).resolve().parent.parent / "examples" / "bench_gelu.py"


def test_json_round_trip_quick():
    req = RemoteRequest(
        mode="quick",
        kernels=["org/a", "org/a@dev"],
        fn="gelu_fast",
        args=["y:1024,1024:float16:output", "x:1024,1024:float16:input"],
        sweeps=["M=512,1024"],
        warmup=5,
        iterations=20,
        validate_outputs=True,
        atol=1e-2,
        rtol=1e-2,
        flops=1000,
        bytes_per_iter=2048,
    )
    assert RemoteRequest.from_json(req.to_json()) == req


def test_json_round_trip_run():
    req = RemoteRequest(mode="run", kernels=["org/a"], bench_filename="bench_gelu.py")
    assert RemoteRequest.from_json(req.to_json()) == req


def test_build_bench_quick_splits_roles():
    req = RemoteRequest(
        mode="quick",
        kernels=["org/a"],
        fn="gelu_fast",
        args=["y:1024,512:float16:output", "x:1024,512:float16:input"],
    )
    bench = req.build_bench()
    assert bench.name == "gelu_fast"
    assert [s.name for s in bench.inputs] == ["x"]
    assert [s.name for s in bench.outputs] == ["y"]
    assert bench._fn is not None


def test_build_bench_quick_preserves_cli_arg_order():
    # The kernel fn must be called as gelu_fast(y, x) — output first, as on the
    # CLI — even though Bench groups inputs before outputs internally. The
    # generated forward only reorders positional args, so plain markers suffice.
    req = RemoteRequest(
        mode="quick",
        kernels=["org/a"],
        fn="myfn",
        args=["y:4,4:float16:output", "x:4,4:float16:input"],
    )
    bench = req.build_bench()

    calls = {}

    class FakeKernel:
        def myfn(self, *args):
            calls["order"] = list(args)

    # Bench calls forward(kernel, *inputs, *outputs) == forward(k, <x>, <y>).
    bench._fn(FakeKernel(), "x-tensor", "y-tensor")
    assert calls["order"] == ["y-tensor", "x-tensor"]  # original CLI order restored


def test_build_bench_quick_with_sweep_params():
    req = RemoteRequest(
        mode="quick",
        kernels=["org/a"],
        fn="f",
        args=["x:M,N:float16:input", "y:M,N:float16:output"],
        sweeps=["M=512,1024", "N=64"],
    )
    bench = req.build_bench()
    assert bench.params == {"M": [512, 1024], "N": [64]}


def test_build_bench_run_loads_file():
    req = RemoteRequest(mode="run", kernels=["org/a"], bench_filename=str(EXAMPLE_BENCH))
    bench = req.build_bench()
    assert bench.name == "gelu_activation"
    assert bench.params == {"M": [1024, 2048, 4096], "N": [1024]}


def test_build_bench_run_without_filename_errors():
    req = RemoteRequest(mode="run", kernels=["org/a"])
    with pytest.raises(ValueError, match="bench_filename"):
        req.build_bench()


def test_build_bench_quick_without_fn_errors():
    req = RemoteRequest(mode="quick", kernels=["org/a"], args=["x:4:float16:input"])
    with pytest.raises(ValueError, match="fn"):
        req.build_bench()
