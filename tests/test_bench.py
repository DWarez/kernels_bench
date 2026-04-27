"""Tests for the Bench class."""

import pytest
import torch

from kernels_bench.bench import Bench, _resolve_workload, auto_bytes
from kernels_bench.spec import TensorSpec


def test_bench_validates_symbolic_dims():
    with pytest.raises(ValueError, match="not provided in params"):
        Bench(
            name="bad",
            inputs=[TensorSpec("x", shape=("M", "N"), dtype=torch.float16)],
            outputs=[],
            params={"M": [1024]},  # missing N
        )


def test_bench_validates_unused_params():
    with pytest.raises(ValueError, match="not used in any spec"):
        Bench(
            name="bad",
            inputs=[TensorSpec("x", shape=(1024,), dtype=torch.float16)],
            outputs=[],
            params={"M": [1024]},  # M not used
        )


def test_bench_validates_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        Bench(
            name="bad",
            inputs=[TensorSpec("x", shape=(1024,), dtype=torch.float16)],
            outputs=[TensorSpec("x", shape=(1024,), dtype=torch.float16, role="output")],
        )


def test_bench_param_combinations():
    bench = Bench(
        name="test",
        inputs=[TensorSpec("x", shape=("M", "N"), dtype=torch.float16)],
        outputs=[],
        params={"M": [512, 1024], "N": [256, 512]},
    )
    combos = bench._param_combinations()
    assert len(combos) == 4
    assert {"M": 512, "N": 256} in combos
    assert {"M": 1024, "N": 512} in combos


def test_bench_no_params():
    bench = Bench(
        name="test",
        inputs=[TensorSpec("x", shape=(1024,), dtype=torch.float16)],
        outputs=[],
    )
    combos = bench._param_combinations()
    assert combos == [{}]


def test_bench_fn_decorator():
    bench = Bench(
        name="test",
        inputs=[TensorSpec("x", shape=(1024,), dtype=torch.float16)],
        outputs=[],
    )

    @bench.fn
    def forward(kernel, x):
        pass

    assert bench._fn is forward


def test_bench_run_without_fn_raises():
    bench = Bench(
        name="test",
        inputs=[TensorSpec("x", shape=(1024,), dtype=torch.float16)],
        outputs=[],
    )
    with pytest.raises(RuntimeError, match="no benchmark function registered"):
        bench.run(kernels=["fake/kernel"])


def test_resolve_workload_static_int():
    assert _resolve_workload(42, {"M": 1024}) == 42


def test_resolve_workload_callable():
    flops = lambda p: 2 * p["M"] * p["N"]  # noqa: E731
    assert _resolve_workload(flops, {"M": 16, "N": 32}) == 1024


def test_resolve_workload_none():
    assert _resolve_workload(None, {}) is None


def test_bench_stores_workload_callables():
    bench = Bench(
        name="t",
        inputs=[TensorSpec("x", shape=("M",), dtype=torch.float16)],
        outputs=[],
        params={"M": [128, 256]},
        flops=lambda p: 2 * p["M"],
        bytes_per_iter=lambda p: 2 * p["M"],
    )
    assert _resolve_workload(bench.flops, {"M": 128}) == 256
    assert _resolve_workload(bench.bytes_per_iter, {"M": 256}) == 512


def test_auto_bytes_sums_specs():
    specs = [
        TensorSpec("x", shape=(1024,), dtype=torch.float16),  # 2048 B
        TensorSpec("y", shape=(1024,), dtype=torch.float32, role="output"),  # 4096 B
    ]
    assert auto_bytes(specs) == 2048 + 4096


def test_auto_bytes_empty():
    assert auto_bytes([]) == 0
