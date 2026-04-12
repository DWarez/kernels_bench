"""Tests for TensorSpec."""

import pytest
import torch

from kernels_bench.spec import TensorSpec


def test_basic_creation():
    spec = TensorSpec("x", shape=(1024, 1024), dtype=torch.float16)
    assert spec.name == "x"
    assert spec.shape == (1024, 1024)
    assert spec.dtype == torch.float16
    assert spec.device == "cuda"
    assert spec.role == "input"


def test_output_role():
    spec = TensorSpec("y", shape=(512,), dtype=torch.float32, role="output")
    assert spec.role == "output"


def test_symbolic_dims():
    spec = TensorSpec("x", shape=("M", "N"), dtype=torch.float16)
    assert spec.symbolic_dims == {"M", "N"}


def test_no_symbolic_dims():
    spec = TensorSpec("x", shape=(1024, 1024), dtype=torch.float16)
    assert spec.symbolic_dims == set()


def test_resolve():
    spec = TensorSpec("x", shape=("M", "N"), dtype=torch.float16)
    resolved = spec.resolve({"M": 512, "N": 1024})
    assert resolved.shape == (512, 1024)
    assert resolved.name == "x"
    assert resolved.dtype == torch.float16
    assert resolved.role == "input"


def test_resolve_preserves_role():
    spec = TensorSpec("y", shape=("M",), dtype=torch.float16, role="output")
    resolved = spec.resolve({"M": 256})
    assert resolved.role == "output"


def test_resolve_missing_param():
    spec = TensorSpec("x", shape=("M", "N"), dtype=torch.float16)
    with pytest.raises(ValueError, match="not found in params"):
        spec.resolve({"M": 512})


def test_resolve_mixed_dims():
    spec = TensorSpec("x", shape=(1024, "N"), dtype=torch.float16)
    resolved = spec.resolve({"N": 512})
    assert resolved.shape == (1024, 512)


def test_allocate_input():
    spec = TensorSpec("x", shape=(32, 32), dtype=torch.float16)
    t = spec.allocate_input()
    assert t.shape == (32, 32)
    assert t.dtype == torch.float16
    assert t.device.type == "cuda"


def test_allocate_output():
    spec = TensorSpec("y", shape=(32, 32), dtype=torch.float16, role="output")
    t = spec.allocate_output()
    assert t.shape == (32, 32)
    assert t.dtype == torch.float16


def test_allocate_by_role():
    input_spec = TensorSpec("x", shape=(16, 16), dtype=torch.float16, role="input")
    output_spec = TensorSpec("y", shape=(16, 16), dtype=torch.float16, role="output")
    ti = input_spec.allocate()
    to = output_spec.allocate()
    assert ti.shape == to.shape


def test_allocate_unresolved_raises():
    spec = TensorSpec("x", shape=("M", 1024), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="unresolved symbolic dims"):
        spec.allocate_input()


def test_empty_shape_raises():
    with pytest.raises(ValueError, match="at least one dimension"):
        TensorSpec("x", shape=(), dtype=torch.float16)


def test_negative_dim_raises():
    with pytest.raises(ValueError, match="positive"):
        TensorSpec("x", shape=(-1, 1024), dtype=torch.float16)


def test_invalid_role_raises():
    with pytest.raises(ValueError, match="role"):
        TensorSpec("x", shape=(1024,), dtype=torch.float16, role="bad")
