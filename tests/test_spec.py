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


@pytest.mark.gpu
def test_allocate_input(device):
    spec = TensorSpec("x", shape=(32, 32), dtype=torch.float16, device=device)
    t = spec.allocate_input()
    assert t.shape == (32, 32)
    assert t.dtype == torch.float16
    assert t.device.type == device


@pytest.mark.gpu
def test_allocate_output(device):
    spec = TensorSpec("y", shape=(32, 32), dtype=torch.float16, device=device, role="output")
    t = spec.allocate_output()
    assert t.shape == (32, 32)
    assert t.dtype == torch.float16


@pytest.mark.gpu
def test_allocate_by_role(device):
    input_spec = TensorSpec("x", shape=(16, 16), dtype=torch.float16, device=device, role="input")
    output_spec = TensorSpec("y", shape=(16, 16), dtype=torch.float16, device=device, role="output")
    ti = input_spec.allocate()
    to = output_spec.allocate()
    assert ti.shape == to.shape


@pytest.mark.gpu
def test_allocate_input_with_device_override(device):
    """Passing an explicit device to allocate_input() overrides spec.device."""
    spec = TensorSpec("x", shape=(8, 8), dtype=torch.float16, device="this_does_not_exist")
    t = spec.allocate_input(device)
    assert t.device.type == device
    assert t.shape == (8, 8)


@pytest.mark.gpu
def test_allocate_output_with_device_override(device):
    """Passing an explicit device to allocate_output() overrides spec.device."""
    spec = TensorSpec(
        "y", shape=(8, 8), dtype=torch.float16, device="this_does_not_exist", role="output"
    )
    t = spec.allocate_output(device)
    assert t.device.type == device
    assert t.shape == (8, 8)


@pytest.mark.gpu
def test_allocate_with_device_override(device):
    """allocate() forwards the device override based on role."""
    fake = "this_does_not_exist"
    input_spec = TensorSpec("x", shape=(8, 8), dtype=torch.float16, device=fake)
    output_spec = TensorSpec("y", shape=(8, 8), dtype=torch.float16, device=fake, role="output")
    ti = input_spec.allocate(device)
    to = output_spec.allocate(device)
    assert ti.device.type == device
    assert to.device.type == device


@pytest.mark.gpu
def test_allocate_input_float8_fallback(device):
    """float8 dtypes that don't support randn should still allocate via cast."""
    spec = TensorSpec("x", shape=(8, 8), dtype=torch.float8_e4m3fn, device=device)
    t = spec.allocate_input()
    assert t.dtype == torch.float8_e4m3fn
    assert t.shape == (8, 8)


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


def test_nbytes_concrete():
    spec = TensorSpec("x", shape=(1024, 1024), dtype=torch.float16)
    assert spec.nbytes == 1024 * 1024 * 2


def test_nbytes_symbolic_raises():
    spec = TensorSpec("x", shape=("M", 1024), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="symbolic dims"):
        _ = spec.nbytes
