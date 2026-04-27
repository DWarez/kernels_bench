"""Tests for CLI argument parsers."""

import click
import pytest
import torch

from kernels_bench.cli import _parse_arg, _parse_sweep


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
