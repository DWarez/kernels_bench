"""Tests for GPU flavor validation and listing."""

import click
import pytest
from huggingface_hub import SpaceHardware

from kernels_bench.remote.flavors import GPU_FLAVORS, gpu_flavors_grouped, validate_flavor


def test_known_flavors_pass_through():
    for flavor in ["h200", "a100-large", "t4-small", "l40sx1", "zero-a10g"]:
        assert validate_flavor(flavor) == flavor


def test_all_listed_flavors_validate():
    for flavor in GPU_FLAVORS:
        assert validate_flavor(flavor) == flavor


def test_flavors_are_real_space_hardware():
    valid_values = {hw.value for hw in SpaceHardware}
    assert GPU_FLAVORS  # non-empty
    assert set(GPU_FLAVORS) <= valid_values


def test_non_gpu_flavors_excluded():
    # CPU / accelerator flavors must not appear in the GPU list.
    assert "cpu-basic" not in GPU_FLAVORS
    assert "inf2x6" not in GPU_FLAVORS


def test_unknown_flavor_rejected():
    with pytest.raises(click.ClickException, match="unknown GPU flavor"):
        validate_flavor("zzz")


def test_bare_a100_is_not_a_flavor():
    # 'a100' alone isn't a real flavor — 'a100-large' is. No silent aliasing.
    with pytest.raises(click.ClickException):
        validate_flavor("a100")


def test_h100_gets_h200_hint():
    with pytest.raises(click.ClickException, match="h200"):
        validate_flavor("h100")


def test_grouped_covers_flat_list():
    grouped = gpu_flavors_grouped()
    flat = sorted(v for vs in grouped.values() for v in vs)
    assert flat == GPU_FLAVORS
    assert "H200" in grouped and "h200" in grouped["H200"]
