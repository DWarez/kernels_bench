"""Tests for the remote torch resolver (pure logic — no network).

Variant lists are built from real folder-name strings via ``kernels.parse_variant``
(offline), so we exercise the same parsing the resolver uses in production.
"""

import pytest
from kernels.variants import parse_variant
from packaging.version import Version

from kernels_bench.remote import torch_select
from kernels_bench.remote.torch_select import NoCompatibleTorch, _pairs, _pick, cuda_from_tag

# Trimmed real variant sets (x86_64-linux, cxx11) for the two flash-attn kernels.
FA2 = [
    "torch27-cxx11-cu126-x86_64-linux",
    "torch28-cxx11-cu128-x86_64-linux",
    "torch29-cxx11-cu126-x86_64-linux",
    "torch211-cxx11-cu128-x86_64-linux",
    "torch212-cxx11-cu130-x86_64-linux",
    "torch28-cxx11-cu126-aarch64-linux",  # wrong arch — must be ignored
    "torch29-cxx98-cu126-x86_64-linux",  # wrong abi — must be ignored
]
FA3 = [
    "torch29-cxx11-cu126-x86_64-linux",
    "torch211-cxx11-cu128-x86_64-linux",
    "torch212-cxx11-cu130-x86_64-linux",
]


def _pairs_of(names):
    return _pairs([parse_variant(n) for n in names])


def test_pairs_filters_arch_and_abi():
    pairs = _pairs_of(FA2)
    # aarch64 and cxx98 entries dropped; only x86_64 cxx11 remain.
    assert (Version("2.9"), Version("12.6")) in pairs
    assert all(isinstance(t, Version) and isinstance(c, Version) for t, c in pairs)
    assert len(pairs) == 5


def test_cuda_tag_roundtrip():
    for tag in ["cu126", "cu128", "cu130"]:
        assert torch_select.cuda_tag(cuda_from_tag(tag)) == tag


def test_pick_intersection_under_ceiling():
    # FA2 ∩ FA3 at cu128 ceiling -> newest common is torch 2.11 / cu128.
    spec = _pick([_pairs_of(FA2), _pairs_of(FA3)], forced_cuda=None, ceiling=Version("12.8"))
    assert spec == (Version("2.11"), Version("12.8"))


def test_pick_forced_cuda_overrides_ceiling():
    # Force cu130 -> the only common torch there is 2.12.
    spec = _pick(
        [_pairs_of(FA2), _pairs_of(FA3)], forced_cuda=Version("13.0"), ceiling=Version("12.8")
    )
    assert spec == (Version("2.12"), Version("13.0"))


def test_pick_returns_none_when_no_kernel_constrains():
    assert _pick([set(), set()], forced_cuda=None, ceiling=Version("12.8")) is None


def test_pick_raises_when_no_common_build():
    # One kernel only ships cu130, the other only cu126 -> nothing in common.
    only_130 = {(Version("2.12"), Version("13.0"))}
    only_126 = {(Version("2.9"), Version("12.6"))}
    with pytest.raises(NoCompatibleTorch):
        _pick([only_130, only_126], forced_cuda=None, ceiling=Version("13.0"))


def test_resolve_strings_from_pick():
    # End-to-end string shaping via a monkeypatched fetch (no network).
    spec = _pick([_pairs_of(FA2), _pairs_of(FA3)], forced_cuda=None, ceiling=Version("12.8"))
    t, c = spec
    assert torch_select.torch_spec(t) == "torch==2.11.*"
    assert torch_select.cuda_tag(c) == "cu128"
