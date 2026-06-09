"""Derive the torch build to install on an HF Job from kernels' published variants.

A prebuilt kernel only loads when the job's torch matches one of the
``(torch-version, CUDA)`` pairs the kernel was built for *and* that CUDA is no
newer than the instance driver. Rather than hard-pinning torch and hoping a
build exists, we read each kernel's build matrix from the Hub (the `kernels`
library already enumerates and parses these), intersect the pairs across all
kernels in the run, and pick the newest torch at a driver-safe CUDA ceiling.

The CUDA ceiling collapses the per-instance driver question: cu128 (and below)
runs on every current HF Jobs GPU driver via backward compatibility, so we never
need a fragile flavor→driver table. Override the whole decision per run with the
``KB_REMOTE_TORCH`` / ``KB_REMOTE_TORCH_CUDA`` env vars (handled in ``launch``).
"""

from __future__ import annotations

from functools import cache

from huggingface_hub import HfApi
from kernels.variants import CUDA, ArchVariant, Torch, get_variants
from packaging.version import Version

# Highest CUDA we install without being told the instance can handle more.
# cu128 == CUDA 12.8; newer drivers run it via backward compat.
DEFAULT_CUDA_CEILING = Version("12.8")

# HF Jobs GPU instances are x86_64 Linux; we match cxx11-ABI torch builds (the
# modern default). Older cxx98 / non-Linux / aarch64 variants are ignored.
_PLATFORM = "x86_64"
_OS = "linux"

# A (torch_version, cuda_version) build the kernel ships for our platform.
Pair = tuple[Version, Version]


class NoCompatibleTorch(Exception):
    """No torch/CUDA build is common to all kernels within the CUDA ceiling."""


def cuda_tag(version: Version) -> str:
    """``Version('12.8')`` -> ``'cu128'`` (the PyTorch wheel-index suffix)."""
    return f"cu{version.major}{version.minor}"


def torch_spec(version: Version) -> str:
    """``Version('2.11')`` -> ``'torch==2.11.*'`` (pins the minor, frees the patch)."""
    return f"torch=={version.major}.{version.minor}.*"


def cuda_from_tag(tag: str) -> Version:
    """``'cu128'`` -> ``Version('12.8')`` (inverse of :func:`cuda_tag`)."""
    digits = tag.removeprefix("cu")
    return Version(f"{digits[:-1]}.{digits[-1]}")


def _pairs(variants: list) -> set[Pair]:
    """Extract (torch, CUDA) pairs for our platform from parsed kernel variants."""
    out: set[Pair] = set()
    for v in variants:
        if not isinstance(v, ArchVariant):
            continue  # skip noarch / torch-stable-abi (handled by overrides for now)
        fw, arch = v.framework, v.arch
        if not isinstance(fw, Torch) or not fw.cxx11_abi:
            continue
        if not isinstance(arch.backend, CUDA):
            continue
        if arch.platform != _PLATFORM or arch.os != _OS:
            continue
        out.add((fw.version, arch.backend.version))
    return out


def _pick(
    pairs_per_kernel: list[set[Pair]],
    *,
    forced_cuda: Version | None,
    ceiling: Version,
) -> Pair | None:
    """Pick the newest (torch, CUDA) common to all constraining kernels.

    Returns ``None`` when no kernel constrains the choice (none publish CUDA
    builds), so the caller can fall back to a default. Raises
    :class:`NoCompatibleTorch` when kernels *do* constrain but share no build at
    or below the ceiling (or at ``forced_cuda``).
    """
    sets = [p for p in pairs_per_kernel if p]
    if not sets:
        return None
    common = set.intersection(*sets)
    if forced_cuda is not None:
        candidates = {(t, c) for t, c in common if c == forced_cuda}
    else:
        candidates = {(t, c) for t, c in common if c <= ceiling}
    if not candidates:
        raise NoCompatibleTorch
    return max(candidates)  # Versions compare: newest torch, then highest CUDA


@cache
def _fetch_variants(repo_id: str, revision: str) -> tuple:
    return tuple(get_variants(HfApi(), repo_id=repo_id, revision=revision))


def resolve(
    kernels: list[str],
    *,
    forced_cuda: Version | None = None,
    ceiling: Version = DEFAULT_CUDA_CEILING,
) -> tuple[str, str] | None:
    """Resolve ``(torch_spec, cuda_tag)`` for a run from the kernels' build matrices.

    ``kernels`` are ``repo[@revision]`` specs. Returns ``None`` if no kernel
    publishes CUDA builds (caller uses its default). Raises
    :class:`NoCompatibleTorch` (with a per-kernel availability summary) when the
    kernels can't share a build within the ceiling.
    """
    from kernels_bench.bench import split_kernel_ref

    per_kernel: list[set[Pair]] = []
    available: dict[str, set[Pair]] = {}
    for spec in kernels:
        repo, revision = split_kernel_ref(spec)
        try:
            variants = _fetch_variants(repo, revision or "main")
        except Exception:
            continue  # can't introspect this repo — leave it unconstrained
        pairs = _pairs(list(variants))
        available[spec] = pairs
        per_kernel.append(pairs)

    try:
        picked = _pick(per_kernel, forced_cuda=forced_cuda, ceiling=ceiling)
    except NoCompatibleTorch:
        raise NoCompatibleTorch(_explain(available, forced_cuda, ceiling)) from None
    if picked is None:
        return None
    torch_version, cuda_version = picked
    return torch_spec(torch_version), cuda_tag(cuda_version)


def _explain(available: dict[str, set[Pair]], forced_cuda: Version | None, ceiling: Version) -> str:
    """Human-readable 'why no common build' message listing each kernel's builds."""
    limit = f"== {cuda_tag(forced_cuda)}" if forced_cuda else f"<= {cuda_tag(ceiling)}"
    lines = [f"No torch/CUDA build common to all kernels with CUDA {limit}.", "Available builds:"]
    for spec, pairs in available.items():
        shown = (
            ", ".join(f"torch {t.major}.{t.minor}/{cuda_tag(c)}" for t, c in sorted(pairs))
            or "(no cxx11 x86_64-linux CUDA builds)"
        )
        lines.append(f"  {spec}: {shown}")
    lines.append(
        "Override with KB_REMOTE_TORCH_CUDA=cuXXX (on an instance whose driver "
        "supports it) and/or KB_REMOTE_TORCH='torch==X.Y.*'."
    )
    return "\n".join(lines)
