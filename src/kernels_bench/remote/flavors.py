"""GPU hardware flavors available on HuggingFace Jobs.

The set of flavors is derived from ``huggingface_hub.SpaceHardware`` (the same
enum HF Jobs uses for the ``flavor`` argument) and filtered to GPU classes, so
it tracks the library rather than going stale in a hardcoded list. Non-GPU
flavors (CPU, Inferentia, etc.) are excluded — this tool only benchmarks GPUs.
"""

from __future__ import annotations

import click
from huggingface_hub import SpaceHardware

# GPU classes in rough capability order, each a (label, match-token) pair. A
# flavor belongs to a class when the token appears in its value string
# ("a10g-small", "h200x4", ...). Tokens are an allowlist: anything that matches
# none of them (cpu-*, sprx8, inf2x6) is treated as non-GPU and dropped.
_GPU_CLASSES: list[tuple[str, str]] = [
    ("T4", "t4"),
    ("L4", "l4x"),
    ("L40S", "l40s"),
    ("A10G", "a10g"),
    ("A100", "a100"),
    ("H200", "h200"),
]


def _classify(value: str) -> str | None:
    for label, token in _GPU_CLASSES:
        if token in value:
            return label
    return None


def gpu_flavors_grouped() -> dict[str, list[str]]:
    """Return GPU flavor strings grouped by class label, in capability order."""
    groups: dict[str, list[str]] = {label: [] for label, _ in _GPU_CLASSES}
    for hw in SpaceHardware:
        label = _classify(hw.value)
        if label is not None:
            groups[label].append(hw.value)
    return {label: sorted(values) for label, values in groups.items() if values}


# Flat, sorted set of all valid GPU flavor strings.
GPU_FLAVORS: list[str] = sorted(v for values in gpu_flavors_grouped().values() for v in values)


def validate_flavor(name: str) -> str:
    """Validate a GPU flavor string, returning it unchanged if valid.

    Accepts only exact HF flavor names. Unknown names — including ``h100``,
    which HF Jobs does not offer — raise a ``click.ClickException`` listing the
    valid flavors, with a hint that ``h200`` is the single-GPU Hopper option.
    """
    if name in GPU_FLAVORS:
        return name

    hint = ""
    if "h100" in name.lower():
        hint = "\n\nNote: HF Jobs has no H100 — use 'h200' (a newer Hopper card) instead."
    valid = ", ".join(GPU_FLAVORS)
    raise click.ClickException(f"unknown GPU flavor {name!r}. Valid flavors: {valid}{hint}")
