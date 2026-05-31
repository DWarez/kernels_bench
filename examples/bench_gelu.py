"""Benchmark GeLU activation across different tensor sizes.

This example sweeps over multiple M dimensions while keeping N fixed,
showing how kernel performance scales with input size.

Usage:
    kernels-bench run examples/bench_gelu.py \
        -k kernels-community/activation \
        -w 10 -n 100
"""

import torch

from kernels_bench import Bench, TensorSpec

bench = Bench(
    name="gelu_activation",
    inputs=[
        TensorSpec("x", shape=("M", "N"), dtype=torch.float16),
    ],
    outputs=[
        TensorSpec("y", shape=("M", "N"), dtype=torch.float16, role="output"),
    ],
    params={"M": [1024, 2048, 4096], "N": [1024]},
)


@bench.fn
def forward(kernel, x, y):
    kernel.gelu_fast(y, x)


# Optional reference: plain PyTorch, used as a speed baseline in the table and
# (with --validate) as a correctness oracle. Takes the inputs, returns the result.
@bench.ref
def reference(x):
    return torch.nn.functional.gelu(x, approximate="tanh")
