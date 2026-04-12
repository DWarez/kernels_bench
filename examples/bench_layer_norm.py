"""Benchmark Layer Normalization kernels.

Compares kernels-community/layer_norm and kernels-community/triton-layer-norm
across different hidden dimensions, a common sweep for transformer workloads.

Usage:
    kernels-bench run examples/bench_layer_norm.py \
        -k kernels-community/triton-layer-norm \
        -w 10 -n 100
"""

import torch

from kernels_bench import Bench, TensorSpec

bench = Bench(
    name="layer_norm",
    inputs=[
        TensorSpec("x", shape=(128, "D"), dtype=torch.float16),
        TensorSpec("weight", shape=("D",), dtype=torch.float16),
        TensorSpec("bias", shape=("D",), dtype=torch.float16),
    ],
    outputs=[
        TensorSpec("y", shape=(128, "D"), dtype=torch.float16, role="output"),
    ],
    params={"D": [768, 1024, 2048, 4096]},
)


@bench.fn
def forward(kernel, x, weight, bias, y):
    kernel.layer_norm(y, x, weight, bias, 1e-5)
