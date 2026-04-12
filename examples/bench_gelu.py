"""Example: benchmark GeLU activation kernels."""

import torch

from kernels_bench import Bench, TensorSpec

bench = Bench(
    name="gelu_activation",
    inputs=[
        TensorSpec("x", shape=("M", "N"), dtype=torch.float16),
    ],
    outputs=[
        TensorSpec("y", shape=("M", "N"), dtype=torch.float16),
    ],
    params={"M": [1024, 2048, 4096], "N": [1024]},
)


@bench.fn
def forward(kernel, x, y):
    kernel.gelu_fast(y, x)
