"""Compare two unrelated elementwise kernels by dispatching per-kernel.

The `quick` command requires both kernels to expose the same function name,
so to compare kernels with different APIs we use a bench file and branch
on which method exists on the loaded kernel.
"""

import torch

from kernels_bench import Bench, TensorSpec

bench = Bench(
    name="elementwise_gelu_vs_relu",
    inputs=[TensorSpec("x", shape=(4096, 4096), dtype=torch.float32)],
    outputs=[TensorSpec("y", shape=(4096, 4096), dtype=torch.float32, role="output")],
)


@bench.fn
def forward(kernel, x, y):
    if hasattr(kernel, "gelu_fast"):
        kernel.gelu_fast(y, x)
    elif hasattr(kernel, "relu"):
        kernel.relu(y, x)
    else:
        raise RuntimeError(f"kernel {kernel!r} has no known elementwise op")
