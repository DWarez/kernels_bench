"""Example: using kernels-bench as a Python library.

This shows how to define and run benchmarks programmatically,
access results, and export to JSON — without using the CLI.

Usage:
    python examples/bench_as_library.py
"""

import json

import torch

from kernels_bench import Bench, TensorSpec, print_results

bench = Bench(
    name="gelu_library_example",
    inputs=[
        TensorSpec("x", shape=(2048, 1024), dtype=torch.float16),
    ],
    outputs=[
        TensorSpec("y", shape=(2048, 1024), dtype=torch.float16, role="output"),
    ],
)


@bench.fn
def forward(kernel, x, y):
    kernel.gelu_fast(y, x)


if __name__ == "__main__":
    # Run the benchmark
    result = bench.run(
        kernels=["kernels-community/activation"],
        warmup=5,
        iterations=50,
    )

    # Display results
    print_results(result)

    # Access results programmatically
    for kr in result.kernel_results:
        print(f"\n{kr.kernel_id}: median={kr.median_ms:.3f}ms, mean={kr.mean_ms:.3f}ms")

    # Export to JSON
    with open("library_results.json", "w") as f:
        json.dump(result.to_dict(), indent=2, fp=f)
    print("\nResults saved to library_results.json")
