# kernels-bench

A benchmarking tool for [HuggingFace Kernels](https://huggingface.co/docs/kernels/index). Compare CUDA kernel performance on your hardware — from the CLI or as a Python library.

## Install

Requires Python 3.12+ and a CUDA GPU.

```bash
pip install git+https://github.com/dwarez/kernels_bench.git
```

Or for development:

```bash
git clone https://github.com/dwarez/kernels_bench.git
cd kernels_bench
uv sync
```

## Quick start

### Discover kernel functions

```bash
kernels-bench list kernels-community/activation
```

### Benchmark from the CLI (no code needed)

```bash
kernels-bench quick \
  -k kernels-community/activation \
  --fn gelu_fast \
  --arg y:1024,1024:float16:output \
  --arg x:1024,1024:float16:input \
  -w 10 -n 100
```

Arguments are passed to the kernel function in the order you specify them. The format is `name:shape:dtype:role` where dtype (default: `float16`) and role (default: `input`) are optional.

### Compare multiple kernels

```bash
kernels-bench quick \
  -k kernels-community/activation,another-org/activation \
  --fn gelu_fast \
  --arg y:1024,1024:float16:output \
  --arg x:1024,1024:float16:input \
  --validate \
  -n 100
```

The `--validate` flag runs each kernel once on the same input data and checks that outputs match across kernels (using `torch.allclose`). Tolerance is configurable with `--atol` and `--rtol`.

### Heavier workloads

Any kernel on the Hub works — here's Flash Attention 2 with sequence length 8k:

```bash
kernels-bench quick \
  -k kernels-community/flash-attn \
  --fn flash_attn_func \
  --arg q:4,8192,32,128:float16 \
  --arg k:4,8192,32,128:float16 \
  --arg v:4,8192,32,128:float16 \
  -w 10 -n 100
```

### Custom benchmarks with a bench file

For more control — parameter sweeps, custom logic, multiple steps — write a bench file:

```python
# bench_gelu.py
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
```

Then run it:

```bash
kernels-bench run bench_gelu.py \
  -k kernels-community/activation \
  -w 10 -n 100
```

Symbolic dimensions (strings like `"M"`, `"N"` in the shape) are resolved from `params`, producing a benchmark for every combination.

### Use as a Python library

```python
from kernels_bench import Bench, TensorSpec, print_results

bench = Bench(
    name="my_benchmark",
    inputs=[TensorSpec("x", shape=(2048, 1024), dtype=torch.float16)],
    outputs=[TensorSpec("y", shape=(2048, 1024), dtype=torch.float16, role="output")],
)

@bench.fn
def forward(kernel, x, y):
    kernel.gelu_fast(y, x)

result = bench.run(
    kernels=["kernels-community/activation"],
    warmup=10,
    iterations=100,
    validate=True,  # check correctness when comparing multiple kernels
)

print_results(result)

# Access results programmatically
for kr in result.kernel_results:
    print(f"{kr.kernel_id}: {kr.median_ms:.3f} ms")

# Export to JSON
import json
json.dump(result.to_dict(), open("results.json", "w"), indent=2)
```

## Output

Results are displayed in a box-drawing table showing timing, comparison bars, and GPU info:

```
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ kernels-bench v0.1.0 ┐
│                   KERNEL BENCHMARK RESULTS                                  │
│                      "gelu_activation"                                      │
│          NVIDIA GeForce RTX 4090 | CUDA 12.8 | 24.0 GB                      │
│                torch 2.11.0 | python 3.12.13                                │
│               2 KERNELS x 3 PARAM SETS                                      │
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                      PARAMS: M=1024, N=1024                                 │
├────────────────────────────┬────────────────────────────────────────────────┤
│ kernel-a                   │ 0.100 ms  ████████████████████████░░░░░░░░░░░░ │
│                            │ mean=0.100  std=0.002  min=0.097  max=0.102    │
│                            │ peak_mem=64.0 MB  util=87% (peak 99%)          │
│                            │ FASTEST                                        │
├────────────────────────────┼────────────────────────────────────────────────┤
│ kernel-b                   │ 0.128 ms  ██████████████████████████████████   │
│                            │ mean=0.128  std=0.002  min=0.124  max=0.131    │
│                            │ peak_mem=64.0 MB  util=72% (peak 94%)          │
│                            │ 1.28x slower  ·  util 72% (fastest: 87%)       │
└────────────────────────────┴────────────────────────────────────────────────┘
```

When `--validate` is used, a validation section appears before the timing results showing PASS/FAIL for each kernel pair with max absolute/relative differences.

### Metrics

On CUDA, each run automatically collects:

- **`peak_mem`** — peak device memory allocated during the timed window (`torch.cuda.max_memory_allocated`).
- **`util`** — mean and peak GPU utilization (SM-busy %) sampled via NVML while the kernel runs. Reported only when the window is long enough to collect ≥3 samples.

When multiple kernels are compared, the slowdown line also shows the slower kernel's util next to the fastest's. That makes it easy to tell whether a slower kernel is *inefficient* (lower util) or just *doing more work* (similar/higher util than the fastest).

Pass `--no-metrics` to skip collection entirely — useful if you want zero overhead or are troubleshooting NVML.

Metrics for MPS backends are not yet collected; those fields stay `null` in the JSON export.

## JSON export

Add `-o results.json` to any command to save results:

```bash
kernels-bench quick \
  -k kernels-community/activation \
  --fn gelu_fast \
  --arg y:1024,1024:float16:output \
  --arg x:1024,1024:float16:input \
  -o results.json
```

The JSON includes device info, timing stats, raw per-iteration times, and validation results.

## CLI reference

```
kernels-bench list <kernel-id>              # list functions in a kernel
kernels-bench quick [options]               # benchmark without a bench file
kernels-bench run <bench-file> [options]    # benchmark with a bench file
```

### Common options

| Flag | Description |
|------|-------------|
| `-k, --kernels` | Comma-separated kernel repo IDs |
| `-w, --warmup` | Warmup iterations (default: 10) |
| `-n, --iterations` | Timed iterations (default: 100) |
| `-o, --output` | Save results to JSON file |
| `--validate` | Check output correctness across kernels |
| `--atol` | Absolute tolerance for validation (default: 1e-3) |
| `--rtol` | Relative tolerance for validation (default: 1e-3) |
| `--no-metrics` | Skip collecting peak memory and GPU utilization |

### `quick` specific options

| Flag | Description |
|------|-------------|
| `-f, --fn` | Kernel function name to benchmark |
| `-a, --arg` | Tensor arg: `name:shape:dtype:role` (repeatable) |

## Supported dtypes

`float16` / `fp16`, `bfloat16` / `bf16`, `float32` / `fp32`, `float8_e4m3fn`, `float8_e5m2`

## Development

```bash
uv sync                          # install deps
uv run pytest tests/ -v          # run tests
uv run ruff check src/           # lint
uv run ruff format src/          # format
uv run ty check src/             # type check
```

## Credits

The CLI output style matches that of [hf-mem](https://github.com/alvarobartt/hf-mem)

## License

MIT
