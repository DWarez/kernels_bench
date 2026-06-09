"""Remote benchmarking on HuggingFace Jobs.

Runs the existing benchmark engine, unchanged, on an ephemeral HF Jobs GPU so
users can compare kernels on hardware they don't physically have. The local
side only orchestrates the job and renders the results that come back.
"""

from kernels_bench.remote.flavors import GPU_FLAVORS, gpu_flavors_grouped, validate_flavor
from kernels_bench.remote.launch import run_remote
from kernels_bench.remote.request import RemoteRequest

__all__ = [
    "GPU_FLAVORS",
    "RemoteRequest",
    "gpu_flavors_grouped",
    "run_remote",
    "validate_flavor",
]
