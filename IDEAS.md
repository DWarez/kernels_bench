# Feature Ideas

Backlog of features worth considering. Ranked roughly by value/effort.

## 1. Roofline / throughput numbers
Convert times to GB/s or TFLOPS given tensor shapes + dtype. Turns "2.3 ms"
into "60% of peak bandwidth." Needs a way to declare bytes-moved and/or
flops-per-call in the bench spec.

## 2. Sweep mode
Run a kernel across a grid of shapes (e.g. `M=[512,1024,2048]`) and emit a
table/CSV. The param-group plumbing already exists in the runner — this is
mostly a CLI + display change.

## 3. Baseline-relative reporting
`--baseline <kernel_id>` flag. Express other kernels as speedup ratios
against the baseline in both display and JSON, instead of just "fastest in
group."

## 4. Result diffing across runs
`kernels-bench diff old.json new.json` to compare two saved JSON outputs.
Useful for regression checks ("did this kernel update slow anything down?").

## 5. Energy / power sampling (CUDA only)
Sample `nvmlDeviceGetPowerUsage` alongside the existing util collector.
Niche but differentiating.
