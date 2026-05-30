# ROCm Notes

## Benchmark Timing

ROCm artifact benchmarks should use the generated native `dino_session_benchmark`
entry point. That path runs inside the compiled module; Python only calls into it
once and does not contribute per-iteration launch overhead.

By default, GPU artifacts try to capture the benchmarked run into a HIP graph and
measure graph replay with device events. This is the preferred number for CLIP
and other launch-heavy graphs because it measures the kernel sequence without
per-launch host overhead dominating the result.

Useful environment variables:

- `DINOML_REQUIRE_BENCHMARK_GRAPH=1`: fail instead of silently falling back when
  graph capture/replay is unavailable. Use this for CLIP comparisons.
- `DINOML_BENCHMARK_WALL_SYNC=1`: measure each iteration with host wall time and
  stream synchronization. This intentionally includes host launch/sync overhead
  and is useful for diagnosing runtime behavior, not for kernel-sequence
  throughput comparisons.
- `DINOML_PROFILE_RUN=1`: print per-launch device event timing from normal
  `run_numpy`/session execution. This disables benchmark graph capture for that
  run.
- `DINOML_PROFILE_RUN_VERBOSE=1`: print each launch label and time, not only
  per-op totals.
- `DINOML_PROFILE_RUN_SECTIONS=name:start:end,...`: additionally time launch
  ranges by generated launch index.

ROCm device events are created with `hipEventDisableSystemFence`. The default
system-fence behavior can add avoidable synchronization cost to small-kernel
benchmarks and made CLIP timing look worse than the underlying kernel sequence.

The generated ROCm/CUDA module owns a non-blocking stream unless the runtime
session is given an external stream. Graph capture is attempted only for owned
streams and only when profiling is not enabled.

## CLIP Performance Checks

For CLIP tower comparisons, compile the text and vision towers separately when
narrowing a regression. Full CLIP forward timing can hide which tower is slow,
and the vision tail has historically been sensitive to extra launches around the
pooled CLS token.

Recent CLIP-sensitive lowering includes:

- packed QKV projection plus CK FlashAttention for fp16 ROCm traces;
- `add_layer_norm` fusion for residual add plus affine LayerNorm;
- contiguous `dynamic_slice` as view metadata with pointer offsets;
- sliced add+LayerNorm fusion for the final vision pooled-token tail;
- direct dense/suffix indexing in fused elementwise kernels;
- CK GEMM candidate preferences for CLIP small-M shapes.

When comparing to PyTorch, make sure the PyTorch benchmark is also using device
events around warmed execution, not Python wall timing. Earlier CLIP reports had
an inaccurate torch baseline, so fresh torch medians should be recorded beside
the DinoML graph-replay medians.
