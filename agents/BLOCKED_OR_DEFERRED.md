# Blocked or Deferred Work

These are intentionally deferred unless explicitly requested.

## Deferred

- Full UI implementation.
- Huge model importers.
- Fused GGUF GEMM beyond experiments.
- Arbitrary non-dense tensor accessor GEMM.
- Full jagged/ragged tensor support.
- Full conv/attention provider expansion before CUTLASS/GGUF stabilization.
- Large op sweeps without maturity audit.

## Needs Design First

- Weight offloading policies.
- Region/repeated-block lowering.
- Quantized provider architecture beyond GGUF dense materialization.
- Multi-GPU/sharding.
