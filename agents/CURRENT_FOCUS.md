# Current Focus

## Primary Focus

- Human-directed feature target: make a bounded dequantize-to-GEMM runtime path
  work for GGUF-backed CUDA weights. The useful product shape is: keep the
  runtime-loaded weight in quantized/encoded form, dequantize on GPU just before
  the GEMM that consumes it, use either a session-owned max-size dequantized
  scratch buffer or a clearly scoped temporary allocation, then launch the
  existing dense GEMM path from that dequantized buffer.
- Keep this artifact-visible and narrow. Prefer a single CUDA GEMM family/weight
  operand slice with an explicit GGUF materialization/residency policy over a
  hidden general offload scheduler. Do not add prefetch/eviction/grouped
  offload policy until the scratch/dequant/GEMM contract is proven.

## Near-Term Priorities

- GGUF constant storage, runtime materialization, CUDA dequant, and future offload policy.
- CUTLASS parity: GEMM/BMM profile loop, execution plans, guarded dispatch, split-K, alignment, epilogues.
- Weight offloading: constant residency, load/unload/reload state, group/layer/leaf-level policies.
- Op porting: bounded v1 primitive coverage with tests and checklist updates.
- v1 gap closure: symbolic shapes, memory planning, runtime ABI, profiling/cache behavior.
- Stabilization: audit newly ported ops, classify maturity, improve tests, reduce duplicate patterns.

## Preferred Next Work
