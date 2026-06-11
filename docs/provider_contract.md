# Provider Contract

A provider is a reusable implementation family such as CUTLASS GEMM, CUTLASS BMM, CK, FlashAttention, GGUF runtime dequant, or a future backend library integration.

## Required Provider Pieces

A mature provider should define:

- provider name
- supported backends
- supported dtypes
- supported layouts or shape limits
- candidate metadata schema
- support-library build and cache strategy
- source and build provenance
- runtime launch ABI
- profiler ABI, if the provider is performance-sensitive
- profile workload generation
- profile report fields
- execution-plan application
- generated lowering path
- validation tests
- docs and checklist status

## Maturity Levels

### Scaffold

Metadata or frontend registration exists, but no admitted runtime path.

### Bounded Runtime

Runtime path exists for a constrained set of dtypes, shapes, layouts, or targets.

### Profiled

The provider emits candidates and profiling workloads.

### Execution-Plan Integrated

Profile selections can be applied to manifests and generated code.

### Mature

The provider has tests, docs, cache and provenance handling, explicit failure modes, and known limits.

## Rules

- Do not treat frontend registration alone as provider support.
- Do not bypass manifests, profile reports, execution plans, or generated lowering with hidden mutable state.
- If a provider is bounded, state the bounds plainly in code, tests, and docs.
