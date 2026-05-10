# Provider Contract

A provider is a reusable implementation family such as CUTLASS GEMM, CUTLASS BMM, GGUF, CUB reductions, FlashAttention, cuDNN, CK, or MIOpen.

## Required Provider Pieces

A mature provider should define:

- provider name
- supported backends
- supported dtypes
- supported layouts
- candidate metadata schema
- support-library build/cache strategy
- source/build provenance
- runtime launch ABI
- profiler ABI, if performance-sensitive
- profile workload generation
- profile report fields
- execution-plan application
- generated lowering path
- validation tests
- docs/checklist status

## Provider Maturity Levels

### Scaffold

Metadata or frontend exists, but no runtime path.

### Bounded Runtime

Runtime path exists for a constrained set of dtypes/shapes/layouts.

### Profiled

Provider emits candidates and profiling workloads.

### Execution-Plan Integrated

Profile selections can be applied to manifests and generated code.

### Mature

Provider has tests, docs, cache/provenance, failure modes, and known limits.
