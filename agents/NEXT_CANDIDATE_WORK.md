# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

### 1. Stabilize recently added ops

Rationale: many bounded ops landed quickly; classify maturity and fill tests.

### 2. CUTLASS provider audit

Rationale: ensure selected candidates are visible in manifests and generated code for GEMM/BMM, including guarded dispatch and split-K.

### 3. GGUF constant lifecycle

Rationale: encoded constants now exist; next step is clear runtime state, planning, and selective materialization.

### 4. Weight offload design

Rationale: ABI/runtime constant unload support enables offload policies.

### 5. Region/repeated-block design

Rationale: preserves model structure and enables block-level profiling/offload.
