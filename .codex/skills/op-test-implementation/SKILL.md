---
name: op-test-implementation
description: Use when adding or revising tests for a DinoML `.work/op/` implementation pack. This skill keeps op test work repo-local and consistent across IR, CPU, ROCm, and CUDA surfaces, while improving on older patterns by preferring shared parity helpers, explicit contract-focused tests, and honest backend-claim boundaries.
---

# Op Test Implementation

Use this skill for the test-authoring part of a DinoML op pack.

This skill is narrower than the full implementation workflow. It does not decide pack scope or backend architecture. It turns the pack's stated test and verification contract into the smallest honest test set that exercises the changed path.

## Read first

Read the pack-local files that define the test contract:

- the pack tests and acceptance file
- the verification commands file
- any backend task files whose support is being claimed

Then read only the nearest local tests and helpers needed to place the new coverage cleanly.

Prefer examples from the same family or surface first:

- `tests/ir/` for frontend validation and shape contract
- `tests/cpu/` for native CPU runtime parity
- `tests/rocm/` for local ROCm parity
- `tests/cuda/` for CUDA parity
- nearby shared helper modules under `tests/` when multiple surfaces need the same cases or oracle

Do not cargo-cult the first older test you find. Use prior ops as pattern input, not as a rulebook.

## Required workflow

### 1) Extract the required test surfaces

Before editing, write down internally:

- which surfaces the pack actually requires: frontend validation, reference parity, CPU runtime parity, ROCm runtime parity, CUDA runtime parity, lowering/render checks, scaffold/manifest checks
- which surfaces are explicitly non-scope
- what the parity oracle is
- whether duplicate indices, dynamic shapes, multiple outputs, or backend-visible generated sources need explicit coverage

Do not add broad extra matrices just because a nearby op did.

### 2) Choose the file layout intentionally

Prefer this layout when multiple runtime surfaces share the same spec builder, inputs, or oracle:

- `tests/<op>_parity.py` for shared case data, trace helpers, randomized inputs, dtype tolerances, and the external oracle
- `tests/ir/test_<op>.py` for frontend validation, output shape checks, shape-spec propagation, and the narrowest semantic/reference checks
- `tests/cpu/test_<op>_parity.py` for compile/load/run CPU parity
- `tests/rocm/test_rocm_<op>_parity.py` for local ROCm parity
- `tests/cuda/test_cuda_<op>_parity.py` for CUDA parity

Use a family file instead of a dedicated per-op file only when the ops genuinely share one contract and one reader would naturally review them together.

Avoid copying large case tables or oracle code into every backend test file.

### 3) Keep each test surface narrow

#### IR tests

Use IR tests for:

- accepted frontend tracing
- rejected argument, dtype, rank, or shape combinations
- output shape and shape-spec assertions
- explicit edge semantics that are cheapest to prove before runtime

If a reference interpreter run is the narrowest semantic check, keep it here or in a closely adjacent helper-backed test.

#### CPU parity tests

Use CPU parity tests for native artifact execution:

- compile the traced spec to `dml.Target("cpu")`
- load the artifact
- create a session
- run `session.run_numpy(...)`
- compare against the oracle with dtype-appropriate tolerances

Do not treat `reference_numpy` as CPU runtime evidence.

#### ROCm and CUDA parity tests

Use backend tests for actual backend runtime parity, not for aspiration.

- keep toolchain and device probes explicit
- reuse the shared case/oracle helper where practical
- assert generated-kernel metadata only when it protects a real backend contract for that op family

Do not replace backend runtime parity with render-only evidence when the pack claims runtime support.

### 4) Improve older patterns where the improvement is clear

Prefer these refinements even if some older ops are looser:

- factor reusable cases, random inputs, spec builders, and oracles into one helper module when more than one surface needs them
- keep frontend/contract checks separate from runtime parity wrappers
- use `assert_array_equal` for exact integer/index/class outputs and `assert_allclose` for floating outputs
- only add scaffold or manifest assertions when they protect a contract that has broken before or is easy to regress silently
- keep backend-specific skips local to backend tests instead of leaking them into shared helpers
- avoid mixing unrelated ops into an umbrella file just because a nearby file already exists

Do not add source-render or manifest checks as decoration.

### 5) Name tests by the surface they prove

Good examples:

- `test_index_add_rejects_non_integer_index_dtype`
- `test_cpu_index_add_parity`
- `test_rocm_index_add_parity`

Avoid names that blur frontend validation, reference execution, and runtime parity into one claim.

### 6) Validate in pack order

Run the narrowest discriminating tests first.

Keep these surfaces distinct in your own reasoning and in your report:

- frontend validation
- reference parity
- CPU runtime parity
- ROCm runtime parity
- CUDA runtime parity
- lowering/render/scaffold checks

One passing surface does not certify the others.

## Common failure modes

Block these explicitly:

- duplicating parity fixtures across CPU, ROCm, and CUDA files
- putting runtime-only coverage into `tests/ir/`
- calling render-only checks "backend support"
- adding GPU files with no device/toolchain gating
- claiming backend support without a runtime parity test on that backend
- growing a giant dtype or shape matrix with no pack requirement
- relying only on unique-index or happy-path examples when the pack calls out hard semantics such as duplicate accumulation

## Reporting style

When summarizing the test work, say:

- which test surfaces were added or updated
- which helper modules were introduced or reused
- which backend claims are actually validated
- which required surfaces remain pending

Be explicit when a backend file only checks lowering or manifest shape rather than runtime execution.
