# Contributing to DinoML v2

This repo is intentionally small, but the boundaries matter. Keep changes
boring, inspectable, and easy to validate before optimizing them.

## Local Workflow

```sh
pip install -e ".[dev]"
python -m pytest -q
```

Generated artifacts, benchmark output, support-library builds, and profile data
belong under `tmp/`, `build/`, or the DinoML support cache. They should not be
committed.

## Adding an Op

1. Add semantic registration under `src/dinoml/ops/`.
2. Add or update validation/pass behavior under `src/dinoml/passes/` if the op
   has special shape, dtype, or graph constraints.
3. Add backend lowering under `src/dinoml/lowering/ops/`.
4. Put reusable CPU/CUDA code in `kernels/` when it can be shared across
   models. Keep model-specific generated code in the artifact wrapper.
5. Add CPU reference tests and backend runtime tests. CUDA kernels should also
   be benchmarked against an appropriate PyTorch or NumPy reference.
6. Update `docs/op_porting_checklist.md` with the new support level and known
   gaps.

Prefer one semantic public op with backend variants over many public names that
only encode layout or epilogue details. Layout-specialized kernels can still be
selected by the backend/profiler.

## Codegen Rules

- Generated module wrappers should stay small: metadata loading, constants,
  pointer binding, shape buffers, workspace/session allocation, and launch
  order.
- Reusable kernels, math helpers, accessors, and profilers should live outside
  the model wrapper.
- Model-specific generated kernels may live in artifacts, but should move toward
  per-op files keyed by normalized signatures rather than one growing wrapper
  source.
- Stable generated names should be based on codegen signatures, not graph node
  ids or temporary names.
- If generated source is useful for review, write it under the artifact
  `debug/generated_src/` tree or under `tmp/`; do not commit generated artifacts.

## Performance Work

Every new CUDA kernel path should have a benchmark before it is treated as
ready. Start with safe kernels and clear references, then add profiler
candidates or library-backed paths such as CUTLASS, CK, CUB, cuDNN/MIOpen, or
oneDNN where they fit.
