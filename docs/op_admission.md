# Op Admission

Before adding or broadening a public op, answer:

1. What v1 behavior does this map to?
2. Is it a real compiled op, a metadata-only op, or a frontend helper?
3. What are the bounded limits?
4. What dtypes are supported?
5. What shape, rank, layout, or dynamic-shape constraints exist?
6. Is dynamic shape support real or deferred?
7. Is a CPU or reference validation path available?
8. Is lowering available for the intended backends?
9. Does it need profiling or execution-plan support?
10. What tests prove it works?
11. What docs or checklists must change?

## Completion Labels

Use these labels consistently in checklists or planning docs:

- `frontend-only`
- `metadata-only`
- `bounded-cpu`
- `bounded-cuda`
- `bounded-rocm`
- `provider-backed`
- `profile-integrated`
- `complete-bounded`
- `deferred`

## Rules

- An op is not complete just because the frontend constructor exists.
- If the runtime behavior is intentionally bounded, keep the unsupported fences explicit.
- Prefer stabilizing and validating existing surface area before expanding public op coverage.
