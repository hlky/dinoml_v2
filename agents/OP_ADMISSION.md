# Op Admission Rules

Before adding a new public op, answer:

1. What v1 behavior does this map to?
2. Is it a real compiled op, metadata-only op, or frontend helper?
3. What are the bounded limits?
4. What dtypes are supported?
5. What shape/rank/layout constraints exist?
6. Is dynamic shape support real or deferred?
7. Is CPU reference available?
8. Is CUDA/runtime lowering available?
9. Does it need profiling?
10. What tests prove it works?
11. What docs/checklist entries must change?

## Completion Labels

Use these labels in checklists:

- `frontend-only`
- `metadata-only`
- `bounded-cpu`
- `bounded-cuda`
- `provider-backed`
- `profile-integrated`
- `complete-bounded`
- `deferred`
