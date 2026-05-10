# Agent Operating Loop

This repository uses docs as project memory and tests as truth.

## Main Loop

1. Read current state:
   - `README.md`
   - `docs/architecture.md`
   - `agents/plans/v1_gap_audit.md`
   - `agents/plans/op_porting_checklist.md`
   - `agents/PROVIDER_CONTRACT.md`
   - `agents/OP_ADMISSION.md`
   - `agents/INVARIANTS.md`
   - `agents/CURRENT_FOCUS.md`
   - `agents/BLOCKED_OR_DEFERRED.md`

2. Inspect recent commits and current tests.

3. Pick one bounded slice of work.

4. Use subagents to investigate, implement, and review.

5. Validate with targeted tests.

6. Update docs/checklists.

7. Commit.

8. Update Codex Progress.

9. Repeat.

## Work Selection Rules

Prefer:
- stabilizing recent work
- closing known v1 gaps
- improving provider maturity
- adding tests for newly landed surface
- small complete slices

Avoid:
- broad unvalidated op expansion
- new architecture exceptions
- provider-specific shortcuts that bypass manifests or execution plans
- stale documentation
